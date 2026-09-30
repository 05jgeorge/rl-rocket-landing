import random
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import os
import matplotlib.pyplot as plt

#Create an output folder
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Setting a fixed seed- makes it easier when debugging as we are able to know that any changes made to the code and the corresponding output is not through random luck
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RocketLander1DEnv(gym.Env):
    def __init__(self, initial_x: float = 30.0, initial_y: float = 30.0,
                 gravity: float = -9.81, thrust_x: float = 5.0, thrust_y: float = 22.0):
        super().__init__()

        self.initial_y = initial_y
        self.initial_x = initial_x
        self.gravity = gravity
        self.thrust_x = thrust_x
        self.thrust_y = thrust_y
        self.dt = 0.05
        self.safe_landing_speed_y = 4.0
        self.safe_landing_speed_x = 2.0

        self.I = 50.0
        self.torque_magnitude = 45.0
        self.theta_tip_limit = np.radians(45.0)
        self.theta_upright_limit = np.radians(10.0)
        self.omega_upright_limit = 0.5

        self.theta_spawn_range = np.radians(5.0)
        self.omega_spawn_range = 0.1

        self.step_count = 0
        self.max_steps = 200

        self.y = 0.0
        self.x = 0.0
        self.velocity_y = 0.0
        self.velocity_x = 0.0
        self.theta = 0.0
        self.omega = 0.0
        self.prev_x = 0.0
        self.prev_theta = 0.0

        self.x_spawn_range = 3.0
        self.vx_spawn_range = 0.5


        # reward shaping weights: small penalties for position/rotation error, larger weights for progress (getting closer) to give denser feedback
        self.x_error_weight = 0.08
        self.vx_error_weight = 0.05
        self.horizontal_progress_weight = 4.0

        self.theta_error_weight = 0.5
        self.omega_error_weight = 0.1
        self.angular_progress_weight = 8.0

        # declares the expected range/shape of the state the agent observes each step
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -50.0, -30.0, -10.0, -np.pi, -10.0], dtype=np.float32),
            high=np.array([60.0, 50.0, 30.0, 10.0, np.pi, 10.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )

        # action space: 2 discrete choices (main engine on/off, lateral thrust off/left/right) + 1 continuous value (torque fraction, -1 to 1)
        self.action_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete([2, 3]),
            gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        ))


    # lets spawn difficulty (position/velocity/tilt randomness) be changed after creation — used for curriculum learning
    def set_spawn_range(self, x_range: float, vx_range: float,
                         theta_range: Optional[float] = None,
                         omega_range: Optional[float] = None):
        self.x_spawn_range = x_range
        self.vx_spawn_range = vx_range
        if theta_range is not None:
            self.theta_spawn_range = theta_range
        if omega_range is not None:
            self.omega_spawn_range = omega_range

    def _get_obs(self) -> np.ndarray:
        return np.array([self.y, self.velocity_y, self.x, self.velocity_x,
                          self.theta, self.omega], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0

        self.y = float(self.np_random.uniform(20.0, 35.0))
        self.velocity_y = float(self.np_random.uniform(-3.0, -0.5))
        self.x = float(self.np_random.uniform(-self.x_spawn_range, self.x_spawn_range))
        self.velocity_x = float(self.np_random.uniform(-self.vx_spawn_range, self.vx_spawn_range))
        self.theta = float(self.np_random.uniform(-self.theta_spawn_range, self.theta_spawn_range))
        self.omega = float(self.np_random.uniform(-self.omega_spawn_range, self.omega_spawn_range))
        # store current position/tilt as "previous" so next step can measure progress (getting closer vs further)
        self.prev_x = self.x
        self.prev_theta = self.theta

        # gym requires (obs, info) from reset — info dict left empty since there's nothing to report yet
        return self._get_obs(), {}


    def step(self, action):
        self.step_count += 1

        # action arrives as a flat list [main, lateral, torque] from the caller (policy or scripted pilot), not from action_space itself
        main_idx, lateral_idx, torque_fraction = action
        main_idx = int(main_idx)
        lateral_idx = int(lateral_idx)
        torque_fraction = float(np.clip(torque_fraction, -1.0, 1.0))

        # convert action indices to physics values: main engine on/off, lateral index -> direction (0 none, 1 = +x, 2 = -x)
        main_fire = bool(main_idx)
        lateral = {0: 0, 1: 1, 2: -1}[lateral_idx]

        # main engine only adds thrust if not already rising faster than 1 m/s (stops the rocket flying away)
        engine_accel_y = self.thrust_y if (main_fire and self.velocity_y <= 1.0) else 0.0
        total_accel_y = self.gravity + engine_accel_y

        self.velocity_y += total_accel_y * self.dt
        self.y += self.velocity_y * self.dt

        prev_x_abs = abs(self.x)

        # lateral thruster applies fixed acceleration only while speed is under ±5 m/s in that direction; otherwise no thrust this step
        engine_accel_x = 0.0
        if lateral == 1 and self.velocity_x <= 5.0:
            engine_accel_x = self.thrust_x
        elif lateral == -1 and self.velocity_x >= -5.0:
            engine_accel_x = -self.thrust_x

        self.velocity_x += engine_accel_x * self.dt
        self.x += self.velocity_x * self.dt


        # torque -> angular acceleration (torque / inertia), then integrate to angular velocity and tilt angle; no damping, so spin persists until countered
        prev_theta_abs = abs(self.theta)

        angular_accel = (self.torque_magnitude * torque_fraction) / self.I
        self.omega += angular_accel * self.dt
        self.theta += self.omega * self.dt



        touchdown = bool(self.y <= 0.0)
        flew_away = bool(self.y > 50.0)
        out_of_bounds_x = bool(abs(self.x) > 30.0)
        tipped_over = bool(abs(self.theta) > self.theta_tip_limit)
        truncated = bool(self.step_count >= self.max_steps)
        terminated = touchdown or flew_away or out_of_bounds_x or tipped_over

        pad_radius = 5.0
        landed_on_pad = False
        soft_landing = False
        upright = False

        #Need to look further into reward shaping for more accurate values to be used
        if touchdown:
            impact_speed_y = abs(self.velocity_y)
            impact_speed_x = abs(self.velocity_x)
            landed_on_pad = bool(abs(self.x) <= pad_radius)
            soft_landing = bool(impact_speed_y <= self.safe_landing_speed_y
                                 and impact_speed_x <= self.safe_landing_speed_x)
            upright = bool(abs(self.theta) <= self.theta_upright_limit
                            and abs(self.omega) <= self.omega_upright_limit)

            if soft_landing and landed_on_pad and upright:
                total = 200.0 + (self.safe_landing_speed_y - impact_speed_y) * 15.0
                reward_rot = 60.0
                reward_lin = total - reward_rot
            else:
                total = -50.0 - (impact_speed_y * 7.5) - (abs(self.x) * 2.0) - (abs(self.theta) * 20.0)
                reward_rot = -(abs(self.theta) * 20.0) - (abs(self.omega) * 5.0)
                reward_lin = total - reward_rot
        elif flew_away or out_of_bounds_x:
            total = -180.0
            reward_lin = total
            reward_rot = 0.0
        elif tipped_over:
            total = -180.0 - (abs(self.theta) * 20.0) - (abs(self.omega) * 10.0)
            reward_rot = total
            reward_lin = 0.0
        elif truncated:
            total = -50.0
            reward_lin = total
            reward_rot = 0.0

        
        else:
            # ideal descent speed for this altitude (constant-deceleration profile, v = sqrt(2*a*h) with a=6), clipped to 0.5-12 m/s; negative because down is negative
            #a will need to be researched further to be further fine-tuned to find optimal value
            target_v = -np.clip(np.sqrt(2.0 * 6.0 * max(0.01, self.y)), 0.5, 12.0)
            #Difference to target value
            v_error = abs(self.velocity_y - target_v)

            x_error = abs(self.x)
            vx_error = abs(self.velocity_x)
            theta_error = abs(self.theta)
            omega_error = abs(self.omega)

            fuel_cost_lin = (0.02 if main_fire else 0.0) + (0.01 if lateral != 0 else 0.0)
            fuel_cost_rot = 0.01 * abs(torque_fraction)

            horizontal_progress = prev_x_abs - abs(self.x)
            angular_progress = prev_theta_abs - abs(self.theta)

            # combine error penalties, progress rewards, and fuel cost into reward_lin/reward_rot; total = both summed, but the split feeds two separate critics
            reward_lin = (
                -(0.15 * v_error)
                - (self.x_error_weight * x_error)
                - (self.vx_error_weight * vx_error)
                + (self.horizontal_progress_weight * horizontal_progress)
                - fuel_cost_lin
            )
            reward_rot = (
                -(self.theta_error_weight * theta_error)
                - (self.omega_error_weight * omega_error)
                + (self.angular_progress_weight * angular_progress)
                - fuel_cost_rot
            )
            total = reward_lin + reward_rot

        self.prev_x = self.x
        self.prev_theta = self.theta

        info = {
            "landed_success": bool(touchdown and soft_landing and landed_on_pad and upright),
            "vertical_ok": bool(touchdown and soft_landing),
            "horizontal_ok": bool(touchdown and landed_on_pad),
            "upright": bool(touchdown and upright),
            "final_x": self.x,
            "final_vx": self.velocity_x,
            "final_theta": self.theta,
            "final_omega": self.omega,
            "touchdown": touchdown,
            "reward_lin": reward_lin,
            "reward_rot": reward_rot,
            "torque_fraction": torque_fraction,
        }

        return self._get_obs(), total, terminated, truncated, info


# verifies torque affects only rotation (theta/omega), never y/vy/x/vx, by comparing several torque values against a shared physics reference
def regression_check_torque_decoupled(env_cls=RocketLander1DEnv):
    test_env = env_cls()
    test_env.reset()
    y0, x0 = test_env.y, test_env.x
    vy0, vx0 = test_env.velocity_y, test_env.velocity_x
    theta0, omega0 = test_env.theta, test_env.omega

    # reference will hold the first result to compare all other torque values against; sample a spread of values across the full range
    reference = None
    for torque_fraction in (-1.0, -0.3, 0.0, 0.5, 1.0):
        test_env.y, test_env.x = y0, x0
        test_env.velocity_y, test_env.velocity_x = vy0, vx0
        test_env.theta, test_env.omega = theta0, omega0
        next_state, reward, terminated, truncated, info = test_env.step([0, 0, torque_fraction])
        y_after, vy_after, x_after, vx_after = next_state[0], next_state[1], next_state[2], next_state[3]
        if reference is None:
            reference = (y_after, vy_after, x_after, vx_after)
        # compare this torque value's linear-motion result against the reference; any mismatch means torque is leaking into translation
        else:
            ref_y, ref_vy, ref_x, ref_vx = reference
            assert abs(y_after - ref_y) < 1e-6, \
                f"Torque decoupling FAILED: altitude differs at torque_fraction={torque_fraction}."
            assert abs(vy_after - ref_vy) < 1e-6, \
                f"Torque decoupling FAILED: vertical velocity differs at torque_fraction={torque_fraction}."
            assert abs(x_after - ref_x) < 1e-6, \
                f"Torque decoupling FAILED: x differs at torque_fraction={torque_fraction}."
            assert abs(vx_after - ref_vx) < 1e-6, \
                f"Torque decoupling FAILED: horizontal velocity differs at torque_fraction={torque_fraction}."

    print("Regression check PASSED: torque never affects y/vy/x/vx (fully decoupled actuator).\n")

# the policy network (actor) — decides what action to take; paired with separate linear/rotational critics defined below
class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, discrete_action_dims: list):
        super().__init__()
        # shared feature-extraction layers: 6-value state -> 128 -> 128, used as input by all three action heads
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        # output heads: discrete heads for main/lateral, plus mean (zero-init, starts at "no torque") and log_std for the continuous torque head
        self.discrete_heads = nn.ModuleList([nn.Linear(128, n) for n in discrete_action_dims])
        self.torque_mean = nn.Linear(128, 1)
        nn.init.zeros_(self.torque_mean.weight)
        nn.init.zeros_(self.torque_mean.bias)
        self.torque_log_std = nn.Linear(128, 1)

    # forward pass: state -> shared features -> discrete action probabilities + torque distribution params (mean, std); doesn't sample an action itself
    def forward(self, x):
        features = self.trunk(x)
        discrete_probs = [torch.softmax(head(features), dim=-1) for head in self.discrete_heads]
        mean = self.torque_mean(features)
        log_std = torch.clamp(self.torque_log_std(features), min=-20.0, max=2.0)
        std = torch.exp(log_std)
        return discrete_probs, mean, std

# critic estimates state value (single number), unlike the actor's action distributions — same trunk shape, but two instances (lin/rot) trained separately
class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    # runs state through the trunk+output layer, returns the single value estimate directly (no heads to split, unlike the actor)
    def forward(self, x):
        return self.net(x)

# tiny epsilon to avoid log(0) when tanh-squashed torque rounds to exactly ±1 (used in the log-prob correction below)
TORQUE_LOGPROB_EPS = 1e-6


def sample_actions(actor, state_tensor):
    # get this state's action distribution parameters from the actor network (not an action yet — sampling happens below)
    discrete_probs, mean, std = actor(state_tensor)
    # sample an action from each discrete head, plus log-prob (for policy gradient loss) and entropy (for exploration bonus)
    dists = [Categorical(p) for p in discrete_probs]
    discrete_actions = [d.sample() for d in dists]
    discrete_log_probs = [d.log_prob(a) for d, a in zip(dists, discrete_actions)]
    discrete_entropies = [d.entropy() for d in dists]

    # sample torque from the Gaussian (unbounded), then tanh-squash into [-1, 1]; squashing needs the log-prob correction below
    normal = Normal(mean, std)
    raw = normal.sample()
    squashed = torch.tanh(raw)

    log_prob_torque = normal.log_prob(raw) - torch.log(1 - squashed.pow(2) + TORQUE_LOGPROB_EPS)
    log_prob_torque = log_prob_torque.sum(dim=-1)
    entropy_torque = normal.entropy().sum(dim=-1)

# bundle the three sampled actions into a flat list [main, lateral, torque] matching what env.step() expects
    action_array = [
        discrete_actions[0].item(),
        discrete_actions[1].item(),
        squashed.item(),
    ]
    log_probs = discrete_log_probs + [log_prob_torque]
    entropies = discrete_entropies + [entropy_torque]
    return action_array, log_probs, entropies, std

# deterministic version of action selection (most likely discrete choice, tanh of the mean for torque) — used for evaluation, not training
def greedy_actions(actor, state_tensor):
    discrete_probs, mean, std = actor(state_tensor)
    discrete = [torch.argmax(p, dim=-1).item() for p in discrete_probs]
    torque_fraction = torch.tanh(mean).item()
    return discrete + [torque_fraction]

# computes discounted return per step (reward + gamma * future return), working backwards through the episode
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return torch.tensor(discounted_rewards, dtype=torch.float32)

# per-component scale factors to normalize the raw state to roughly [-1, 1] before feeding the network — tuned to typical ranges, not the full observation bounds
NORM_SCALES = np.array([30.0, 15.0, 10.0, 2.0, np.radians(45.0), 5.0], dtype=np.float32)


def normalize_state(state: np.ndarray) -> np.ndarray:
    return (state / NORM_SCALES).astype(np.float32)


CURRICULUM_STAGES = [
    (3.0, 0.5, np.radians(5.0), 0.10),
    (5.0, 0.75, np.radians(12.0), 0.20),
    (7.0, 1.0, np.radians(20.0), 0.35),
    (10.0, 1.5, np.radians(30.0), 0.50),
]
CURRICULUM_SUCCESS_THRESHOLD = 45.0
CURRICULUM_MIN_EPISODES_PER_STAGE = 300
SUCCESS_WINDOW_SIZE = 100
FINAL_STAGE_MIN_EPISODES = 1200
DIAGNOSTIC_INTERVAL = 2000

HEAD_ENTROPY_SCHEDULE = {
    "main":    (0.05, 0.005),
    "lateral": (0.05, 0.005),
    "torque":  (0.08, 0.01),
}


def diagnose_torque_behavior(env, actor, num_samples=300, theta_deadband_deg=1.0):
    deadband = np.radians(theta_deadband_deg)
    buckets = {"theta>0": [], "theta<0": [], "theta~0": []}

    for _ in range(num_samples):
        state, _ = env.reset()
        theta = state[4]
        state_tensor = torch.FloatTensor(normalize_state(state)).unsqueeze(0)
        with torch.no_grad():
            action = greedy_actions(actor, state_tensor)
        torque_fraction = action[2]

        if theta > deadband:
            buckets["theta>0"].append(torque_fraction)
        elif theta < -deadband:
            buckets["theta<0"].append(torque_fraction)
        else:
            buckets["theta~0"].append(torque_fraction)

    print("Torque-head behavior by initial theta sign (mode-collapse-style check):")
    for bucket, vals in buckets.items():
        if not vals:
            print(f"  {bucket:>8}: no samples in this bucket")
            continue
        arr = np.array(vals)
        print(f"  {bucket:>8} (n={len(vals):3d}): mean torque_fraction={arr.mean():6.3f} | std={arr.std():5.3f}")
    print("  Healthy: mean torque_fraction should be NEGATIVE for theta>0 and POSITIVE")
    print("  for theta<0 (opposing lean).\n")


def train(env, actor, critic_lin, critic_rot, actor_optimizer, critic_optimizer,
          episodes=6000, batch_size=32):
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    success_window = []
    vertical_window = []
    horizontal_window = []
    upright_window = []
    miss_x_window = []
    miss_vx_window = []

    curriculum_idx = 0
    episodes_in_stage = 0
    final_stage_episodes = 0
    env.set_spawn_range(*CURRICULUM_STAGES[curriculum_idx])

    episode = 0
    while episode < episodes:
        episode += 1

        frac = episode / episodes
        ent_coef = {
            head: start + (end - start) * frac
            for head, (start, end) in HEAD_ENTROPY_SCHEDULE.items()
        }

        state, _ = env.reset()
        log_probs_main, log_probs_lateral, log_probs_torque = [], [], []
        values_lin, values_rot = [], []
        rewards_lin, rewards_rot = [], []
        entropies_main, entropies_lateral, entropies_torque = [], [], []
        done = False
        last_info = {}

        while not done:
            state_tensor = torch.FloatTensor(normalize_state(state)).unsqueeze(0)

            val_lin = critic_lin(state_tensor)
            val_rot = critic_rot(state_tensor)
            action_array, log_probs, entropies, std_t = sample_actions(actor, state_tensor)

            next_state, reward, terminated, truncated, info = env.step(action_array)
            done = terminated or truncated
            last_info = info

            log_probs_main.append(log_probs[0])
            log_probs_lateral.append(log_probs[1])
            log_probs_torque.append(log_probs[2])
            values_lin.append(val_lin)
            values_rot.append(val_rot)
            rewards_lin.append(info["reward_lin"])
            rewards_rot.append(info["reward_rot"])
            entropies_main.append(entropies[0])
            entropies_lateral.append(entropies[1])
            entropies_torque.append(entropies[2])

            state = next_state

        total_rewards = [l + r for l, r in zip(rewards_lin, rewards_rot)]

        episode_success = last_info.get("landed_success", False)
        success_window.append(episode_success)
        if len(success_window) > SUCCESS_WINDOW_SIZE:
            success_window.pop(0)

        if last_info.get("touchdown", False):
            vertical_window.append(last_info.get("vertical_ok", False))
            horizontal_window.append(last_info.get("horizontal_ok", False))
            upright_window.append(last_info.get("upright", False))
            miss_x_window.append(abs(last_info["final_x"]))
            miss_vx_window.append(abs(last_info["final_vx"]))
            if len(vertical_window) > SUCCESS_WINDOW_SIZE:
                vertical_window.pop(0)
                horizontal_window.pop(0)
                upright_window.pop(0)
                miss_x_window.pop(0)
                miss_vx_window.pop(0)

        episodes_in_stage += 1
        at_final_stage = curriculum_idx == len(CURRICULUM_STAGES) - 1
        if at_final_stage:
            final_stage_episodes += 1

        discounted_lin = compute_discounted_rewards(rewards_lin)
        discounted_rot = compute_discounted_rewards(rewards_rot)
        values_lin_t = torch.cat(values_lin).squeeze(-1)
        values_rot_t = torch.cat(values_rot).squeeze(-1)

        log_probs_main_t = torch.cat(log_probs_main)
        log_probs_lateral_t = torch.cat(log_probs_lateral)
        log_probs_torque_t = torch.cat(log_probs_torque)

        entropy_main = torch.cat(entropies_main).mean()
        entropy_lateral = torch.cat(entropies_lateral).mean()
        entropy_torque = torch.cat(entropies_torque).mean()

        advantages_lin = (discounted_lin - values_lin_t).detach()
        advantages_rot = (discounted_rot - values_rot_t).detach()
        if len(advantages_lin) > 1:
            advantages_lin = (advantages_lin - advantages_lin.mean()) / (advantages_lin.std() + 1e-5)
        if len(advantages_rot) > 1:
            advantages_rot = (advantages_rot - advantages_rot.mean()) / (advantages_rot.std() + 1e-5)

        entropy_bonus = (ent_coef["main"] * entropy_main
                          + ent_coef["lateral"] * entropy_lateral
                          + ent_coef["torque"] * entropy_torque)

        actor_loss = -(
            (log_probs_main_t * advantages_lin).sum()
            + (log_probs_lateral_t * advantages_lin).sum()
            + (log_probs_torque_t * advantages_rot).sum()
        ) - entropy_bonus

        critic_loss = (nn.SmoothL1Loss()(values_lin_t, discounted_lin)
                       + nn.SmoothL1Loss()(values_rot_t, discounted_rot))

        (actor_loss / batch_size).backward()
        (critic_loss / batch_size).backward()

        if episode % batch_size == 0:
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(list(critic_lin.parameters()) + list(critic_rot.parameters()), max_norm=0.5)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if (curriculum_idx < len(CURRICULUM_STAGES) - 1
                and episodes_in_stage >= CURRICULUM_MIN_EPISODES_PER_STAGE
                and len(success_window) >= SUCCESS_WINDOW_SIZE):
            success_rate = 100.0 * sum(success_window) / len(success_window)
            if success_rate >= CURRICULUM_SUCCESS_THRESHOLD:
                curriculum_idx += 1
                episodes_in_stage = 0
                env.set_spawn_range(*CURRICULUM_STAGES[curriculum_idx])
                stage = CURRICULUM_STAGES[curriculum_idx]
                print(f"  >>> Curriculum advanced to stage {curriculum_idx} "
                      f"(x_range=\u00b1{stage[0]}, vx_range=\u00b1{stage[1]}, "
                      f"theta_range=\u00b1{np.degrees(stage[2]):.0f}deg, omega_range=\u00b1{stage[3]}) "
                      f"at episode {episode} (success rate {success_rate:.1f}%)")

        if episode % 50 == 0:
            success_rate = 100.0 * sum(success_window) / len(success_window) if success_window else 0.0
            vert_rate = 100.0 * sum(vertical_window) / len(vertical_window) if vertical_window else float('nan')
            horiz_rate = 100.0 * sum(horizontal_window) / len(horizontal_window) if horizontal_window else float('nan')
            upright_rate = 100.0 * sum(upright_window) / len(upright_window) if upright_window else float('nan')
            avg_miss_x = (sum(miss_x_window) / len(miss_x_window)) if miss_x_window else float('nan')
            avg_miss_vx = (sum(miss_vx_window) / len(miss_vx_window)) if miss_vx_window else float('nan')
            print(f"Episode {episode:4d} | Total Reward: {sum(total_rewards):7.2f} | Steps: {len(total_rewards):3d} | "
                  f"Final Alt: {state[0]:.2f}m | Final Vy: {state[1]:.2f}m/s | Final Vx: {state[3]:.2f}m/s | "
                  f"Final Theta: {np.degrees(state[4]):.1f}deg | "
                  f"Success: {success_rate:5.1f}% | Vert-OK: {vert_rate:5.1f}% | Horiz-OK: {horiz_rate:5.1f}% | "
                  f"Upright-OK: {upright_rate:5.1f}% | "
                  f"Avg|miss_x|: {avg_miss_x:5.2f}m | Avg|miss_vx|: {avg_miss_vx:4.2f}m/s | "
                  f"Stage: {curriculum_idx} | TorqueEntCoef: {ent_coef['torque']:.4f} | "
                  f"TorqueEnt: {entropy_torque.item():.4f} | TorqueStd: {std_t.item():.3f}")

        if episode % DIAGNOSTIC_INTERVAL == 0:
            print(f"  --- diagnostic at episode {episode} ---")
            diagnose_torque_behavior(env, actor, num_samples=150)

        if episode == episodes and at_final_stage and final_stage_episodes < FINAL_STAGE_MIN_EPISODES:
            extra = FINAL_STAGE_MIN_EPISODES - final_stage_episodes
            episodes += extra
            print(f"  >>> Extending training: only {final_stage_episodes} episodes at final stage "
                  f"(want >= {FINAL_STAGE_MIN_EPISODES}). Adding {extra} more episodes...")


def evaluate_and_plot(env, actor, num_tests=5):
    for test_idx in range(1, num_tests + 1):
        state, _ = env.reset()
        done = False

        times, alts, vels, xs, vxs, thetas, omegas, mains, laterals, torques = (
            [], [], [], [], [], [], [], [], [], []
        )
        step = 0
        last_info = {}

        while not done:
            times.append(step * env.dt)
            alts.append(state[0])
            vels.append(state[1])
            xs.append(state[2])
            vxs.append(state[3])
            thetas.append(np.degrees(state[4]))
            omegas.append(state[5])

            state_tensor = torch.FloatTensor(normalize_state(state)).unsqueeze(0)
            with torch.no_grad():
                action = greedy_actions(actor, state_tensor)

            mains.append(action[0])
            laterals.append(action[1])
            torques.append(action[2])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            last_info = info
            step += 1

        is_success = last_info.get("landed_success", False)
        print(f"Test Run {test_idx}: Final Alt = {state[0]:.2f}m | Final Vy = {state[1]:.2f}m/s | "
              f"Final x = {state[2]:.2f}m | Final Vx = {state[3]:.2f}m/s | "
              f"Final Theta = {np.degrees(state[4]):.1f}deg | Landed = {is_success}")

        if test_idx == 1:
            fig, axes = plt.subplots(5, 1, figsize=(8, 13), sharex=True)
            ax1, ax2, ax3, ax4, ax5 = axes

            ax1.plot(times, alts, 'b-', lw=2, label="Altitude (m)")
            ax1.axhline(0, color='k', ls='--')
            ax1.set_ylabel("Altitude (m)")
            ax1.set_title("Stage B (decoupled continuous torque) Telemetry")
            ax1.grid(True, ls=":", alpha=0.6)
            ax1.legend()

            ax2.plot(times, vels, 'r-', lw=2, label="Vertical Velocity (m/s)")
            ax2.axhline(-env.safe_landing_speed_y, color='g', ls='--',
                        label=f"Safe Vy Threshold (-{env.safe_landing_speed_y} m/s)")
            ax2.axhline(0, color='k', ls='--')
            ax2.set_ylabel("Velocity (m/s)")
            ax2.grid(True, ls=":", alpha=0.6)
            ax2.legend()

            ax3.plot(times, xs, 'm-', lw=2, label="Horizontal Position x (m)")
            ax3.plot(times, vxs, 'c--', lw=1.8, label="Horizontal Velocity vx (m/s)")
            ax3.axhline(5.0, color='g', ls=':', alpha=0.6, label="Pad edge")
            ax3.axhline(-5.0, color='g', ls=':', alpha=0.6)
            ax3.set_ylabel("Horizontal State")
            ax3.grid(True, ls=":", alpha=0.6)
            ax3.legend()

            ax4.plot(times, thetas, 'orange', lw=2, label="Theta (deg)")
            ax4.plot(times, omegas, 'brown', lw=1.8, ls='--', label="Omega (rad/s)")
            ax4.axhline(np.degrees(env.theta_tip_limit), color='r', ls=':', label="Tip-over limit")
            ax4.axhline(-np.degrees(env.theta_tip_limit), color='r', ls=':')
            ax4.set_ylabel("Rotational State")
            ax4.grid(True, ls=":", alpha=0.6)
            ax4.legend(fontsize=7)

            ax5.step(times, mains, where='post', label="Main (0/1)", alpha=0.6)
            ax5.step(times, laterals, where='post', label="Lateral (0/1/2)", alpha=0.6)
            ax5_twin = ax5.twinx()
            ax5_twin.plot(times, torques, 'g-', lw=1.8, label="Torque fraction [-1,1]")
            ax5_twin.set_ylabel("Torque fraction")
            ax5.set_ylabel("Discrete actions")
            ax5.set_xlabel("Time (s)")
            lines1, labels1 = ax5.get_legend_handles_labels()
            lines2, labels2 = ax5_twin.get_legend_handles_labels()
            ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
            ax5.grid(True, ls=":", alpha=0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_telemetry_stageB.png"), dpi=120)
            plt.close(fig)


def evaluate_generalization(env, actor, num_tests=50, x_range=10.0, vx_range=1.5,
                             theta_range=np.radians(35.0), omega_range=0.6):
    env.set_spawn_range(x_range, vx_range, theta_range, omega_range)
    successes = 0
    miss_x_list, miss_vx_list, final_vy_list, final_theta_list = [], [], [], []

    for _ in range(num_tests):
        state, _ = env.reset()
        done = False
        last_info = {}

        while not done:
            state_tensor = torch.FloatTensor(normalize_state(state)).unsqueeze(0)
            with torch.no_grad():
                action = greedy_actions(actor, state_tensor)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            last_info = info

        if last_info.get("landed_success", False):
            successes += 1
        if last_info.get("touchdown", False):
            miss_x_list.append(abs(last_info["final_x"]))
            miss_vx_list.append(abs(last_info["final_vx"]))
        final_vy_list.append(state[1])
        final_theta_list.append(np.degrees(state[4]))

    success_rate = 100.0 * successes / num_tests
    avg_miss_x = sum(miss_x_list) / len(miss_x_list) if miss_x_list else float('nan')
    avg_miss_vx = sum(miss_vx_list) / len(miss_vx_list) if miss_vx_list else float('nan')
    avg_final_vy = sum(final_vy_list) / len(final_vy_list)
    avg_final_theta = sum(final_theta_list) / len(final_theta_list)

    print(f"\nGeneralization check over {num_tests} random spawns "
          f"(x in \u00b1{x_range}, vx in \u00b1{vx_range}, "
          f"theta in \u00b1{np.degrees(theta_range):.0f}deg, omega in \u00b1{omega_range}):")
    print(f"  Success rate: {success_rate:.1f}% | Avg |miss_x| at touchdown: {avg_miss_x:.2f}m | "
          f"Avg |miss_vx|: {avg_miss_vx:.2f}m/s | Avg final Vy: {avg_final_vy:.2f}m/s | "
          f"Avg final Theta: {avg_final_theta:.1f}deg\n")


def validate_stage_b(env: gym.Env, num_episodes: int = 3, kp: float = 3.0, kd: float = 1.0):
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False

        times, ys, vys, xs, vxs, thetas, omegas, torques = [], [], [], [], [], [], [], []
        step = 0

        while not done:
            y, vy, x, vx, theta, omega = state

            times.append(step * env.dt)
            ys.append(y)
            vys.append(vy)
            xs.append(x)
            vxs.append(vx)
            thetas.append(np.degrees(theta))
            omegas.append(omega)

            main_idx = 1 if y < 15.0 else 0

            if vx > 0.5:
                lateral_idx = 2
            elif vx < -0.5:
                lateral_idx = 1
            else:
                lateral_idx = 0

            torque_fraction = float(np.clip(-kp * theta - kd * omega, -1.0, 1.0))
            torques.append(torque_fraction)

            action = [main_idx, lateral_idx, torque_fraction]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            step += 1

        print(f"Validation Run {ep}: Start theta = {thetas[0]:.1f}deg | Final theta = {thetas[-1]:.1f}deg | "
              f"Final vx = {vxs[-1]:.2f}m/s | Final vy = {vys[-1]:.2f}m/s")

        if ep == 1:
            fig, (ax_traj, ax_x, ax_theta) = plt.subplots(1, 3, figsize=(16, 5))

            ax_traj.plot(xs, ys, 'b-', lw=2, label="Flight Trajectory")
            ax_traj.plot(xs[0], ys[0], 'go', markersize=8, label="Start Point")
            ax_traj.plot(xs[-1], ys[-1], 'rx', markersize=10, mew=2, label="End Point")
            ax_traj.axhline(0, color='k', ls='--', label="Ground (y=0)")
            ax_traj.axvline(0, color='g', ls=':', alpha=0.5, label="Pad Center (x=0)")
            ax_traj.set_xlim([-15, 15])
            ax_traj.set_ylim([0, 40])
            ax_traj.set_xlabel("Horizontal Position x (m)")
            ax_traj.set_ylabel("Altitude y (m)")
            ax_traj.set_title("Stage B Scripted PD Pilot (continuous torque)")
            ax_traj.grid(True, ls=":", alpha=0.6)
            ax_traj.legend()

            ax_x.plot(times, xs, 'b-', lw=2, label="x position (m)")
            ax_x.plot(times, vxs, 'm-', lw=2, label="vx velocity (m/s)")
            ax_x.set_xlabel("Time (s)")
            ax_x.set_ylabel("Horizontal State")
            ax_x.set_title("Horizontal Dynamics")
            ax_x.grid(True, ls=":", alpha=0.6)
            ax_x.legend()

            ax_theta.plot(times, thetas, 'orange', lw=2, label="Theta (deg)")
            ax_theta.plot(times, omegas, 'brown', lw=1.8, ls='--', label="Omega (rad/s)")
            ax_theta_twin = ax_theta.twinx()
            ax_theta_twin.plot(times, torques, 'g-', lw=1.2, alpha=0.7, label="Torque fraction")
            ax_theta_twin.set_ylabel("Torque fraction")
            ax_theta.axhline(np.degrees(env.theta_tip_limit), color='r', ls=':', label="Tip-over limit")
            ax_theta.axhline(-np.degrees(env.theta_tip_limit), color='r', ls=':')
            ax_theta.set_xlabel("Time (s)")
            ax_theta.set_ylabel("Rotational State")
            ax_theta.set_title("Rotational Dynamics (continuous torque, PD pilot)")
            ax_theta.grid(True, ls=":", alpha=0.6)
            lines1, labels1 = ax_theta.get_legend_handles_labels()
            lines2, labels2 = ax_theta_twin.get_legend_handles_labels()
            ax_theta.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "stage_b_validation.png"), dpi=120)
            plt.close(fig)


if __name__ == "__main__":
    regression_check_torque_decoupled()

    env = RocketLander1DEnv(initial_y=30.0, initial_x=5, thrust_y=22.0, thrust_x=5.0)
    state_dim = env.observation_space.shape[0]
    discrete_action_dims = [2, 3]

    actor = ActorNetwork(state_dim, discrete_action_dims)
    critic_lin = CriticNetwork(state_dim)
    critic_rot = CriticNetwork(state_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(
        list(critic_lin.parameters()) + list(critic_rot.parameters()), lr=1e-3
    )

    validate_stage_b(env, num_episodes=3)

    train(env, actor, critic_lin, critic_rot, actor_optimizer, critic_optimizer,
          episodes=10000, batch_size=32)
    evaluate_and_plot(env, actor, num_tests=5)
    evaluate_generalization(env, actor, num_tests=50, x_range=10.0, vx_range=1.5)
    diagnose_torque_behavior(env, actor, num_samples=300)