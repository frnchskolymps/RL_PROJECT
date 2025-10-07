# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# from lock_key_env import LockKeyEnv   # make sure this matches your file name


# def run(episodes, is_training=True, render=False):
#     # Initialize custom LockKey environment
#     env = LockKeyEnv(render_mode='human' if render else None, size=6)

#     if is_training:
#         q = np.zeros((env.observation_space.n, env.action_space.n))
#     else:
#         with open('lockkey.pkl', 'rb') as f:
#             q = pickle.load(f)

#     # --- Q-learning parameters ---
#     learning_rate_a = 0.9     # alpha
#     discount_factor_g = 0.9   # gamma
#     epsilon = 1.0             # exploration rate
#     epsilon_decay_rate = 0.0001
#     rng = np.random.default_rng()

#     rewards_per_episode = np.zeros(episodes)

#     for i in range(episodes):
#         # Store episode number for display in render()
#         env.current_episode = i + 1

#         state, _ = env.reset()
#         terminated = False
#         truncated = False
#         total_reward = 0

#         while not terminated and not truncated:
#             # --- Epsilon-greedy action selection ---
#             if is_training and rng.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(q[state, :])

#             new_state, reward, terminated, truncated, _ = env.step(action)
#             total_reward += reward

#             # --- Q-learning update ---
#             if is_training:
#                 q[state, action] += learning_rate_a * (
#                     reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
#                 )

#             state = new_state

#             # Render if enabled
#             if render:
#                 env.render()

#         # --- Decay epsilon ---
#         epsilon = max(epsilon - epsilon_decay_rate, 0)
#         if epsilon == 0:
#             learning_rate_a = 0.0001

#         rewards_per_episode[i] = total_reward

#         # --- Progress output every 500 episodes ---
#         if is_training and (i + 1) % 500 == 0:
#             avg_reward = np.mean(rewards_per_episode[max(0, i - 100):i + 1])
#             print(f"Episode {i + 1}/{episodes} | Epsilon: {epsilon:.4f} | AvgReward(100): {avg_reward:.2f}")

#     env.close()

#     # --- Plot total rewards per episode ---
#     plt.figure(figsize=(8, 5))
#     plt.plot(rewards_per_episode, color='royalblue')
#     plt.title("Lock & Key Q-Learning Performance")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward per Episode")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("lockkey_training_curve.png")
#     plt.show()

#     # --- Save Q-table ---
#     if is_training:
#         with open("lockkey.pkl", "wb") as f:
#             pickle.dump(q, f)
#         print("✅ Training complete. Q-table saved as lockkey.pkl.")


# if __name__ == "__main__":
#     # --- Training phase ---
#     run(episodes=15000, is_training=False, render=False)

#     # --- Test/Render phase ---
#     run(episodes=10, is_training=False, render=True)



## newwww



# train_lock_key.py
# train_lock_key.py
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lock_key_env import LockKeyEnv
import os
import warnings

def train_curriculum(
    phase_schedule,
    episodes_per_phase,
    save_path="lockkey_unified.pkl",
    base_alpha=0.9,
    base_gamma=0.9,
    epsilon_start=1.0,
    base_epsilon_decay=1e-4,
    per_phase_hparams=None,
    seed=None,
    render_eval=False,
    max_steps_per_episode=200
):
    """
    Train across phases sequentially using a single unified Q-table.
    - phase_schedule: list e.g. [1,2,3,4]
    - episodes_per_phase: int or list (same length as phase_schedule)
    - per_phase_hparams: dict mapping phase -> dict of {alpha, epsilon_decay, episodes_override}
      Example: {4: {"alpha":0.1, "epsilon_decay":1e-5, "episodes":2000}}
    """

    rng = np.random.default_rng(seed)

    # instantiate env with phase 1 by default to get state/action sizes
    env = LockKeyEnv(render_mode=None, size=6, phase=1, seed=seed)
    obs_n = env.observation_space.n
    act_n = env.action_space.n

    # initialize unified Q-table
    q = np.zeros((obs_n, act_n), dtype=float)

    # normalize episodes_per_phase to list
    if isinstance(episodes_per_phase, int):
        episodes_per_phase = [episodes_per_phase] * len(phase_schedule)
    assert len(episodes_per_phase) == len(phase_schedule), "episodes_per_phase length mismatch"

    # normalize per_phase_hparams
    if per_phase_hparams is None:
        per_phase_hparams = {}

    epsilon = epsilon_start
    alpha = base_alpha
    gamma = base_gamma
    base_eps_decay = base_epsilon_decay

    total_episode_counter = 0
    rewards_history = []

    print("Starting curriculum training. Phases:", phase_schedule)
    for idx, phase in enumerate(phase_schedule):
        # allow per-phase overrides
        phase_episodes = episodes_per_phase[idx]
        phase_hp = per_phase_hparams.get(phase, {})
        phase_alpha = phase_hp.get("alpha", base_alpha)
        phase_epsilon_decay = phase_hp.get("epsilon_decay", base_eps_decay)
        # allow episodes override inside per_phase_hparams
        phase_episodes = phase_hp.get("episodes", phase_episodes)

        # If environment doesn't support the phase (e.g., phase 5 with enemy), warn & skip
        if phase == 5:
            # check if env supports phase 5 by trying to set and reset
            try:
                env.phase = 5
                _obs, _ = env.reset()
            except Exception as e:
                warnings.warn(
                    "Phase 5 detected in schedule but environment does not appear to support an enemy (phase=5). "
                    "Skipping Phase 5. Implement enemy dynamics in lock_key_env.py and retry to train Phase 5."
                )
                # restore phase 1 and continue
                env.phase = phase_schedule[0] if phase_schedule else 1
                continue

        env.phase = phase
        print(f"\n--- Phase {phase} training: {phase_episodes} episodes | alpha={phase_alpha} | eps_decay={phase_epsilon_decay} ---")

        for ep in range(phase_episodes):
            total_episode_counter += 1
            env.current_episode = total_episode_counter
            state, _ = env.reset()
            terminated = False
            truncated = False
            ep_reward = 0
            step_count = 0

            while not terminated and not truncated and step_count < max_steps_per_episode:
                # epsilon-greedy
                if rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q[state, :]))

                new_state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward

                # Q-learning update
                q[state, action] += phase_alpha * (reward + gamma * np.max(q[new_state, :]) - q[state, action])

                state = new_state
                step_count += 1

            # decay epsilon for this phase (linear)
            epsilon = max(0.0, epsilon - phase_epsilon_decay)

            rewards_history.append(ep_reward)

            # occasional logging
            if total_episode_counter % 500 == 0:
                avg100 = np.mean(rewards_history[max(0, len(rewards_history)-100):])
                print(f"Episode {total_episode_counter} | Phase {phase} | Epsilon: {epsilon:.4f} | AvgReward(100): {avg100:.2f}")

    # Save Q-table
    with open(save_path, "wb") as f:
        pickle.dump(q, f)
    print(f"\n✅ Training complete. Q-table saved to {save_path}")

    # Plot rewards
    plt.figure(figsize=(9, 5))
    plt.plot(rewards_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode (curriculum)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lockkey_unified_training_curve.png")
    print("Training curve saved to lockkey_unified_training_curve.png")

    # optional short eval
    print("\n--- Running short evaluation (rendering optional) ---")
    evaluate(save_path, episodes=10, render=render_eval, phase=3)

    return q


def evaluate(q_path, episodes=5, render=True, phase=3, seed=None, max_steps=200):
    """
    Load Q-table and run episodes for visual check in chosen phase.
    """
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"Q-table not found: {q_path}")

    with open(q_path, "rb") as f:
        q = pickle.load(f)

    env = LockKeyEnv(render_mode='human' if render else None, size=6, phase=phase, seed=seed)

    # quick check for phase support (phase 5 may not exist)
    try:
        _ = env.reset()
    except Exception as e:
        print(f"Environment reset failed for phase {phase}: {e}")
        env.close()
        return []

    success_count = 0
    total_rewards = []

    for ep in range(episodes):
        env.current_episode = ep + 1
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0
        step_count = 0

        while not terminated and not truncated and step_count < max_steps:
            action = int(np.argmax(q[state, :]))
            new_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            state = new_state
            step_count += 1

            if render:
                env.render()

        total_rewards.append(ep_reward)
        if terminated:
            success_count += 1

    env.close()
    print(f"Evaluation | Phase {phase} | Episodes {episodes} | Successes: {success_count}/{episodes} | AvgReward: {np.mean(total_rewards):.2f}")
    return total_rewards


if __name__ == "__main__":
    # ---------- Default curriculum based on your table ----------
    # Phase descriptions you provided:
    # 1: Fixed grid/walls/key/lock (Base logic)        - Easy
    # 2: Random agent/key (Generalization)             - Medium
    # 3: Random lock (More task diversity)             - Medium
    # 4: Stabilize & tune (Reward & transitions)       - Easy (stabilize)
    # 5: Add enemy (random moves)                      - Hard (requires env support)
    #
    # Configure here (change numbers as you want)
    PHASE_SCHEDULE = [1, 2, 3, 4, 5]   # include 5 but script will skip if env lacks support
    EPISODES_PER_PHASE = [15000, 10000, 10000, 5000, 5000]  # sample counts; tune as needed

    # Per-phase hyperparameter tweaks (especially Phase 4 to stabilize)
    PER_PHASE_HPARAMS = {
        1: {"alpha": 0.9, "epsilon_decay": 1e-4, "episodes": EPISODES_PER_PHASE[0], "epsilon_start": 1.0},
        2: {"alpha": 0.85, "epsilon_decay": 8e-5, "episodes": EPISODES_PER_PHASE[1], "epsilon_start": 0.9},
        3: {"alpha": 0.8, "epsilon_decay": 5e-5, "episodes": EPISODES_PER_PHASE[2], "epsilon_start": 0.8},
        4: {"alpha": 0.15, "epsilon_decay": 1e-5, "episodes": EPISODES_PER_PHASE[3], "epsilon_start": 0.5},
        5: {"alpha": 0.1, "epsilon_decay": 5e-5, "episodes": EPISODES_PER_PHASE[4], "epsilon_start": 0.8},  # Phase 5: harder
    }


    # Train
    q_table = train_curriculum(
        phase_schedule=PHASE_SCHEDULE,
        episodes_per_phase=EPISODES_PER_PHASE,
        save_path="lockkey_unified.pkl",
        base_alpha=0.9,
        base_gamma=0.9,
        epsilon_start=1.0,
        base_epsilon_decay=1e-4,
        per_phase_hparams=PER_PHASE_HPARAMS,
        seed=None,
        render_eval=False
    )

    # After training, allow interactive evaluation by phase
    while True:
        choice = input("\nEnter a phase to run evaluation (1-5), or 'q' to quit: ").strip().lower()
        if choice == 'q':
            print("Quitting.")
            break
        if choice not in ['1', '2', '3', '4', '5']:
            print("Invalid input; enter 1,2,3,4,5 or q.")
            continue

        phase_to_eval = int(choice)
        episodes_to_eval = input("How many episodes to run for this simulation? (default 5): ").strip()
        try:
            episodes_to_eval = int(episodes_to_eval)
        except:
            episodes_to_eval = 5

        render_choice = input("Render? (y/n, default y): ").strip().lower()
        render_flag = (render_choice in ['', 'y', 'yes'])

        # If evaluating Phase 5 but env doesn't support it, warn and skip
        if phase_to_eval == 5:
            try:
                # quick instantiation check
                tmp_env = LockKeyEnv(render_mode=None, size=6, phase=5)
                tmp_env.reset()
                tmp_env.close()
            except Exception:
                print("Phase 5 (enemy) not supported by your environment. Implement enemy dynamics in lock_key_env.py to enable phase 5 training/eval.")
                continue

        evaluate("lockkey_unified.pkl", episodes=episodes_to_eval, render=render_flag, phase=phase_to_eval)
