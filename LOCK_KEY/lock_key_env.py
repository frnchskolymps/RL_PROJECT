# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pygame
# import sys


# class LockKeyEnv(gym.Env):
#     metadata = {"render_modes": ["human"], "render_fps": 5}

#     def __init__(self, render_mode=None, size=6):
#         super().__init__()
#         self.size = size
#         self.window_size = 800   # total window width
#         self.grid_size = 500     # main grid area
#         self.info_width = 300    # side panel width
#         self.cell_size = self.grid_size // self.size

#         self.action_space = spaces.Discrete(6)  # S, N, E, W, Pickup, Unlock
#         self.observation_space = spaces.Discrete(self.size * self.size * 2)

#         self.render_mode = render_mode
#         self.window = None
#         self.clock = None

#         # --- New attributes ---
#         self.framerate = 5  # Default FPS
#         self.unlimited_fps = False  # Option for no cap
#         self.current_episode = 0  # ✅ added for episode tracking

#         # --- Fixed positions and walls for 6x6 grid ---
#         self.lock_pos = np.array([0, 5])  # top-right corner
#         self.key_pos = np.array([5, 3])   # near bottom center

#         self.walls = {
#             (0, 2), (1, 1), (1, 4),
#             (2, 3), (3, 0), (3, 2),
#             (4, 4)
#         }

#         self.agent_pos = None
#         self.has_key = False
#         self.steps = 0
#         self.last_reward = 0

#     # ------------------------------------------------------------------

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.agent_pos = np.array([self.size - 1, 0])  # bottom-left corner
#         self.has_key = False
#         self.steps = 0
#         self.last_reward = 0

#         observation = self._get_obs()
#         if self.render_mode == "human":
#             self._render_frame()

#         return observation, {}

#     # ------------------------------------------------------------------

#     def step(self, action):
#         self.steps += 1
#         reward = -1
#         terminated = False

#         move_map = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
#         if action in move_map:
#             dr, dc = move_map[action]
#             new_pos = self.agent_pos + np.array([dr, dc])

#             if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size
#                     and (new_pos[0], new_pos[1]) not in self.walls):
#                 self.agent_pos = new_pos
#             else:
#                 reward = -1  # hitting wall or out of bounds

#         elif action == 4:  # Manual pickup
#             if np.array_equal(self.agent_pos, self.key_pos) and not self.has_key:
#                 self.has_key = True
#                 reward = 10
#             else:
#                 reward = -10

#         elif action == 5:  # Manual unlock
#             if np.array_equal(self.agent_pos, self.lock_pos):
#                 if self.has_key:
#                     reward = 20
#                     terminated = True
#                 else:
#                     reward = -10
#             else:
#                 reward = -10

#         # --- ✅ AUTO PICKUP AND AUTO UNLOCK ---
#         if np.array_equal(self.agent_pos, self.key_pos) and not self.has_key:
#             self.has_key = True
#             reward += 10  # bonus for picking up automatically

#         if np.array_equal(self.agent_pos, self.lock_pos) and self.has_key:
#             reward += 20
#             terminated = True

#         self.last_reward = reward
#         if self.render_mode == "human":
#             self._render_frame()

#         return self._get_obs(), reward, terminated, False, {}

#     # ------------------------------------------------------------------

#     def _get_obs(self):
#         r, c = self.agent_pos
#         return r * self.size + c + (self.size * self.size * int(self.has_key))

#     # ------------------------------------------------------------------
#     # --------------------------- Rendering -----------------------------
#     # ------------------------------------------------------------------

#     def _render_frame(self):
#         if self.window is None:
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.grid_size))
#         if self.clock is None:
#             self.clock = pygame.time.Clock()

#         # --- Handle input events ---
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 self.close()
#                 sys.exit()
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     self.close()
#                     sys.exit()
#                 elif event.key == pygame.K_UP:
#                     self.framerate += 5
#                     self.unlimited_fps = False
#                 elif event.key == pygame.K_DOWN:
#                     self.framerate = max(1, self.framerate - 5)
#                     self.unlimited_fps = False
#                 elif event.key == pygame.K_u:
#                     self.unlimited_fps = not self.unlimited_fps

#         # --- Drawing ---
#         canvas = pygame.Surface((self.grid_size, self.grid_size))
#         canvas.fill((230, 230, 230))

#         for x in range(self.size + 1):
#             pygame.draw.line(canvas, (0, 0, 0),
#                              (0, x * self.cell_size),
#                              (self.grid_size, x * self.cell_size), 1)
#             pygame.draw.line(canvas, (0, 0, 0),
#                              (x * self.cell_size, 0),
#                              (x * self.cell_size, self.grid_size), 1)

#         for (r, c) in self.walls:
#             pygame.draw.rect(
#                 canvas, (100, 100, 100),
#                 pygame.Rect(
#                     c * self.cell_size, r * self.cell_size,
#                     self.cell_size, self.cell_size
#                 ),
#             )

#         pygame.draw.rect(
#             canvas, (200, 50, 50),
#             pygame.Rect(
#                 self.lock_pos[1] * self.cell_size + self.cell_size // 4,
#                 self.lock_pos[0] * self.cell_size + self.cell_size // 4,
#                 self.cell_size // 2, self.cell_size // 2
#             )
#         )

#         if not self.has_key:
#             pygame.draw.circle(
#                 canvas, (255, 215, 0),
#                 (self.key_pos[1] * self.cell_size + self.cell_size // 2,
#                  self.key_pos[0] * self.cell_size + self.cell_size // 2),
#                 self.cell_size // 4
#             )

#         pygame.draw.circle(
#             canvas, (50, 100, 255),
#             (self.agent_pos[1] * self.cell_size + self.cell_size // 2,
#              self.agent_pos[0] * self.cell_size + self.cell_size // 2),
#             self.cell_size // 3
#         )

#         if self.has_key:
#             pygame.draw.circle(
#                 canvas, (255, 215, 0),
#                 (self.agent_pos[1] * self.cell_size + self.cell_size // 2,
#                  self.agent_pos[0] * self.cell_size + self.cell_size // 2),
#                 self.cell_size // 6
#             )

#         # --- Side info panel ---
#         info_panel = pygame.Surface((self.info_width, self.grid_size))
#         info_panel.fill((245, 245, 245))
#         font = pygame.font.SysFont("arial", 20)

#         texts = [
#             f"Episode: {self.current_episode}",  # ✅ Added line
#             f"Steps: {self.steps}",
#             f"Reward (last): {self.last_reward}",
#             f"Has Key: {'Yes' if self.has_key else 'No'}",
#             "",
#             f"FPS: {'∞' if self.unlimited_fps else self.framerate}",
#             "",
#             "Controls:",
#             "↑ - Increase FPS",
#             "↓ - Decrease FPS",
#             "U - Toggle Unlimited FPS",
#             "ESC - Quit",
#             "",
#             "Actions:",
#             "0 - South",
#             "1 - North",
#             "2 - East",
#             "3 - West",
#             "4 - Pickup",
#             "5 - Unlock"
#         ]

#         y = 30
#         for t in texts:
#             txt_surface = font.render(t, True, (0, 0, 0))
#             info_panel.blit(txt_surface, (20, y))
#             y += 30

#         self.window.blit(canvas, (0, 0))
#         self.window.blit(info_panel, (self.grid_size, 0))
#         pygame.display.update()

#         if not self.unlimited_fps:
#             self.clock.tick(self.framerate)

#     def render(self):
#         if self.render_mode == "human":
#             self._render_frame()

#     def close(self):
#         if self.window is not None:
#             pygame.display.quit()
#             pygame.quit()


# # Quick test
# if __name__ == "__main__":
#     env = LockKeyEnv(render_mode="human", size=6)
#     env.current_episode = 1  # ✅ test example
#     obs, _ = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, done, trunc, info = env.step(action)
#     env.close()



# newwwwww



# # lock_key_env.py
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pygame
# import sys


# class LockKeyEnv(gym.Env):
#     """
#     Unified Lock & Key environment with integer-encoded observation:
#     (agent_r, agent_c, key_r, key_c, lock_r, lock_c, has_key) -> single int

#     Phases (controlled by env.phase):
#       1 -> fixed agent, fixed key, fixed lock  (default)
#       2 -> random agent & random key, fixed lock
#       3 -> random agent, random key, random lock
#     """

#     metadata = {"render_modes": ["human"], "render_fps": 5}

#     def __init__(self, render_mode=None, size=6, phase=1, seed=None):
#         super().__init__()
#         assert size == 6, "This environment assumes a 6x6 grid (size=6)."
#         assert phase in (1, 2, 3), "phase must be 1, 2 or 3."

#         self.size = size
#         self.num_cells = self.size * self.size
#         # total states: agent_pos * key_pos * lock_pos * has_key(2)
#         self.total_states = self.num_cells * self.num_cells * self.num_cells * 2

#         self.action_space = spaces.Discrete(6)  # 0:S,1:N,2:E,3:W,4:Pickup,5:Unlock
#         self.observation_space = spaces.Discrete(self.total_states)

#         # rendering
#         self.render_mode = render_mode
#         self.window = None
#         self.clock = None
#         self.window_size = 800
#         self.grid_size = 500
#         self.info_width = 300
#         self.cell_size = self.grid_size // self.size

#         # default positions (Phase 1 / canonical)
#         self._default_agent_pos = np.array([self.size - 1, 0])  # bottom-left (5,0)
#         self._default_key_pos = np.array([5, 3])               # as before
#         self._default_lock_pos = np.array([0, 5])              # top-right

#         # walls (same set you used previously)
#         self.walls = {
#             (0, 2), (1, 1), (1, 4),
#             (2, 3), (3, 0), (3, 2),
#             (4, 4)
#         }

#         # runtime variables
#         self.agent_pos = None
#         self.key_pos = None
#         self.lock_pos = self._default_lock_pos.copy()
#         self.has_key = False
#         self.steps = 0
#         self.last_reward = 0
#         self.current_episode = 0

#         # phase and rng
#         self.phase = phase
#         self._rng = np.random.default_rng(seed)

#         # FPS control for rendering
#         self.framerate = 5
#         self.unlimited_fps = False

#     # -------------------------
#     # Gym API
#     # -------------------------
#     def reset(self, seed=None, options=None):
#         # update rng if seed provided
#         if seed is not None:
#             self._rng = np.random.default_rng(seed)

#         # Phase-specific initialization:
#         if self.phase == 1:
#             # Fixed agent, fixed key, fixed lock
#             self.agent_pos = self._default_agent_pos.copy()
#             self.key_pos = self._default_key_pos.copy()
#             self.lock_pos = self._default_lock_pos.copy()
#         elif self.phase == 2:
#             # Random agent & random key, fixed lock
#             self.lock_pos = self._default_lock_pos.copy()
#             self.agent_pos = self._random_free_cell(exclude={tuple(self.lock_pos)})
#             self.key_pos = self._random_free_cell(exclude={tuple(self.lock_pos), tuple(self.agent_pos)})
#         elif self.phase == 3:
#             # Random agent, random key, random lock
#             self.lock_pos = self._random_free_cell(exclude=set())
#             # ensure agent and key not on lock and not on walls and distinct
#             self.agent_pos = self._random_free_cell(exclude={tuple(self.lock_pos)})
#             self.key_pos = self._random_free_cell(exclude={tuple(self.lock_pos), tuple(self.agent_pos)})
#         else:
#             raise ValueError("Invalid phase; must be 1, 2, or 3")

#         self.has_key = False
#         self.steps = 0
#         self.last_reward = 0

#         obs = self._get_obs()
#         if self.render_mode == "human":
#             self._render_frame()
#         return obs, {}

#     def step(self, action):
#         self.steps += 1
#         reward = -1  # default step penalty
#         terminated = False
#         truncated = False

#         # movement mapping: 0=S,1=N,2=E,3=W
#         move_map = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
#         if action in move_map:
#             dr, dc = move_map[action]
#             new_pos = self.agent_pos + np.array([dr, dc])
#             if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size
#                     and (new_pos[0], new_pos[1]) not in self.walls):
#                 self.agent_pos = new_pos
#             else:
#                 # bump into wall or out of bounds: keep -1 reward
#                 reward = -1

#         elif action == 4:  # Manual pickup
#             if (not self.has_key) and np.array_equal(self.agent_pos, self.key_pos):
#                 self.has_key = True
#                 reward = 10
#             else:
#                 reward = -10

#         elif action == 5:  # Manual unlock
#             if np.array_equal(self.agent_pos, self.lock_pos):
#                 if self.has_key:
#                     reward = 20
#                     terminated = True
#                 else:
#                     reward = -10
#             else:
#                 reward = -10

#         # AUTO PICKUP (key remains at its randomized position in the state)
#         if (not self.has_key) and np.array_equal(self.agent_pos, self.key_pos):
#             self.has_key = True
#             reward += 10  # additional reward for stepping onto key

#         # AUTO UNLOCK
#         if self.has_key and np.array_equal(self.agent_pos, self.lock_pos):
#             reward += 20
#             terminated = True

#         self.last_reward = reward
#         if self.render_mode == "human":
#             self._render_frame()
#         return self._get_obs(), reward, terminated, truncated, {}

#     # -------------------------
#     # Helpers: random free cell and encoding/decoding
#     # -------------------------
#     def _random_free_cell(self, exclude=None):
#         """
#         Return a random (r,c) numpy array that's not in walls and not in exclude set.
#         exclude can be a set of (r,c) tuples.
#         """
#         if exclude is None:
#             exclude = set()
#         while True:
#             r = int(self._rng.integers(0, self.size))
#             c = int(self._rng.integers(0, self.size))
#             if (r, c) in self.walls:
#                 continue
#             if (r, c) in exclude:
#                 continue
#             return np.array([r, c])

#     def _get_obs(self):
#         """
#         Flatten (agent_idx, key_idx, lock_idx, has_key) into a single integer:
#         idx = (((agent_idx * num_cells) + key_idx) * num_cells + lock_idx) * 2 + has_key
#         """
#         ar, ac = int(self.agent_pos[0]), int(self.agent_pos[1])
#         kr, kc = int(self.key_pos[0]), int(self.key_pos[1])
#         lr, lc = int(self.lock_pos[0]), int(self.lock_pos[1])

#         agent_idx = ar * self.size + ac
#         key_idx = kr * self.size + kc
#         lock_idx = lr * self.size + lc
#         idx = (((agent_idx * self.num_cells) + key_idx) * self.num_cells + lock_idx) * 2 + int(self.has_key)
#         return int(idx)

#     @staticmethod
#     def decode_obs(obs, size=6):
#         """
#         Decode the flattened observation integer back to
#         (agent_r, agent_c, key_r, key_c, lock_r, lock_c, has_key)
#         """
#         num_cells = size * size
#         has_key = obs % 2
#         tmp = obs // 2
#         lock_idx = tmp % num_cells
#         tmp = tmp // num_cells
#         key_idx = tmp % num_cells
#         agent_idx = tmp // num_cells

#         ar, ac = divmod(agent_idx, size)
#         kr, kc = divmod(key_idx, size)
#         lr, lc = divmod(lock_idx, size)
#         return (ar, ac, kr, kc, lr, lc, int(has_key))

#     # -------------------------
#     # Rendering (pygame)
#     # -------------------------
#     def _render_frame(self):
#         if self.window is None:
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.grid_size))
#         if self.clock is None:
#             self.clock = pygame.time.Clock()

#         # event handling
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 self.close()
#                 sys.exit()
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     self.close()
#                     sys.exit()
#                 elif event.key == pygame.K_UP:
#                     self.framerate += 5
#                     self.unlimited_fps = False
#                 elif event.key == pygame.K_DOWN:
#                     self.framerate = max(1, self.framerate - 5)
#                     self.unlimited_fps = False
#                 elif event.key == pygame.K_u:
#                     self.unlimited_fps = not self.unlimited_fps

#         # draw grid
#         canvas = pygame.Surface((self.grid_size, self.grid_size))
#         canvas.fill((230, 230, 230))
#         for x in range(self.size + 1):
#             pygame.draw.line(canvas, (0, 0, 0),
#                              (0, x * self.cell_size),
#                              (self.grid_size, x * self.cell_size), 1)
#             pygame.draw.line(canvas, (0, 0, 0),
#                              (x * self.cell_size, 0),
#                              (x * self.cell_size, self.grid_size), 1)

#         # walls
#         for (r, c) in self.walls:
#             pygame.draw.rect(canvas, (100, 100, 100),
#                              pygame.Rect(c * self.cell_size, r * self.cell_size,
#                                          self.cell_size, self.cell_size))

#         # lock (draw as red square)
#         pygame.draw.rect(
#             canvas, (200, 50, 50),
#             pygame.Rect(
#                 self.lock_pos[1] * self.cell_size + self.cell_size // 4,
#                 self.lock_pos[0] * self.cell_size + self.cell_size // 4,
#                 self.cell_size // 2, self.cell_size // 2
#             )
#         )

#         # key (if not held)
#         if not self.has_key:
#             pygame.draw.circle(
#                 canvas, (255, 215, 0),
#                 (self.key_pos[1] * self.cell_size + self.cell_size // 2,
#                  self.key_pos[0] * self.cell_size + self.cell_size // 2),
#                 self.cell_size // 4
#             )

#         # agent
#         pygame.draw.circle(
#             canvas, (50, 100, 255),
#             (self.agent_pos[1] * self.cell_size + self.cell_size // 2,
#              self.agent_pos[0] * self.cell_size + self.cell_size // 2),
#             self.cell_size // 3
#         )

#         # key indicator on agent if has_key
#         if self.has_key:
#             pygame.draw.circle(
#                 canvas, (255, 215, 0),
#                 (self.agent_pos[1] * self.cell_size + self.cell_size // 2,
#                  self.agent_pos[0] * self.cell_size + self.cell_size // 2),
#                 self.cell_size // 6
#             )

#         # info panel
#         info_panel = pygame.Surface((self.info_width, self.grid_size))
#         info_panel.fill((245, 245, 245))
#         font = pygame.font.SysFont("arial", 18)

#         texts = [
#             f"Phase: {self.phase}",
#             f"Episode: {self.current_episode}",
#             f"Steps: {self.steps}",
#             f"Reward (last): {self.last_reward}",
#             f"Has Key: {'Yes' if self.has_key else 'No'}",
#             "",
#             f"FPS: {'∞' if self.unlimited_fps else self.framerate}",
#             "",
#             "Controls:",
#             "↑ - Increase FPS",
#             "↓ - Decrease FPS",
#             "U - Toggle Unlimited FPS",
#             "ESC - Quit",
#             "",
#             "Actions:",
#             "0 - South",
#             "1 - North",
#             "2 - East",
#             "3 - West",
#             "4 - Pickup",
#             "5 - Unlock"
#         ]

#         y = 20
#         for t in texts:
#             txt_surface = font.render(t, True, (0, 0, 0))
#             info_panel.blit(txt_surface, (12, y))
#             y += 26

#         self.window.blit(canvas, (0, 0))
#         self.window.blit(info_panel, (self.grid_size, 0))
#         pygame.display.update()

#         if not self.unlimited_fps:
#             self.clock.tick(self.framerate)

#     def render(self):
#         if self.render_mode == "human":
#             self._render_frame()

#     def close(self):
#         if self.window is not None:
#             pygame.display.quit()
#             pygame.quit()



#with phase 4 and 5

# lock_key_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys


class LockKeyEnv(gym.Env):
    """
    Lock & Key environment (10x10) with unified integer-encoded observation:
    (agent_r, agent_c, key_r, key_c, lock_r, lock_c, has_key) -> single int

    Phases:
      1 -> fixed agent, fixed key, fixed lock  (default)
      2 -> random agent & random key, fixed lock
      3 -> random agent, random key, random lock
      4 -> stabilize & tune (same as above but used for hyperparam tuning)
      5 -> Phase 5 adds a single random-walking enemy (hazard). Enemy moves AFTER agent.
           Collision immediately terminates the episode with a penalty.
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode=None, size=6, phase=1, seed=None):
        super().__init__()
        assert size == 6, "This environment instance expects a 10x10 grid (size=10)."
        assert phase in (1, 2, 3, 4, 5), "phase must be 1..5."

        # grid config
        self.size = size
        self.num_cells = self.size * self.size
        # total states: agent_pos * key_pos * lock_pos * has_key(2)
        # = size^3 * 2
        self.total_states = self.num_cells * self.num_cells * self.num_cells * 2

        # action / observation
        self.action_space = spaces.Discrete(6)  # 0:S,1:N,2:E,3:W,4:Pickup,5:Unlock
        self.observation_space = spaces.Discrete(self.total_states)

        # rendering attributes
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 800
        self.grid_size = 500
        self.info_width = self.window_size - self.grid_size
        self.cell_size = self.grid_size // self.size

        # default (scaled/adjusted positions for 10x10)
        # self._default_agent_pos = np.array([self.size - 1, 0])  # bottom-left (9,0)
        # self._default_key_pos = np.array([8, 5])               # lower-mid-right (8,5)
        # self._default_lock_pos = np.array([0, 9])              # top-right-ish (0,9)


        self._default_agent_pos = np.array([5, 0])  # bottom-left (5,0)
        self._default_key_pos = np.array([5, 3])    # near bottom center
        self._default_lock_pos = np.array([0, 5])   # top-right corner

        # walls (adapted for 10x10, fixed layout)
        self.walls = {
            (0, 2), (1, 1), (1, 4),
            (2, 3), (3, 0), (3, 2),
            (4, 4)
        }


        # runtime state
        self.agent_pos = None
        self.key_pos = None
        self.lock_pos = self._default_lock_pos.copy()
        self.has_key = False
        self.steps = 0
        self.last_reward = 0
        self.current_episode = 0

        # enemy (phase 5)
        self.enemy_pos = None  # numpy array [r,c] or None when no enemy
        self.enemy_active = False

        # phase and RNG
        self.phase = phase
        self._rng = np.random.default_rng(seed)

        # render control
        self.framerate = 5
        self.unlimited_fps = False
        
    def render_rgb_array(self, frame_size=50):
        """
        Return a frame as RGB numpy array (for video recording)
        """
        frame = np.zeros((self.size*frame_size, self.size*frame_size, 3), dtype=np.uint8)

        # Example: draw agent (red)
        ax, ay = self.agent_pos
        frame[ay*frame_size:(ay+1)*frame_size, ax*frame_size:(ax+1)*frame_size] = [255,0,0]

        # Draw key (yellow)
        kx, ky = self.key_pos
        frame[ky*frame_size:(ky+1)*frame_size, kx*frame_size:(kx+1)*frame_size] = [255,255,0]

        # Draw lock (green)
        lx, ly = self.lock_pos
        frame[ly*frame_size:(ly+1)*frame_size, lx*frame_size:(lx+1)*frame_size] = [0,255,0]

        # Draw enemy if exists (blue)
        if hasattr(self, "enemy_pos"):
            ex, ey = self.enemy_pos
            frame[ey*frame_size:(ey+1)*frame_size, ex*frame_size:(ex+1)*frame_size] = [0,0,255]

        return frame


    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        # update rng if seed provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Phase-specific initialization
        if self.phase == 1:
            # Fixed agent, fixed key, fixed lock
            self.agent_pos = self._default_agent_pos.copy()
            self.key_pos = self._default_key_pos.copy()
            self.lock_pos = self._default_lock_pos.copy()
        elif self.phase == 2:
            # Random agent & random key, fixed lock
            self.lock_pos = self._default_lock_pos.copy()
            self.agent_pos = self._random_free_cell(exclude={tuple(self.lock_pos)})
            self.key_pos = self._random_free_cell(exclude={tuple(self.lock_pos), tuple(self.agent_pos)})
        elif self.phase in (3, 4):
            # Random agent, random key, random lock
            self.lock_pos = self._random_free_cell(exclude=set())
            self.agent_pos = self._random_free_cell(exclude={tuple(self.lock_pos)})
            self.key_pos = self._random_free_cell(exclude={tuple(self.lock_pos), tuple(self.agent_pos)})
        elif self.phase == 5:
            # Random agent, random key, random lock, spawn enemy
            self.lock_pos = self._random_free_cell(exclude=set())
            self.agent_pos = self._random_free_cell(exclude={tuple(self.lock_pos)})
            self.key_pos = self._random_free_cell(exclude={tuple(self.lock_pos), tuple(self.agent_pos)})
            # spawn enemy in a free cell
            exclude_set = {tuple(self.lock_pos), tuple(self.agent_pos), tuple(self.key_pos)}.union(self.walls)
            self.enemy_pos = self._random_free_cell(exclude=exclude_set)
            self.enemy_active = True
        else:
            raise ValueError("Invalid phase; must be 1..5")

        self.has_key = False
        self.steps = 0
        self.last_reward = 0

        obs = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()
        return obs, {}

    def step(self, action):
        """
        Returns: obs, reward, terminated, truncated, info
        Agent moves first. Then environment handles auto-pickup/unlock.
        If phase==5 and enemy is active, enemy moves after agent; collision checked after enemy move.
        """
        self.steps += 1
        reward = -1  # default step penalty
        terminated = False
        truncated = False

        # movement mapping: 0=S,1=N,2=E,3=W
        move_map = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
        if action in move_map:
            dr, dc = move_map[action]
            new_pos = self.agent_pos + np.array([dr, dc])
            if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size
                    and (new_pos[0], new_pos[1]) not in self.walls):
                self.agent_pos = new_pos
            else:
                # bump into wall or out of bounds: keep -1 reward
                reward = -1

        elif action == 4:  # Manual pickup
            if (not self.has_key) and np.array_equal(self.agent_pos, self.key_pos):
                self.has_key = True
                reward = 10
            else:
                reward = -10

        elif action == 5:  # Manual unlock
            if np.array_equal(self.agent_pos, self.lock_pos):
                if self.has_key:
                    reward = 20
                    terminated = True
                else:
                    reward = -10
            else:
                reward = -10

        # AUTO PICKUP (key remains at its randomized position in the state)
        if (not self.has_key) and np.array_equal(self.agent_pos, self.key_pos):
            self.has_key = True
            reward += 10  # extra for stepping onto key

        # AUTO UNLOCK
        if self.has_key and np.array_equal(self.agent_pos, self.lock_pos):
            reward += 20
            terminated = True

        # If Phase 5, move enemy AFTER agent moves and after pickup/unlock processing
        if self.phase == 5 and self.enemy_active and not terminated:
            self._move_enemy_random()
            # collision check: if enemy and agent overlap -> terminate with penalty
            if np.array_equal(self.enemy_pos, self.agent_pos):
                reward -= 10  # penalty (net effect: step penalty + collision penalty)
                terminated = True

        self.last_reward = reward
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    # -------------------------
    # Helpers: random free cell and encoding/decoding
    # -------------------------
    def _random_free_cell(self, exclude=None):
        """
        Return a random (r,c) numpy array that's not in walls and not in exclude set.
        exclude can be a set of (r,c) tuples.
        """
        if exclude is None:
            exclude = set()
        # ensure exclude includes walls as tuples for convenience
        exclude = set(exclude)
        # add walls to exclusion to avoid sampling them
        attempts = 0
        while True:
            attempts += 1
            r = int(self._rng.integers(0, self.size))
            c = int(self._rng.integers(0, self.size))
            if (r, c) in self.walls:
                continue
            if (r, c) in exclude:
                continue
            return np.array([r, c])

    def _move_enemy_random(self):
        """Enemy takes one random valid step (or stays). Does not step on walls or out of bounds."""
        if self.enemy_pos is None:
            return
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]  # include 'stay' option
        self._rng.shuffle(directions)
        for dr, dc in directions:
            cand = self.enemy_pos + np.array([dr, dc])
            rr, cc = int(cand[0]), int(cand[1])
            if 0 <= rr < self.size and 0 <= cc < self.size and (rr, cc) not in self.walls:
                # accept this move
                self.enemy_pos = np.array([rr, cc])
                return
        # if none valid, stays put

    def _get_obs(self):
        """
        Flatten (agent_idx, key_idx, lock_idx, has_key) into a single integer:
        idx = (((agent_idx * num_cells) + key_idx) * num_cells + lock_idx) * 2 + has_key
        """
        ar, ac = int(self.agent_pos[0]), int(self.agent_pos[1])
        kr, kc = int(self.key_pos[0]), int(self.key_pos[1])
        lr, lc = int(self.lock_pos[0]), int(self.lock_pos[1])

        agent_idx = ar * self.size + ac
        key_idx = kr * self.size + kc
        lock_idx = lr * self.size + lc
        idx = (((agent_idx * self.num_cells) + key_idx) * self.num_cells + lock_idx) * 2 + int(self.has_key)
        return int(idx)

    @staticmethod
    def decode_obs(obs, size=6):
        """
        Decode the flattened observation integer back to
        (agent_r, agent_c, key_r, key_c, lock_r, lock_c, has_key)
        """
        num_cells = size * size
        has_key = obs % 2
        tmp = obs // 2
        lock_idx = tmp % num_cells
        tmp = tmp // num_cells
        key_idx = tmp % num_cells
        agent_idx = tmp // num_cells

        ar, ac = divmod(agent_idx, size)
        kr, kc = divmod(key_idx, size)
        lr, lc = divmod(lock_idx, size)
        return (ar, ac, kr, kc, lr, lc, int(has_key))

    # -------------------------
    # Rendering (pygame)
    # -------------------------
    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.grid_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    sys.exit()
                elif event.key == pygame.K_UP:
                    self.framerate += 5
                    self.unlimited_fps = False
                elif event.key == pygame.K_DOWN:
                    self.framerate = max(1, self.framerate - 5)
                    self.unlimited_fps = False
                elif event.key == pygame.K_u:
                    self.unlimited_fps = not self.unlimited_fps

        # draw grid background
        canvas = pygame.Surface((self.grid_size, self.grid_size))
        canvas.fill((230, 230, 230))
        for x in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0),
                             (0, x * self.cell_size),
                             (self.grid_size, x * self.cell_size), 1)
            pygame.draw.line(canvas, (0, 0, 0),
                             (x * self.cell_size, 0),
                             (x * self.cell_size, self.grid_size), 1)

        # walls (dark blocks)
        for (r, c) in self.walls:
            pygame.draw.rect(canvas, (80, 80, 80),
                             pygame.Rect(c * self.cell_size, r * self.cell_size,
                                         self.cell_size, self.cell_size))

        # lock (red square)
        pygame.draw.rect(
            canvas, (200, 50, 50),
            pygame.Rect(
                self.lock_pos[1] * self.cell_size + self.cell_size // 6,
                self.lock_pos[0] * self.cell_size + self.cell_size // 6,
                int(self.cell_size * 0.66), int(self.cell_size * 0.66)
            )
        )

        # key (gold circle) if not held
        if not self.has_key:
            pygame.draw.circle(
                canvas, (255, 215, 0),
                (int(self.key_pos[1] * self.cell_size + self.cell_size / 2),
                 int(self.key_pos[0] * self.cell_size + self.cell_size / 2)),
                max(4, self.cell_size // 5)
            )

        # agent (blue circle)
        pygame.draw.circle(
            canvas, (50, 100, 255),
            (int(self.agent_pos[1] * self.cell_size + self.cell_size / 2),
             int(self.agent_pos[0] * self.cell_size + self.cell_size / 2)),
            max(6, self.cell_size // 3)
        )

        # key indicator on agent if has_key
        if self.has_key:
            pygame.draw.circle(
                canvas, (255, 215, 0),
                (int(self.agent_pos[1] * self.cell_size + self.cell_size / 2),
                 int(self.agent_pos[0] * self.cell_size + self.cell_size / 2)),
                max(3, self.cell_size // 7)
            )

        # enemy emoji (Phase 5 only) - draw after other things so it's visible
        if self.phase == 5 and self.enemy_active and self.enemy_pos is not None:
            # render emoji using font (may depend on system fonts supporting emoji)
            try:
                font = pygame.font.SysFont("arial", self.cell_size)  # size ~ cell_size
                emoji_surf = font.render("👾", True, (0, 0, 0))
                # center emoji in the cell
                ex = int(self.enemy_pos[1] * self.cell_size + (self.cell_size - emoji_surf.get_width()) / 2)
                ey = int(self.enemy_pos[0] * self.cell_size + (self.cell_size - emoji_surf.get_height()) / 2)
                canvas.blit(emoji_surf, (ex, ey))
            except Exception:
                # fallback: draw a small red square if emoji can't render
                pygame.draw.rect(
                    canvas, (180, 0, 180),
                    pygame.Rect(self.enemy_pos[1] * self.cell_size + self.cell_size // 4,
                                self.enemy_pos[0] * self.cell_size + self.cell_size // 4,
                                self.cell_size // 2, self.cell_size // 2)
                )

        # side info panel
        info_panel = pygame.Surface((self.info_width, self.grid_size))
        info_panel.fill((245, 245, 245))
        font = pygame.font.SysFont("arial", 18)

        texts = [
            f"Phase: {self.phase}",
            f"Episode: {self.current_episode}",
            f"Steps: {self.steps}",
            f"Reward (last): {self.last_reward}",
            f"Has Key: {'Yes' if self.has_key else 'No'}",
            "",
            f"FPS: {'∞' if self.unlimited_fps else self.framerate}",
            "",
            "Controls:",
            "↑ - Increase FPS",
            "↓ - Decrease FPS",
            "U - Toggle Unlimited FPS",
            "ESC - Quit",
            "",
            "Actions:",
            "0 - South",
            "1 - North",
            "2 - East",
            "3 - West",
            "4 - Pickup",
            "5 - Unlock"
        ]

        # draw text lines
        y = 12
        for t in texts:
            try:
                txt_surface = font.render(t, True, (0, 0, 0))
                info_panel.blit(txt_surface, (8, y))
            except Exception:
                # in rare cases fallback to skip
                pass
            y += 24

        # blit canvas + panel
        self.window.blit(canvas, (0, 0))
        self.window.blit(info_panel, (self.grid_size, 0))
        pygame.display.update()

        if not self.unlimited_fps:
            self.clock.tick(self.framerate)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Quick local test if run as script
if __name__ == "__main__":
    env = LockKeyEnv(render_mode="human", size=6, phase=5)
    env.current_episode = 1
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        steps += 1
    env.close()
