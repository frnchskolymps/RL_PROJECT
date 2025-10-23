# simulator.py
# Full integrated simulator: menu -> load Phase{N}locknkey.pkl -> playback using LockKeyEnv visuals

import os, sys, time, pickle
import numpy as np
import pygame
import torch
import torch.nn as nn
from collections import deque
import gymnasium as gym
from gymnasium import spaces

# ---------------- Helper Functions ----------------
def load_q_table_for_phase(phase):
    """
    Safely loads the Q-table or policy dictionary for the given phase.
    Works for both old-style (direct Q dict) and new-style wrapped dicts.
    """
    candidates = [f"Phase{phase}locknkey.pkl", f"phase{phase}locknkey.pkl"]
    fname = next((c for c in candidates if os.path.exists(c)), None)
    if not fname:
        raise FileNotFoundError(f"No pickle file found for phase {phase} in {os.getcwd()}")

    try:
        with open(fname, 'rb') as f:
            q_table = pickle.load(f)

        # Fix numpy integer keys
        fixed_q_table = {}
        for k, v in q_table.items():
            if isinstance(k, tuple):
                fixed_key = tuple(int(x) for x in k)
                fixed_q_table[fixed_key] = v
        q_table = fixed_q_table

        print(f"[LOADED] {fname} | Entries in Q-table: {len(q_table)}")
        try:
            sample_key = next(iter(q_table.keys()))
            print("Sample key:", sample_key)
        except Exception:
            print("[WARN] Q-table is empty or unreadable.")
        return q_table

    except Exception as e:
        print(f"[ERROR] Could not load Q-table for phase {phase}: {e}")
        raise

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_pkl_path(phase_num):
    return os.path.join(BASE_DIR, f"phase{phase_num}locknkey.pkl")

def pygame_safe_init():
    if not pygame.get_init():
        pygame.init()
        pygame.display.init()

def draw_button(screen, rect, text, font, color=(180,220,255), border=2):
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, (0,0,0), rect, border)
    txt = font.render(text, True, (0,0,0))
    screen.blit(txt, (rect.x + (rect.width - txt.get_width())//2, rect.y + (rect.height - txt.get_height())//2))

def draw_input_box(screen, rect, text, font, active):
    color = (200,255,200) if active else (255,255,255)
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, (0,0,0), rect, 2)
    txt_surface = font.render(str(text), True, (0,0,0))
    screen.blit(txt_surface, (rect.x+5, rect.y+5))

# ---------------- LockKeyEnv ----------------
class LockKeyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode="human", size=6, phase=1, seed=None):
        super().__init__()
        assert phase in (1,2,3), "Phase must be 1–3"
        self.size = size
        self.phase = phase
        self._rng = np.random.default_rng(seed)

        # Window/grid layout
        self.window_size = 800
        self.grid_size = 500
        self.info_width = self.window_size - self.grid_size
        self.cell_size = self.grid_size // self.size

        # Spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(self.size**6)

        # Defaults and walls
        self._default_agent_pos = np.array([0, 0])
        self._default_key_pos = np.array([2, 5])
        self._default_lock_pos = np.array([5, 5])
        self.walls = {(0,3),(1,3),(2,3),(3,1),(4,2),(4,3)}
        self._fixed_phase1_walls = self.walls.copy()

        # Enemy & trail
        self.enemy_pos = None
        self.enemy_active = (phase == 3)
        self.trail = deque(maxlen=4)

        # State
        self.agent_pos = None
        self.key_pos = None
        self.lock_pos = None
        self.has_key = False
        self.steps = 0
        self.last_reward = 0
        self.current_episode = 0

        # distance trackers
        self.prev_enemy_dist = None
        self.prev_key_dist = None
        self.prev_door_dist = None

        # rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # speed control
        self.framerate = 5
        self.unlimited_fps = False
        self.speed_multiplier = 1.0
        self.speed_levels = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60]
        self.current_speed_idx = 1

        self.max_steps = 200

        # images (to be loaded)
        self.player_img = None
        self.key_img = None
        self.door_img = None
        self.enemy_img = None
        self.obstacle_img = None

        # reward config
        self.STEP_PENALTY = -0.1
        self.KEY_REWARD = 10.0
        self.DOOR_REWARD = 20.0
        self.CAUGHT_PENALTY = -15.0
        self.SURVIVAL_BONUS = 0.00
        self.DIST_SCALE = 0.05
        self.APPROACH_SCALE = 0.1

    # ---------- Environment Methods ----------
    def _load_images(self):
        pygame_safe_init()
        if not pygame.display.get_init() or pygame.display.get_surface() is None:
            pygame.display.set_mode((1,1))

        def try_load(name):
            try:
                if os.path.exists(name):
                    img = pygame.image.load(name).convert_alpha()
                    print(f"[OK] Loaded {name}")
                    return img
                return None
            except Exception as e:
                print(f"[WARN] Failed to load {name}: {e}")
                return None

        self.player_img = try_load("player.png")
        self.key_img = try_load("key.png")
        self.door_img = try_load("door.png")
        self.enemy_img = try_load("enemy.png") or try_load("enemy.pmg")
        self.obstacle_img = try_load("obstacle.png")

        for name, img in [("player.png", self.player_img), ("key.png", self.key_img),
                          ("door.png", self.door_img), ("enemy", self.enemy_img), ("obstacle.png", self.obstacle_img)]:
            if img is None:
                print(f"[INFO] {name} not found -> placeholder will be used.")

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.close(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: self._increase_speed()
                elif event.key == pygame.K_DOWN: self._decrease_speed()
                elif event.key == pygame.K_ESCAPE: self.close(); sys.exit()

    def reset(self, seed=None, options=None):
        if seed is not None: self._rng = np.random.default_rng(seed)

        if self.phase == 1:
            self.agent_pos = self._default_agent_pos.copy()
            self.key_pos = self._default_key_pos.copy()
            self.lock_pos = self._default_lock_pos.copy()
            self.walls = self._fixed_phase1_walls.copy()

        elif self.phase == 2:
            self.lock_pos = self._default_lock_pos.copy()
            self.agent_pos = self._random_free_cell(exclude={tuple(self.lock_pos)})
            self.key_pos = self._random_free_cell(exclude={tuple(self.lock_pos), tuple(self.agent_pos)})
            self.walls = self._fixed_phase1_walls.copy()  # could also randomize mildly here
        else:
            # phase 3
            self.agent_pos = self._random_free_cell()
            self.key_pos = self._random_free_cell(exclude={tuple(self.agent_pos)})
            self.lock_pos = self._random_free_cell(exclude={tuple(self.agent_pos), tuple(self.key_pos)})
            self.enemy_active = True
            self.enemy_pos = self._random_free_cell(exclude={tuple(self.agent_pos), tuple(self.key_pos), tuple(self.lock_pos)})
            self.walls = self._random_walls()


        self.has_key = False
        self.steps = 0
        self.last_reward = 0
        self.current_episode += 1

        self.prev_enemy_dist = self._manhattan(self.agent_pos, self.enemy_pos) if self.enemy_active else None
        self.prev_key_dist = self._manhattan(self.agent_pos, self.key_pos)
        self.prev_door_dist = self._manhattan(self.agent_pos, self.lock_pos)

        obs = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()
        return obs, {}

    def step(self, action):
        self.steps += 1
        reward = self.STEP_PENALTY
        terminated = False
        truncated = False

        move_map = {0:(1,0), 1:(-1,0), 2:(0,1), 3:(0,-1)}
        if action in move_map:
            dr, dc = move_map[action]
            new_pos = self.agent_pos + np.array([dr, dc])
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and (tuple(new_pos) not in self.walls):
                self.trail.appendleft(self.agent_pos.copy())
                self.agent_pos = new_pos

        key_dist = self._manhattan(self.agent_pos, self.key_pos)
        door_dist = self._manhattan(self.agent_pos, self.lock_pos)

        if not self.has_key:
            delta = self.prev_key_dist - key_dist
            reward += self.APPROACH_SCALE * delta
            self.prev_key_dist = key_dist
        else:
            delta = self.prev_door_dist - door_dist
            reward += self.APPROACH_SCALE * delta
            self.prev_door_dist = door_dist

        if not self.has_key and np.array_equal(self.agent_pos, self.key_pos):
            self.has_key = True
            reward += self.KEY_REWARD

        if self.has_key and np.array_equal(self.agent_pos, self.lock_pos):
            reward += self.DOOR_REWARD
            terminated = True

        if self.phase == 3 and self.enemy_active:
            if len(self.trail) >= 3:
                self.enemy_pos = self.trail[2].copy()
            else:
                self._move_enemy_random()

            dist_now = self._manhattan(self.agent_pos, self.enemy_pos)
            if self.prev_enemy_dist is not None:
                delta = dist_now - self.prev_enemy_dist
                reward += self.DIST_SCALE * delta
            self.prev_enemy_dist = dist_now

            if np.array_equal(self.enemy_pos, self.agent_pos):
                reward += self.CAUGHT_PENALTY
                terminated = True

        if not terminated: reward += self.SURVIVAL_BONUS

        info = {}
        if self.steps >= self.max_steps: truncated = True; info['timeout'] = True
        if terminated:
            if self.phase == 3 and self.enemy_active and np.array_equal(self.enemy_pos, self.agent_pos):
                info['caught'] = True
            elif self.has_key and np.array_equal(self.agent_pos, self.lock_pos):
                info['unlocked'] = True

        self.last_reward = reward
        if self.render_mode == "human": self._render_frame()
        return self._get_obs(), reward, terminated, truncated, info

    # ---------- Internal Helpers ----------
    def _random_free_cell(self, exclude=None):
        if exclude is None: exclude = set()
        while True:
            r, c = self._rng.integers(0, self.size, size=2)
            if (r,c) not in self.walls and (r,c) not in exclude: return np.array([r,c])
            
    def _random_walls(self, n_walls=8):
        walls = set()
        attempts = 0
        max_attempts = 200
        exclude = {tuple(self.agent_pos), tuple(self.key_pos), tuple(self.lock_pos)}
        if hasattr(self, "enemy_pos"):
            exclude.add(tuple(self.enemy_pos))
        while len(walls) < n_walls and attempts < max_attempts:
            r, c = np.random.randint(0, self.size, size=2)
            if (r, c) not in exclude and (r, c) not in walls:
                walls.add((r, c))
            attempts += 1
        return list(walls)


    def _manhattan(self, p1, p2):
        if p1 is None or p2 is None: return 0
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def _move_enemy_random(self):
        directions = [(1,0),(-1,0),(0,1),(0,-1),(0,0)]
        self._rng.shuffle(directions)
        for dr, dc in directions:
            cand = self.enemy_pos + np.array([dr, dc])
            rr, cc = int(cand[0]), int(cand[1])
            if 0 <= rr < self.size and 0 <= cc < self.size and (rr, cc) not in self.walls:
                self.enemy_pos = np.array([rr, cc])
                return

    def _get_obs(self):
        return np.array([*self.agent_pos, *self.key_pos, *self.lock_pos, int(self.has_key)])

    # ---------- Rendering ----------
    def _render_frame(self):
        pygame_safe_init()
        if self.player_img is None: self._load_images()
        if self.window is None: self.window = pygame.display.set_mode((self.window_size, self.grid_size))
        if self.clock is None: self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.grid_size, self.grid_size))
        canvas.fill((137,207,240))

        # grid lines
        for x in range(self.size+1):
            pygame.draw.line(canvas, (0,0,0), (0, x*self.cell_size), (self.grid_size, x*self.cell_size), 1)
            pygame.draw.line(canvas, (0,0,0), (x*self.cell_size, 0), (x*self.cell_size, self.grid_size), 1)

        # walls
        for r,c in self.walls:
            rect = pygame.Rect(c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size)
            if self.obstacle_img:
                canvas.blit(pygame.transform.scale(self.obstacle_img,(self.cell_size,self.cell_size)), rect.topleft)
            else: pygame.draw.rect(canvas,(100,100,100),rect)

        # key
        if not self.has_key:
            rect = pygame.Rect(self.key_pos[1]*self.cell_size, self.key_pos[0]*self.cell_size, self.cell_size, self.cell_size)
            if self.key_img:
                canvas.blit(pygame.transform.scale(self.key_img,(self.cell_size,self.cell_size)), rect.topleft)
            else:
                pygame.draw.circle(canvas,(255,215,0), rect.center, max(6,self.cell_size//4))

        # door
        rect = pygame.Rect(self.lock_pos[1]*self.cell_size, self.lock_pos[0]*self.cell_size, self.cell_size, self.cell_size)
        if self.door_img:
            canvas.blit(pygame.transform.scale(self.door_img,(self.cell_size,self.cell_size)), rect.topleft)
        else: pygame.draw.rect(canvas,(200,50,50),rect)

        # agent
        rect = pygame.Rect(self.agent_pos[1]*self.cell_size, self.agent_pos[0]*self.cell_size, self.cell_size, self.cell_size)
        if self.player_img:
            canvas.blit(pygame.transform.scale(self.player_img,(self.cell_size,self.cell_size)), rect.topleft)
        else:
            pygame.draw.circle(canvas,(50,100,255), rect.center, max(6,self.cell_size//3))

        # enemy
        if self.phase==5 and self.enemy_active and self.enemy_pos is not None:
            rect = pygame.Rect(self.enemy_pos[1]*self.cell_size, self.enemy_pos[0]*self.cell_size, self.cell_size, self.cell_size)
            if self.enemy_img:
                canvas.blit(pygame.transform.scale(self.enemy_img,(self.cell_size,self.cell_size)), rect.topleft)
            else:
                pygame.draw.circle(canvas,(255,0,0), rect.center, max(6,self.cell_size//3))

        # info panel
        info_panel = pygame.Surface((self.info_width, self.grid_size))
        info_panel.fill((137,207,240))
        font = pygame.font.SysFont("verdana",18)
        header_font = pygame.font.SysFont("verdana", 24, bold=True)
        info_panel.blit(header_font.render("LOCK N KEY", True, (0,0,0)), (10,10))

        y_start_status = 50
        line_spacing = 28
        lines = [
            f"Phase: {self.phase}",
            f"Episode: {self.current_episode}",
            "",
            f"Steps: {self.steps}",
            f"Reward: {round(self.last_reward,2)}",
            f"Has Key: {'Yes' if self.has_key else 'No'}",
            "",
            f"Speed: ×{self.speed_multiplier}",
            "",
            "Controls:",
            "  UP: Increase Speed",
            "  DOWN: Decrease Speed",
            "  Close Window/ESC to Quit",
        ]
        for i,text in enumerate(lines):
            y_pos = y_start_status + i*line_spacing
            info_panel.blit(font.render(text,True,(0,0,0)),(10,y_pos))

        self.window.blit(canvas,(0,0))
        self.window.blit(info_panel,(self.grid_size,0))
        pygame.display.flip()

        if not self.unlimited_fps:
            effective_fps = max(1, int(self.framerate*self.speed_multiplier))
            self.clock.tick(effective_fps)

    def _increase_speed(self):
        if self.current_speed_idx < len(self.speed_levels)-1:
            self.current_speed_idx += 1
            self.speed_multiplier = self.speed_levels[self.current_speed_idx]

    def _decrease_speed(self):
        if self.current_speed_idx > 0:
            self.current_speed_idx -= 1
            self.speed_multiplier = self.speed_levels[self.current_speed_idx]

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()

# ---------------- Q-table Lookup ----------------
def get_q_for_state(q_table, obs, n_actions):
    obs_key = tuple(int(x) for x in obs)
    if obs_key in q_table: return np.array(q_table[obs_key], dtype=float)
    for key in q_table.keys():
        try:
            if isinstance(key,(tuple,list,np.ndarray)):
                if np.allclose(np.array(key,dtype=float), np.array(obs,dtype=float)):
                    return np.array(q_table[key],dtype=float)
        except: continue
    str_key = str(list(obs_key))
    if str_key in q_table: return np.array(q_table[str_key],dtype=float)
    return np.zeros(n_actions,dtype=float)

# ---------------- Actor-Critic ----------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc_out = nn.Linear(128,action_dim)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc_out(x)
        return torch.softmax(logits,dim=-1)

def load_actor_for_phase(phase,state_dim,action_dim):
    fname = f"actor_phase{phase}.pth"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"[ERROR] Actor model not found: {fname}")
    actor = ActorNetwork(state_dim,action_dim)
    actor.load_state_dict(torch.load(fname,map_location=torch.device('cpu')))
    actor.eval()
    print(f"[LOADED] Actor model for Phase {phase}: {fname}")
    return actor

# ---------------- Training File Resolver ----------------
def get_training_file(phase, algo_name):
    base1 = f"Phase{phase}_{algo_name.replace(' ','-')}"
    base2 = f"Phase{phase}_{algo_name.replace(' ','')}"
    candidates = [f"{base1}.pkl", f"{base2}.pkl", base1, base2, f"Phase{phase}locknkey.pkl"]
    for fname in candidates:
        if os.path.exists(fname):
            print(f"[FOUND] {fname}")
            return fname
    raise FileNotFoundError(f"[ERROR] No saved file found for Phase {phase} and algorithm '{algo_name}'. Tried: {candidates}")

# ---------------- Tabular Playback ----------------
def run_tabular_playback(phase, episodes, algo_name="Q-Learning", grid_size=6):
    try: fname = get_training_file(phase, algo_name)
    except FileNotFoundError as e: print(e); return

    try:
        with open(fname,"rb") as f:
            q_table = pickle.load(f)
        if isinstance(q_table, dict) and "q_table" in q_table: q_table = q_table["q_table"]
        print(f"[LOADED] {fname} | Entries in table: {len(q_table)}")
    except Exception as e: print(f"[ERROR] Could not load file {fname}: {e}"); return

    env = LockKeyEnv(render_mode='human', size=grid_size, phase=phase)
    env._load_images()
    clock = pygame.time.Clock()
    total_rewards = []
    successes = 0

    for ep in range(1, episodes+1):
        obs,_ = env.reset(seed=42)
        obs = np.array(obs)
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps<env.max_steps:
            env.handle_events(pygame.event.get())
            qvals = get_q_for_state(q_table, obs, env.action_space.n)
            action = int(np.argmax(qvals))
            obs, reward, terminated, truncated, info = env.step(action)
            obs = np.array(obs)
            episode_reward += reward
            steps += 1
            env._render_frame()
            clock.tick(int(env.framerate*env.speed_multiplier))
            if terminated or truncated: done=True

        if info.get('unlocked'): successes+=1; status="Door Unlocked"
        elif info.get('caught'): status="Caught"
        elif info.get('timeout'): status="Timeout"
        else: status="Ended"

        print(f"[{algo_name}] Ep {ep}/{episodes} | Reward={episode_reward:.2f} | {status}")
        total_rewards.append(episode_reward)
        time.sleep(0.3)

    env.close()
    print(f"\n[{algo_name}] Phase {phase} — Success Rate={successes/episodes*100:.1f}%, Avg Reward={np.mean(total_rewards):.2f}")

# ---------------- Actor-Critic Playback ----------------
def run_actorcritic_playback(phase, episodes, grid_size=6):
    env = LockKeyEnv(render_mode='human', size=grid_size, phase=phase)
    env._load_images()
    clock = pygame.time.Clock()

    try:
        fname = get_training_file(phase,"Actor-Critic")
        data = pickle.load(open(fname,"rb"))
        print(f"[LOADED] {fname} | Keys: {list(data.keys())}")
    except Exception as e: print(f"[ERROR] Could not load Actor–Critic file for Phase {phase}: {e}"); return

    policy = None
    if isinstance(data,dict):
        if 'pi' in data: policy=data['pi']
        elif 'policy' in data: policy=data['policy']
        else: raise ValueError(f"No policy found in Actor-Critic file. Keys: {list(data.keys())}")
    else: policy = data

    total_rewards=[]
    for ep in range(episodes):
        obs,_ = env.reset()
        done=False
        ep_reward=0
        steps=0

        while not done:
            clock.tick(int(env.framerate*env.speed_multiplier))
            env.handle_events(pygame.event.get())
            env._render_frame()
            state = tuple(int(x) for x in obs)
            if state in policy: action=int(np.argmax(policy[state]))
            else: action=env.action_space.sample(); print("[WARN] No policy entry for:",state)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done=terminated or truncated

        total_rewards.append(ep_reward)
        print(f"[Ep {ep+1}/{episodes}] Reward={ep_reward:.2f} | Door {'Unlocked' if info.get('unlocked') else 'Locked'}")

    env.close()
    print(f"[SUMMARY] Phase {phase}: Avg Reward={np.mean(total_rewards):.2f}")

# ---------------- Main Menu ----------------
def main_menu():
    pygame.init()
    W,H=980,640
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("Lock N Key — Algorithm Selection")
    font = pygame.font.SysFont('verdana',24)
    small = pygame.font.SysFont('verdana',16)
    header_font = pygame.font.SysFont('verdana',34,bold=True)
    clock = pygame.time.Clock()

    algos = ["Q-Learning","Monte Carlo","Actor-Critic"]
    algo_buttons = [pygame.Rect(100,200+i*80,250,60) for i in range(len(algos))]
    chosen_algo=None

    while chosen_algo is None:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: pygame.quit(); raise SystemExit()
            elif ev.type==pygame.MOUSEBUTTONDOWN:
                mx,my = ev.pos
                for i,rect in enumerate(algo_buttons):
                    if rect.collidepoint(mx,my): chosen_algo=algos[i]
        screen.fill((137,207,240))
        screen.blit(header_font.render("Choose Algorithm",True,(20,20,20)),(330,60))
        for i,rect in enumerate(algo_buttons):
            pygame.draw.rect(screen,(200,220,255),rect)
            pygame.draw.rect(screen,(0,0,0),rect,2)
            txt = font.render(algos[i],True,(0,0,0))
            screen.blit(txt,(rect.x+(rect.w-txt.get_width())//2, rect.y+10))
        pygame.display.update()
        clock.tick(30)

    # Phase selection + episodes
    phase_buttons=[pygame.Rect(120,160+i*80,160,60) for i in range(3)]
    exit_btn=pygame.Rect(700,500,160,60)
    input_box=pygame.Rect(350,500,200,40)
    active_box=False
    episodes="10"

    while True:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: pygame.quit(); raise SystemExit()
            elif ev.type==pygame.MOUSEBUTTONDOWN:
                mx,my=ev.pos
                for i,rect in enumerate(phase_buttons):
                    if rect.collidepoint(mx,my):
                        phase=i+1
                        pygame.display.quit()
                        return chosen_algo, phase, int(episodes) if episodes.isdigit() else 10
                if exit_btn.collidepoint(mx,my): pygame.quit(); raise SystemExit()
                active_box = input_box.collidepoint(mx,my)
            elif ev.type==pygame.KEYDOWN:
                if active_box:
                    if ev.key==pygame.K_BACKSPACE: episodes=episodes[:-1]
                    elif ev.unicode.isdigit(): episodes+=ev.unicode
                elif ev.key==pygame.K_ESCAPE: pygame.quit(); raise SystemExit()

        screen.fill((137,207,240))
        screen.blit(header_font.render(f"{chosen_algo} — Playback",True,(20,20,20)),(250,40))
        screen.blit(font.render("Select Phase (click) and set Episodes:",True,(20,20,20)),(120,100))
        for i,rect in enumerate(phase_buttons):
            pygame.draw.rect(screen,(200,220,255),rect)
            pygame.draw.rect(screen,(0,0,0),rect,2)
            txt=font.render(f"Phase {i+1}",True,(0,0,0))
            screen.blit(txt,(rect.x+(rect.w-txt.get_width())//2,rect.y+10))

        screen.blit(small.render("Episodes:",True,(10,10,10)),(260,510))
        pygame.draw.rect(screen,(200,255,200) if active_box else (255,255,255),input_box)
        pygame.draw.rect(screen,(0,0,0),input_box,2)
        txt_surface = font.render(str(episodes),True,(0,0,0))
        screen.blit(txt_surface,(input_box.x+5,input_box.y+5))

        pygame.draw.rect(screen,(255,150,150),exit_btn)
        pygame.draw.rect(screen,(0,0,0),exit_btn,2)
        screen.blit(font.render("Exit",True,(0,0,0)),(exit_btn.x+40,exit_btn.y+10))

        pygame.display.update()
        clock.tick(30)

# ---------------- Launcher ----------------
def launch():
    while True:
        res = main_menu()
        if res is None: break
        algo,phase,episodes=res
        print(f"Launching {algo} | Phase {phase} | Episodes {episodes}")
        if algo=="Q-Learning": run_tabular_playback(phase,episodes,"Q-Learning")
        elif algo=="Monte Carlo": run_tabular_playback(phase,episodes,"Monte Carlo")
        elif algo=="Actor-Critic": run_actorcritic_playback(phase,episodes)

if __name__=="__main__":
    launch()
