import numpy as np
import gymnasium as gym

from BotTrajectories.run_configs import get_config, ENV_NAME_AND_TYPES





def make_botparkenv(config_type):
    def make_env(**kwargs):
        config = get_config(config_type)
        if kwargs is not None:
            for key in kwargs.keys:
                if key not in config.keys:
                    raise Exception("Invalid argument passed to the BotParkingEnv constructor.")
                if isinstance(kwargs[key], str) and not isinstance(config[key], str):
                    config[key] = eval(kwargs[key])
                    continue
                config[key] = kwargs[key]
        return BotParkingEnv(initial_position_generator = config["startpos_generator"],
                             rewarder = config["rewarder"],
                             goal_checker = config["goal_checker"],
                             oob_checker = config["oob_checker"],
                             logger_visualizer = config["logvis"],
                             render_mode = "logger_visualizer",
                             u_scale = config["u_scale"],
                             w_scale = config["w_scale"],
                             epsilon = config["epsilon"],
                             dt = config["dt"],
                             steps_per_episode = config["steps_per_ep"],
                             debug = config["debug_env"])
    return make_env

def register_envs():
    for name, config_type in ENV_NAME_AND_TYPES.items():
        gym.register(name, entry_point=make_botparkenv(config_type=config_type))



class BotParkingEnv(gym.Env):
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render_modes': ['console', 'logger_visualizer', 'no_rendering']}

    def __init__(self, initial_position_generator, rewarder, goal_checker,
                oob_checker, logger_visualizer,
                render_mode="logger_visualizer",
                u_scale=1.0, w_scale=1.0, epsilon=1e-7, dt=0.05, steps_per_episode=100,
                debug=False):
        super(BotParkingEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

        #obs = (e, cosa, sina, cosd, sind, T-t)
        self.observation_space = gym.spaces.Box(low=np.array([0.0,-1.0,-1.0,-1.0,-1.0, 0.0]),
                                                high=np.array([1.0,1.0,1.0,1.0,1.0, 1.0]),
                                                dtype=np.single)

        self.startpos_gen = initial_position_generator
        self.__cartesian_pos = np.array([0,0,0], dtype=np.single)
        self.__polar_pos = np.array([0,0,0], dtype=np.single)
        self.xyphi_pos = initial_position_generator.get_startpos(cartesian=True)
        self.time_remaining = np.single(1.0)

        self.r_fn = rewarder
        self.goal_check = goal_checker
        self.oob_check = oob_checker

        self.action_scale = np.array([u_scale, w_scale], np.single)

        self.eps = np.single(epsilon)
        self.dt = np.single(dt)

        self.step_n = 0
        self.steps_per_episode = steps_per_episode
        self.ep_time_delta = np.single(1.0 / steps_per_episode)

        if render_mode not in self.metadata['render_modes']:
            raise Exception("Error! Rendering mode not valid.")
            #self.metadata["render_modes"].append(render_mode)
        self.render_mode = render_mode
        self.log_vis = logger_visualizer
        self.is_rendering_episode = False
        self._done = False

        self.additional_info = {}

        self.debug = debug

    @property
    def xyphi_pos(self):
        return self.__cartesian_pos.copy()

    @xyphi_pos.setter
    def xyphi_pos(self, value:np.ndarray):
        self.__cartesian_pos = value.copy()
        e = np.sqrt(np.square(value[0]) + np.square(value[1]))
        d = np.arctan2(- value[1], - value[0])
        a = d - value[2]
        if a < -np.pi: a += 2 * np.pi
        elif a > np.pi: a -= 2 * np.pi
        self.__polar_pos = np.array([e, a, d])

    @property
    def polar_pos(self):
        return self.__polar_pos.copy()

    def get_obs(self):
        pos = self.polar_pos
        return np.array([pos[0],
                        np.cos(pos[1]),
                        np.sin(pos[1]),
                        np.cos(pos[2]),
                        np.sin(pos[2]),
                        self.time_remaining])

    def get_info(self):
        info = {}
        info.update(self.additional_info)
        self.additional_info.clear()
        return info

    def add_info(self, info:dict):
        self.additional_info.update(info)

    def reset(self, seed=None, options=None):
        self.xyphi_pos = self.startpos_gen.get_startpos(cartesian=True)
        self.time_remaining = np.single(1.0)
        self.step_n = 0
        self._done = False
        self.log_vis.reset_trajectory()
        self.is_rendering_episode = False
        return self.get_obs(), self.get_info()

    def update_state(self, action):
        pos = self.xyphi_pos
        dx = action[0] * np.cos(pos[2]) * self.dt
        dy = action[0] * np.sin(pos[2]) * self.dt
        dp = action[1] * self.dt

        dxdydp = np.array([dx, dy, dp])
        pos += dxdydp
        #pos[2] = np.arctan2(np.sin(pos[2]), np.cos(pos[2]))
        if pos[2] > np.pi: pos[2] -= 2 * np.pi
        elif pos[2] < -np.pi: pos[2] += 2 * np.pi

        if self.debug:
            if pos[2] > np.pi or pos[2] < -np.pi:
                self.add_info({"PhiBigUpdate" : f"Huge update to phi. Outside [-pi,pi] after reduction/addition of 2*pi. Phi: {pos[2]}."})

        self.xyphi_pos = pos

    def step(self, action:np.ndarray):
        action_in = action
        action = np.multiply(np.clip(action, -1.0, 1.0), self.action_scale) #clipping and rescaling
        s_old = self.polar_pos
        old_cartesian = self.xyphi_pos
        self.update_state(action)
        s_new = self.polar_pos

        is_g = self.goal_check(s_new)
        is_oob = self.oob_check(s_new)
        is_terminated = is_g or is_oob

        r = self.r_fn(old_state=s_old, action=action, new_state=s_new, is_goal=is_g, is_oob=is_oob)

        self.time_remaining -= self.ep_time_delta
        self.step_n += 1
        is_truncated = self.step_n == self.steps_per_episode # and not is_terminated

        self.add_info({})
        obs = self.get_obs()

        self.log_vis.add_point(old_cartesian, s_old, obs[1:5], action, r, 1.0 - self.time_remaining)
        self._done = is_terminated or is_truncated
        if self._done:
            self.log_vis.end_current_trajectory(is_g)

        return obs, r.item(), bool(is_terminated), bool(is_truncated), self.get_info() #r & is_term/trun are np objs, must be native

    def render(self):
        #1. render_mode check, possibly raise error
        #2. rendering
        if self.render_mode == "logger_visualizer":
            if not self.is_rendering_episode:
                if self.step_n > 1:
                    print(f"Warning! First call to env.render on an episode, but environment has progressed {self.step_n} steps.")
                self.is_rendering_episode = True
            if self._done:
                self.log_vis.create_and_save_plot()

    def close(self):
        #handle shutting-down-processes if necessary
        pass