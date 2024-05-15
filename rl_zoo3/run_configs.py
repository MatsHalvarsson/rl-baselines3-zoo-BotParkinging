import numpy as np
from os.path import join, exists
from os import makedirs

from BotTrajectories.visualizations import Visualizer, LoggerVisualizer
from BotTrajectories.supporting_functions import StartPosGeneratorBase, GoalChecker, OOBChecker, RewardLyapEq11





ENV_NAME_AND_TYPES = {"BotParking-easy" : "easy"}

def get_save_dir(name=""):
    root_dir = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\SB3 Testing\experiment_plots"
    save_folder = name + "_experiment_"
    nbr = 1
    while exists(join(root_dir, save_folder + str(nbr))):
        nbr += 1
    dir = join(root_dir, save_folder + str(nbr))
    makedirs(dir, exist_ok=False)
    return dir

def make_easy_config():
    __config_name = "easy"
    config = {}

    env_max_steps_per_episode = 120
    num_state_vars = 3
    num_sinecosine_s_vars = 4
    num_c_vars = 2

    u_scale = 2.0
    w_scale = 2.5
    epsilon = 1e-7
    dt = 0.03

    debug_env = False

    n_vehicle_markers = 8
    file_name = "BotParking_easy"
    fig_name = file_name

    startpos_generator = StartPosGeneratorBase(fix_pos=[0.5, 0.0, 0.0])
    goal_checker = GoalChecker(polar_state_representation=True,
                               err_dist=0.05,
                               err_angl=np.pi / 18)
    oob_checker = OOBChecker(polar_state_representation=True,
                             oob_dist=1.00)
    rewarder = RewardLyapEq11(lambdagamma=1.0,
                              k=1.0)

    #plotting:

    controller_info = None #{}

    simulation_info = {"steps/ep" : str(env_max_steps_per_episode),
                       "$\u03B4t$" : str(dt),
                       "$\u03B5$" : str(epsilon),
                       "start" : str(startpos_generator)}

    training_info = None #{}

    reward_info = {"R" : str(rewarder),
                   "Goal" : str(goal_checker),
                   "OOB" : str(oob_checker)}

    plotting_kwargs = {"controller_info" : controller_info, "simulation_info" : simulation_info,
                        "training_info" : training_info, "reward_info" : reward_info,
                        "n_vehicle_markers" : n_vehicle_markers, "file_name" : file_name, "fig_name" : fig_name}

    #config["config_plotting_kwargs"] = plotting_kwargs

    visualizer = Visualizer(save_dir=get_save_dir(name=__config_name), n_vehicle_shows=n_vehicle_markers)

    logvis = LoggerVisualizer(visualizer, steps_per_episode=env_max_steps_per_episode, num_s_vars=num_state_vars,
                                num_sinecosine_s_vars=num_sinecosine_s_vars, num_c_vars=num_c_vars,
                                **plotting_kwargs)

    #env kwargs: (except render mode)
    config["startpos_generator"] = startpos_generator
    config["goal_checker"] = goal_checker
    config["oob_checker"] = oob_checker
    config["rewarder"] = rewarder
    config["u_scale"] = u_scale
    config["w_scale"] = w_scale
    config["epsilon"] = epsilon
    config["dt"] = dt
    config["steps_per_ep"] = env_max_steps_per_episode
    config["debug_env"] = debug_env

    config["visualizer"] = visualizer
    config["logvis"] = logvis

    return config


def get_config(config_type:str):
    assert config_type in ENV_NAME_AND_TYPES.values()

    if config_type == "easy":
        return make_easy_config()