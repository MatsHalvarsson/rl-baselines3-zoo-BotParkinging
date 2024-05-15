from matplotlib import pyplot as plt, ticker as tck
from os.path import join as ospjoin
import numpy as np





class TrajectoryExport:
    def __init__(self, traj_dict:dict):
        if not TrajectoryExport.check_compatibility(traj_dict):
            print("Error! Dict not compatible with TrajectoryExport format and cannot be exported.")
            raise Exception("Error! Dict not compatible with TrajectoryExport format and cannot be exported.")
        self._traj_dict = traj_dict

    @staticmethod
    def check_compatibility(trajectory_dict:dict):
        compatstring = ["polar_states", "cartesian_states", "control_inputs",
                        "sinecosine_states", "rewards", "time", "length",
                        "point_count", "reached_goal"]
        for key_name in compatstring:
            if key_name not in trajectory_dict:
                return False
        return True

    @property
    def cartesian_states(self):
        return self._traj_dict["cartesian_states"]

    @property
    def polar_states(self):
        return self._traj_dict["polar_states"]

    @property
    def sinecosine_states(self):
        return self._traj_dict["sinecosine_states"]

    @property
    def control_inputs(self):
        return self._traj_dict["control_inputs"]

    @property
    def rewards(self):
        return self._traj_dict["rewards"]

    @property
    def time(self):
        return self._traj_dict["time"]

    @property
    def length(self):
        return self._traj_dict["length"]

    @property
    def point_count(self):
        return self._traj_dict["point_count"]

    @property
    def reached_goal(self):
        return self._traj_dict["reached_goal"]



class Visualizer:
    def __init__(self, save_dir, n_vehicle_shows=5):
        self.vehicle_show_count = n_vehicle_shows
        self.save_dir = save_dir

    @staticmethod
    def vehicle_plot_triangle(x_center, y_center, phi, height, base):
        return [[x_center+np.cos(phi)*height/2,
        x_center-np.cos(phi)*height/2-np.cos(np.pi/2-phi)*base/2,
        x_center-np.cos(phi)*height/2+np.cos(np.pi/2-phi)*base/2,
        x_center+np.cos(phi)*height/2],
        [y_center+np.sin(phi)*height/2,
        y_center-np.sin(phi)*height/2+np.sin(np.pi/2-phi)*base/2,
        y_center-np.sin(phi)*height/2-np.sin(np.pi/2-phi)*base/2,
        y_center+np.sin(phi)*height/2]]

    def visualize_state_trajectory(self, tr_ex:TrajectoryExport,
                                    controller_info:dict|None=None, simulation_info:dict|None=None,
                                    training_info:dict|None=None, reward_info:dict|None=None,
                                    n_vehicle_markers=None, file_name=None, fig_name:str=None):

        if n_vehicle_markers is None: n_vehicle_markers = self.vehicle_show_count
        if file_name is None and fig_name is None: file_name = "fig"
        if fig_name is None: fig_name = file_name

        fig = plt.figure(layout="constrained", figsize=(15,10), dpi=100)
        fig.suptitle(fig_name, fontsize=24)
        axS = plt.subplot2grid((8,12), (0,0), rowspan=6, colspan=6)
        axU = plt.subplot2grid((8,12), (0,6), rowspan=2, colspan=3)
        axW = plt.subplot2grid((8,12), (2,6), rowspan=2, colspan=3)
        axR = plt.subplot2grid((8,12), (4,6), rowspan=2, colspan=3)
        axE = plt.subplot2grid((8,12), (0,9), rowspan=2, colspan=3)
        axAT = plt.subplot2grid((8,12), (2,9), rowspan=2, colspan=3)
        axTrig = plt.subplot2grid((8,12), (4,9), rowspan=2, colspan=3)
        txt = plt.subplot2grid((8,12), (6,0), rowspan=2, colspan=12)

        axS.set_title("State Space")
        axW.set_title("Angular Drive")
        axU.set_title("Linear Drive")
        axR.set_title("Rewards Yielded")
        axE.set_title("State \'e\'")
        axAT.set_title("States \'\u03B1\', \'\u03B8\'")
        axTrig.set_title("Sine and Cosine of \'\u03B1\',  \'\u03B8\'")

        x_min = np.min(tr_ex.cartesian_states[:,0])
        x_max = np.max(tr_ex.cartesian_states[:,0])
        y_min = np.min(tr_ex.cartesian_states[:,1])
        y_max = np.max(tr_ex.cartesian_states[:,1])
        x_width = x_max - x_min
        y_width = y_max - y_min
        x_mid = x_max - x_width
        y_mid = y_max - y_width
        s_width = max(x_width, y_width)
        s_width += s_width / 10
        s_width /= 2
        x_min, x_max, y_min, y_max = x_mid - s_width, x_mid + s_width, y_mid - s_width, y_mid + s_width
        s_width *= 2
        s_bounds = {"x_min" : x_min, "x_max" : x_max, "y_min" : y_min, "y_max" : y_max}

        #vehicle points:
        step = tr_ex.point_count // (n_vehicle_markers - 1)
        v_idxs = list(range(0, tr_ex.point_count, step))
        if tr_ex.point_count % (n_vehicle_markers - 1) != 0:
            v_idxs[-1] = -1
        else:
            v_idxs.append(-1)
        #tri_height = tr_ex.length / n_vehicle_markers / 8
        tri_height = s_width * 1.337 * 3.0 / n_vehicle_markers / 8

        axS.plot(tr_ex.cartesian_states[:,0], tr_ex.cartesian_states[:,1], '-b')
        vehicle_markers = [self.vehicle_plot_triangle(tr_ex.cartesian_states[idx,0],
                                                    tr_ex.cartesian_states[idx,1],
                                                    tr_ex.cartesian_states[idx,2],
                                                    tri_height,
                                                    tri_height/1.5) for idx in v_idxs]
        for tri in vehicle_markers:
            axS.plot(tri[0], tri[1], color=(1.0,0.0,0.0,0.8), zorder=1, linewidth=1.0)
        axS.plot(0.0, 0.0, color=(0.0,0.0,0.0,0.6), marker='x', markersize=50.0, zorder=0, linewidth=2.2)

        #axS.set_xlabel("x", loc="left")
        #axS.set_ylabel("y", loc="bottom")
        axS.set_xbound(s_bounds["x_min"], s_bounds["x_max"])
        axS.set_ybound(s_bounds["y_min"], s_bounds["y_max"])
        axS.grid(True, which='both', axis='both', color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.3, zorder=-1)
        axS.axhline(y=0.0, xmin=s_bounds["x_min"], xmax=s_bounds["x_max"], color=(0.0,0.0,0.0,0.5), linestyle='-', linewidth=0.35, zorder=0)
        axS.axvline(x=0.0, ymin=s_bounds["x_min"], ymax=s_bounds["x_max"], color=(0.0,0.0,0.0,0.5), linestyle='-', linewidth=0.35, zorder=0)

        u_bounds = {"x_min": np.min(tr_ex.time), "x_max": np.max(tr_ex.time),
                    "y_min": np.min(tr_ex.control_inputs[:,0]), "y_max": np.max(tr_ex.control_inputs[:,0])}
        u_bounds["x_min"] = u_bounds["x_min"] - (u_bounds["x_max"] - u_bounds["x_min"]) / 16
        u_bounds["x_max"] = u_bounds["x_max"] + (u_bounds["x_max"] - u_bounds["x_min"]) / 17
        u_bounds["y_min"] = u_bounds["y_min"] - (u_bounds["y_max"] - u_bounds["y_min"]) / 10
        u_bounds["y_max"] = u_bounds["y_max"] + (u_bounds["y_max"] - u_bounds["y_min"]) / 11

        w_bounds = {"x_min" : u_bounds["x_min"], "x_max" : u_bounds["x_max"],
                    "y_min": np.min(tr_ex.control_inputs[:,1]), "y_max": np.max(tr_ex.control_inputs[:,1])}
        w_bounds["y_min"] = w_bounds["y_min"] - (w_bounds["y_max"] - w_bounds["y_min"]) / 10
        w_bounds["y_max"] = w_bounds["y_max"] + (w_bounds["y_max"] - w_bounds["y_min"]) / 11

        r_bounds = {"x_min" : u_bounds["x_min"], "x_max" : u_bounds["x_max"],
                    "y_min": 1.1 * np.min((np.min(tr_ex.rewards), -1.0)), "y_max": 1.1 * np.max((np.max(tr_ex.rewards), 1.0))}

        veh_vlines_kwargs = {"colors":(1.0,0.0,0.0,0.5), "linewidths":0.7, "zorder":1}

        axU.plot(tr_ex.time[:-1], tr_ex.control_inputs[:,0][:-1], color='tab:orange')
        axU.vlines(x=tr_ex.time[v_idxs], ymin=u_bounds["y_min"], ymax=u_bounds["y_max"], **veh_vlines_kwargs)
        axU.set_xbound(u_bounds["x_min"], u_bounds["x_max"])
        axU.set_ybound(u_bounds["y_min"], u_bounds["y_max"])
        axU.grid(True, which='both', axis='y', color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.25, zorder=0)

        axW.plot(tr_ex.time[:-1], tr_ex.control_inputs[:,1][:-1]/np.pi, color='tab:orange')
        axW.yaxis.set_major_formatter(tck.FormatStrFormatter('%g$\pi$'))
        axW.yaxis.set_major_locator(tck.AutoLocator())
        axW.vlines(x=tr_ex.time[v_idxs], ymin=w_bounds["y_min"], ymax=w_bounds["y_max"], **veh_vlines_kwargs)
        axW.set_xbound(w_bounds["x_min"], w_bounds["x_max"])
        axW.set_ybound(w_bounds["y_min"]/np.pi, w_bounds["y_max"]/np.pi)
        axW.grid(True, which='both', axis='y', color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.25, zorder=0)

        axR.plot(tr_ex.time[:-1], tr_ex.rewards[:-1], "-b")
        axR.vlines(x=tr_ex.time[v_idxs], ymin=r_bounds["y_min"], ymax=r_bounds["y_max"], **veh_vlines_kwargs)
        axR.set_xbound(r_bounds["x_min"], r_bounds["x_max"])
        axR.set_ybound(r_bounds["y_min"], r_bounds["y_max"])

        e_bounds = {"x_min" : u_bounds["x_min"], "x_max" : u_bounds["x_max"],
                    "y_min" : 0.0, "y_max" : 1.1 * np.max(tr_ex.polar_states[:,0])}
        axE.plot(tr_ex.time, tr_ex.polar_states[:,0], "-b")
        axE.vlines(x=tr_ex.time[v_idxs], ymin=e_bounds["y_min"], ymax=e_bounds["y_max"], **veh_vlines_kwargs)
        axE.set_xbound(e_bounds["x_min"], e_bounds["x_max"])
        axE.set_ybound(e_bounds["y_min"], e_bounds["y_max"])

        axAT.plot(tr_ex.time, tr_ex.polar_states[:,1]/np.pi, "-r", label="alfa")
        axAT.plot(tr_ex.time, tr_ex.polar_states[:,2]/np.pi, "-b", label="theta")
        axAT.yaxis.set_major_formatter(tck.FormatStrFormatter('%g$\pi$'))
        axAT.yaxis.set_major_locator(tck.AutoLocator())
        axAT.vlines(x=tr_ex.time[v_idxs], ymin=-1, ymax=1, **veh_vlines_kwargs)
        axAT.axhline(y=0.0, xmin=u_bounds["x_min"], xmax=u_bounds["x_max"], color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.3, zorder=0)
        axAT.set_xbound(u_bounds["x_min"], u_bounds["x_max"])
        axAT.set_ybound(-1, 1)
        axAT.legend()

        axTrig.plot(tr_ex.time, tr_ex.sinecosine_states[:,0], linestyle="-", color="tab:red", label="sin(\u03B1)")
        axTrig.plot(tr_ex.time, tr_ex.sinecosine_states[:,1], linestyle="-", color="tab:orange", label="cos(\u03B1)")
        axTrig.plot(tr_ex.time, tr_ex.sinecosine_states[:,2], linestyle="-", color="tab:blue", label="sin(\u03B8)")
        axTrig.plot(tr_ex.time, tr_ex.sinecosine_states[:,3], linestyle="-", color="tab:purple", label="cos(\u03B8)")
        axTrig.vlines(x=tr_ex.time[v_idxs], ymin=-1, ymax=1, **veh_vlines_kwargs)
        axTrig.axhline(y=0.0, xmin=u_bounds["x_min"], xmax=u_bounds["x_max"], color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.3, zorder=0)
        axTrig.set_xbound(u_bounds["x_min"], u_bounds["x_max"])
        axTrig.set_ybound(-1, 1)
        axTrig.legend()

        txt.axis([0,24,0,4])
        txt.set_axis_off()

        if simulation_info:
            sim_text = "Simulation Info| "
            for key in simulation_info:
                sim_text = sim_text + key + ": " + str(simulation_info[key]) + ", "
            sim_text = sim_text[:-2]
            sim_text = sim_text + "."
            txt.text(0.1, 3.4, sim_text, fontsize=10)

        #if tr_ex.length.shape == ():
        #    l_tr = str(tr_ex.length)
        #else:
        #    l_tr = str(tr_ex.length.flatten()[0])
        l_tr = str(tr_ex.length)[:4]
        t_tr = str(tr_ex.time[-1].flatten())[1:-1]
        for i, char in enumerate(t_tr, start=1):
            if char == '.':
                t_tr = t_tr[:min(i+1, len(t_tr))]
                break
        if tr_ex.reached_goal:
            g_tr = "yes"
        else:
            g_tr = "no"
        trajectory_infotext = "Trajectory Info| Reached Goal: {yesgoal}, Length: {length}, Total Time: {time}.".format(length=l_tr, yesgoal=g_tr, time=t_tr)
        txt.text(0.1, 2.8, trajectory_infotext, fontsize=10)

        if controller_info:
            controller_text = "Controller Info| "
            for key in controller_info:
                controller_text = controller_text + key + ": " + str(controller_info[key]) + ", "
            controller_text = controller_text[:-2]
            controller_text = controller_text + "."
            txt.text(0.1, 2.2, controller_text, fontsize=10)

        if training_info:
            training_text = "Training Info| "
            for key in training_info:
                training_text = training_text + key + ": " + str(training_info[key]) + ", "
            training_text = training_text[:-2]
            training_text = training_text + "."
            txt.text(0.1, 1.6, training_text, fontsize=10)

        if reward_info:
            reward_text = "Reward Info| "
            for key in reward_info:
                reward_text = reward_text + key + ": " + str(reward_info[key]) + ", "
            reward_text = reward_text[:-2]
            reward_text = reward_text + "."
            txt.text(0.1, 1.0, reward_text, fontsize=10)

        fig.savefig(ospjoin(self.save_dir, file_name))
        plt.close(fig)



class LoggerVisualizer:
    def __init__(self, visualizer, steps_per_episode, num_s_vars=3,
                num_sinecosine_s_vars=4, num_c_vars=2,
                **config_plotting_kwargs):
        self.visualizer = visualizer

        self.current_idx = np.ushort(0)
        self.traj_arr = np.empty(shape=(steps_per_episode, 2 * num_s_vars + num_sinecosine_s_vars + num_c_vars + 2), dtype=np.single)
        self.reached_goal = False
        self.n_s = num_s_vars
        self.n_trig_s = num_sinecosine_s_vars
        self.n_c = num_c_vars
        self.length = np.single(0)

        self.traj_ended = False

        self.config_plotting(**config_plotting_kwargs)
        self.file_name_number = 0

    def config_plotting(self, controller_info:dict|None=None, simulation_info:dict|None=None,
                        training_info:dict|None=None, reward_info:dict|None=None,
                        n_vehicle_markers=None, file_name=None, fig_name:str=None):
        d = {"controller_info" : controller_info, "simulation_info" : simulation_info,
            "training_info" : training_info, "reward_info" : reward_info,
            "n_vehicle_markers" : n_vehicle_markers, "file_name" : file_name, "fig_name" : fig_name}
        if not hasattr(self, "plot_call_kwargs"):
            self.plot_call_kwargs = d
        else:
            for key, value in d.items():
                if value is not None:
                    self.plot_call_kwargs[key] = value

    def add_point(self, cartesian_state, polar_state, sine_cosine_states, control_inputs, reward, time):
        self.traj_arr[self.current_idx, :] = np.concatenate([cartesian_state, polar_state, sine_cosine_states, control_inputs, np.array([reward, time])], axis=0)
        self.current_idx += 1
        if self.current_idx > 1:
            s = self.traj_arr[self.current_idx - 2, :2]
            self.length += np.sqrt(np.square(cartesian_state[0] - s[0]) + np.square(cartesian_state[1] - s[1]))

    def end_current_trajectory(self, reached_goal):
        self.reached_goal = reached_goal
        self.traj_ended = True

    def reset_trajectory(self):
        self.current_idx = np.ushort(0)
        self.traj_arr = np.empty_like(self.traj_arr)
        self.reached_goal = False
        self.length = np.single(0)
        self.traj_ended = False

    def create_and_save_plot(self):
        if not self.traj_ended:
            print("Warning! Trajectory not ended but call to create and save plot has been made.")
        tr_ex = TrajectoryExport(self.trajectory_in_TrEx_format())
        plot_kwargs = self.plot_call_kwargs.copy()
        fname = "_plot_" + str(self.file_name_number)
        self.file_name_number += 1
        plot_kwargs["file_name"] = plot_kwargs["file_name"] + fname
        plot_kwargs["fig_name"] = plot_kwargs["file_name"]
        if tr_ex.point_count < self.visualizer.vehicle_show_count:
            plot_kwargs["n_vehicle_markers"] = tr_ex.point_count
        self.visualizer.visualize_state_trajectory(tr_ex, **plot_kwargs)

    def trajectory_in_TrEx_format(self):
        i = self.current_idx
        ns = self.n_s
        ntrigs = self.n_trig_s
        nc = self.n_c
        return {"cartesian_states" : self.traj_arr[:i, :ns],
                "polar_states" : self.traj_arr[:i, ns:2 * ns],
                "sinecosine_states" : self.traj_arr[:i, 2 * ns : 2 * ns + ntrigs],
                "control_inputs" : self.traj_arr[:i, 2 * ns + ntrigs : 2 * ns + ntrigs + nc],
                "rewards" : self.traj_arr[:i, -2],
                "time" : self.traj_arr[:i, -1],
                "length" : self.length,
                "point_count" : i,
                "reached_goal" : self.reached_goal}