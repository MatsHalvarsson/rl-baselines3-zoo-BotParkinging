import numpy as np





class StartPosGeneratorBase:
    def __init__(self, fix_pos=[0.5, 0.0, 0.0]):
        self.pos_str = str(fix_pos)
        self.pos = np.array(fix_pos, dtype=np.single)  #polar coordinates

    def get_startpos(self, cartesian=True):
        pos = self.pos.copy()
        if cartesian:
            return np.array([- pos[0] * np.cos(pos[2]),
                             - pos[0] * np.sin(pos[2]),
                             np.arctan2(np.sin(pos[2] - pos[1]),
                                        np.cos(pos[2] - pos[1]))])
        return pos

    def __call__(self, cartesian=True):
        return self.get_startpos(cartesian)

    def __str__(self):
        return "Fix startpos: " + self.pos_str



class GoalChecker:
    def __init__(self, polar_state_representation=True, err_dist=0.05, err_angl=np.pi/18):
        self.polar_s = polar_state_representation
        self.err_e = np.single(err_dist)
        self.err_phi = np.single(err_angl)

    def __call__(self, position):
        if self.polar_s:
            return position[0] < self.err_e and np.abs(position[2] - position[1]) < self.err_phi
        else:
            return np.sqrt(np.sum(np.square(position[:2]))) < self.err_e and np.abs(position[2]) < self.err_phi

    def __str__(self):
        e_str = str(self.err_e)
        if len(e_str) > 4:
            e_str = e_str[:4]
            while e_str[-1] == "0":
                e_str = e_str[:-1]
        phi_str = str(self.err_phi)
        if len(phi_str) > 6:
            phi_str = phi_str[:6]
        return f"Proximity Goal: e<{e_str} && phi<{phi_str}"



class OOBChecker:
    def __init__(self, polar_state_representation=True, oob_dist=1.00):
        self.polar_s = polar_state_representation
        self.oob_e = np.single(oob_dist)

    def __call__(self, position):
        if self.polar_s:
            return position[0] >= self.oob_e
        else:
            return np.sqrt(np.sum(np.square(position[:2]))) >= self.oob_e

    def __str__(self):
        dist_str = str(self.oob_e)
        if len(dist_str) > 3: dist_str = dist_str[:3]
        return f"Distance OOB: e>={dist_str}"



class RewarderBase:
    def __init__(self):
        pass

    def __call__(self, old_state=None, action=None, new_state=None, is_goal=False, is_oob=False):
        return np.single(0.0)

    def terminal_r(self, is_goal, is_oob):
        if is_goal: return True, np.single(1.0)
        if is_oob: return True, np.single(-1.0)
        return False, None

    def __str__(self):
        return ""

class RewardLyapEq11(RewarderBase):
    def __init__(self, lambdagamma=1.0, k=1.0):
        self.str_lambdagamma = str(lambdagamma)[0:min(len(str(lambdagamma)), 3)]
        if self.str_lambdagamma[-1] == ".": self.str_lambdagamma = self.str_lambdagamma[0:-1]
        self.str_k = str(k)[0:min(len(str(k)), 3)]
        if self.str_k[-1] == ".": self.str_k = self.str_k[0:-1]
        self.consts = np.array([lambdagamma, k], dtype=np.single)
        self.sum_of_consts = np.sum(self.consts)

    def __call__(self, new_state=None, is_goal=False, is_oob=False, **kwargs):
        #division with sum of self.consts normalizes reward to [-1,0)
        #if not goal, returns
        #    - w_1 * cos(alfa)^2 * e^2 - w_2 * alfa^2
        # with [w_1, w_2] = self.consts

        is_terminal, r = self.terminal_r(is_goal, is_oob)
        if is_terminal: return r

        r = np.square(np.array([np.cos(new_state[1]) * new_state[0], new_state[1]]))
        r = - np.divide(np.sum(np.multiply(self.consts, r)), self.sum_of_consts)
        return r

    def __str__(self):
        return f"paper Eq11 l*g: {self.str_lambdagamma}, $k$: {self.str_k}"