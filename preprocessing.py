import cv2

import numpy as np
from gym import ObservationWrapper, spaces, Wrapper


def prepro2(I):
    """ prepro 210x160 frame into 80x80 frame """
    I = I[34:194]  # crop
    I = I[::2, ::2]  # downsample by factor of 2
    I[I <= 0.4] = 0  # erase background
    I[I > 0.4] = 1  # everything else (paddles, ball) just set to 1
    I = np.pad(I, pad_width=2, mode='constant')
    return I.astype(np.float)


def prepro(o):
    o = np.mean(o, axis=2)
    o = o / 255.0
    return o


class EnvWrapper():
    def __init__(self, env, pool=False, frameskip=1, prepro2=None):
        self.env = env
        self.pool = pool
        self.prepro2 = prepro2
        # 1 = don't skip
        # 2 = skip every other frame
        self.frameskip = frameskip
        self.action_space = env.action_space
        # gym.utils.play() wants these
        self.observation_space = spaces.Box(low=0., high=1.,
                                            shape=(84, 84),
                                            dtype=np.float32)
        self.unwrapped = env.unwrapped
        self.reward_range = env.unwrapped.reward_range
        self.metadata = env.unwrapped.metadata

    def reset(self):
        o = self.env.reset()
        self.prev_o = o
        o = prepro(o)
        if self.prepro2 is not None:
            o = self.prepro2(o)
        return o

    def step(self, a):
        i = 0
        done = False
        rs = []
        while i < self.frameskip and not done:
            o_raw, r, done, _ = self.env.step(a)
            rs.append(r)
            if not self.pool:
                o = o_raw
            else:
                # Note that the first frame to come out of this
                # _might_ be a little funny because the first prev_o
                # is the first frame after reset which at least in Pong
                # has a different colour palette (though it turns out
                # it works fine for Pong)
                o = np.maximum(o_raw, self.prev_o)
                self.prev_o = o_raw
            i += 1
        o = prepro(o)
        if self.prepro2 is not None:
            o = self.prepro2(o)
        r = sum(rs)
        info = None
        return o, r, done, info

    def render(self):
        self.env.render()


class ConcatFrameStack(ObservationWrapper):
    """
    Concatenate a stack horizontally into one long frame (for debugging).
    """

    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        # Important so that gym's play.py picks up the right resolution
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(84, 4 * 84),
                                            dtype=np.uint8)

    def observation(self, obs):
        assert obs.shape[0] == 4
        return np.hstack(obs)


class NumberFrames(ObservationWrapper):
    """
    Draw number of frames since reset (for debugging).
    """

    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.frames_since_reset = None

    def reset(self):
        self.frames_since_reset = 0
        return self.env.reset()

    def observation(self, obs):
        cv2.putText(obs,
                    str(self.frames_since_reset),
                    org=(0, 70),  # x, y position of bottom-left corner of text
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2.0,
                    color=(255, 255, 255),
                    thickness=2)
        self.frames_since_reset += 1
        return obs


class EarlyReset(Wrapper):
    """
    Reset the environment after 100 steps (for debugging).
    """

    def reset(self):
        self.n_steps = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.n_steps += 1
        if self.n_steps >= 100:
            done = True
        return obs, reward, done, info
