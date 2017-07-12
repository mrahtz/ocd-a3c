#!/usr/bin/env python3

import gym
env = gym.make('Enduro-v0')
try:
    for i_episode in range(20):
        observation = env.reset()
        while True:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(reward)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
except KeyboardInterrupt:
    print("Exiting...")
