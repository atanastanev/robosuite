import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import gym
import numpy as np
import pygame
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from robosuite.wrappers import GymWrapper

if __name__ == "__main__":
    env = GymWrapper(
            suite.make(
                "Lift",
                robots="wx250s", 
                use_camera_obs=False,
                use_object_obs=False,
                has_offscreen_renderer=False, 
                has_renderer=True, 
                reward_shaping=True,  
                control_freq=20, 
            )
        )

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("liftTest")

    #del model # remove to demonstrate saving and loading
    model = PPO2.load("liftTest")

    #env.reset()
    #env.viewer.set_camera(camera_id=0)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break

