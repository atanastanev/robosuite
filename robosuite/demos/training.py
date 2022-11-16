
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

"""
controller_name = choose_controller()
options["controller_configs"] = load_controller_config(default_controller=controller_name)
# create environment instance
env = suite.make(
    **options
    env_name="PickPlace", # try with other tasks like "Stack" and "Door"
    robots="wx250s",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display 
    
    """

"""
if __name__ == "__main__":    
    options = {}
    options["env_name"] = "PickPlace"
    options["robots"] = "wx250s"
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec
    print(low, high)

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low,high)
        action = [0, 0, 0, 0, 0, 0, 0.1]
        obs, reward, done, _ = env.step(action)
        #print("reward", reward)
        #print("obs", obs)
        #print("done", done)
        #print(_)        
        env.render()
    """

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
"""
env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

if __name__ == "__main__":    
    options = {}
    options["env_name"] = "NutAssembly"
    options["robots"] = "wx250s"
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=False,
        use_camera_obs=True,
        control_freq=20,
    )
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)
    
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
      """ 
       
"""

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="wx250s",  # use Sawyer robot
            use_camera_obs=True,  # do not use pixel observations
            has_offscreen_renderer=True,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth 
            render_camera='frontview', 
            render_collision_mesh=False, 
            render_visual_mesh=True, 
            render_gpu_device_id=- 1, 
            horizon=1000, 
            ignore_done=False, 
            hard_reset=True, 
            renderer='mujoco', 
            renderer_config=None,
        )
    )

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            env.render()
            print(reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
             
        """        
                
"""                
if __name__ == "__main__":    
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="wx250s",  # use Sawyer robot
            use_camera_obs=True,  # do not use pixel observations
            has_offscreen_renderer=True,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )
    
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)
    
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
  """                    
                
                
                
                
"""pygame rendering demo.

This script provides an example of using the pygame library for rendering
camera observations as an alternative to the default mujoco_py renderer.
This is useful for running robosuite on operating systems where mujoco_py is incompatible.

Example:
    $ python demo_pygame_renderer.py --environment Stack --width 1000 --height 1000

"""
"""
import sys

import numpy as np
import pygame

import robosuite

if __name__ == "__main__":    
    screen = pygame.display.set_mode((1000, 800))
    env = robosuite.make(
        "Lift",
        robots="Panda",  
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        camera_names="frontview",
        camera_heights=800,
        camera_widths=1000,
        use_camera_obs=False,
        use_object_obs=False,
    ) 

    obs = env.reset()
    #model = PPO2(MlpPolicy, env, verbose=1)
    #model.learn(10000)
    
    #obs = env.reset()
    #env.viewer.set_camera(camera_id=0)


    for i in range(1000):
        action = np.random.randn(env.robots[0].dof)
        obs, reward, done, info = env.step(action)

        #action, _states = model.predict(obs)
        #obs, rewards, done, info = env.step(action)
        # issue random actions
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # read camera observation
        im = np.flip(obs["frontview" + "_image"].transpose((1, 0, 2)), 1)
        pygame.pixelcopy.array_to_surface(screen, im)
        pygame.display.update()

        if i % 100 == 0:
            print("step #{}".format(i))

        if done:
            break








"""

import gym
import sys

import numpy as np
import pygame
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
if __name__ == "__main__":
    env = GymWrapper(
            suite.make(
                "Lift",
                robots="wx250s",  # use Sawyer robot
                use_camera_obs=False,  # do not use pixel observations
                use_object_obs=True,
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  # make sure we can render to the screen
                reward_shaping=False,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
            )
        )

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    #model.save("liftTest")

    #del model # remove to demonstrate saving and loading
    #model = PPO2.load("liftTest")

    #env.reset()
    env.viewer.set_camera(camera_id=0)

    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

