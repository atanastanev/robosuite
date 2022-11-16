import robosuite as suite
import gym
import numpy as np

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines import PPO2
#from stable_baselines.common.save_util import save_to_zip_file, load_from_zip_file
#from stable_baselines.common.monitor import Monitor
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

"""
# Training
env = GymWrapper(
        suite.make(
            'Lift',
            robots='wx250s',
            gripper_types='wx250sGripper',
            has_renderer = False,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq = 50,
            render_camera = None,
            horizon = 2000,
            reward_shaping = True,
        )
    )
def wrap_env(env):
    wrapped_env = Monitor(env, "testwx250slong4m")                          # Needed for extracting eprewmean and eplenmean
    wrapped_env = DummyVecEnv([lambda : wrapped_env])   # Needed for all environments (e.g. used for mulit-processing)
    wrapped_env = VecNormalize(wrapped_env)             # Needed for improving training when using MuJoCo envs?
    return wrapped_env
    
env = wrap_env(env) 
filename = 'testwx250slong4m'

model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log='./ppo_fetchpush_tensorboard/')
model.learn(total_timesteps=10000, tb_log_name=filename)

model.save('trained_models/' + filename)
env.save('trained_models/vec_normalize_' + filename + '.pkl')     # Save VecNormalize statistics

"""
# Testing
'''
Create identical environment with renderer or override render function in environment to something like this

def render(self, mode=None):
    super().render()
'''
filename = 'testwx250slong4m'

env_robo = GymWrapper(  
        suite.make(
            'Lift',
            robots='wx250s',
            gripper_types='wx250sGripper',
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq = 50,
            render_camera = None,
            horizon = 2000,
            reward_shaping = True
        )
    )

# Load model
model = PPO2.load('trained_models/' + filename)
# Load the saved statistics
env = DummyVecEnv([lambda : env_robo])
env = VecNormalize.load('trained_models/vec_normalize_' + filename + '.pkl', env)
#  do not update them at test time
env.training = False
# reward normalization
env.norm_reward = False

obs = env.reset()

for i in range(3000):
    
    action, _states = model.predict(obs) #, deterministic=True
    print("action",action)
    obs, reward, done, info = env.step(action) #service->step(action)
    #print("obs")
    #print(obs)
    #sent action
    env_robo.render()
    if done:
        obs = env.reset()

env.close()
