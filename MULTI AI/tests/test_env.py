



import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from envs.commons_env import CommonsEnv, CommonsEnvConfig

cfg = CommonsEnvConfig(n_agents=3, regen=5, action_max=3)
env = CommonsEnv(cfg)
print(env.reset())
print(env.step([1,1,1]))
print(env.step([2,0,1]))
print("Steps logged:", len(env.history))
