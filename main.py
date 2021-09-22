"""Launch file for the discrete DDQN algorithm

This script instantiate the gym environment, the agent, and start the training
"""

import argparse
import os

import gym
import yaml
from make_env import make_env

from agent import DDQN
from utils.tracker import Tracker

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

if not cfg['setup']['use_gpu']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, help='Gym env', default=cfg['train']['name'])
parser.add_argument('-epochs', type=int, help='Epochs', default=cfg['train']['n_episodes'])
parser.add_argument('-verbose', type=int, help='Save stats freq', default=cfg['train']['verbose'])
parser.add_argument('-eps_d', type=float, help='Exploration decay', default=cfg['agent']['eps_d'])
parser.add_argument('-tau', type=float, help='Target net τ', default=cfg['agent']['tau'])
parser.add_argument('-use_polyak', type=float, help='Polyak update', default=cfg['agent']['polyak'])
parser.add_argument('-tg_update', type=float, help='Standard update', default=cfg['agent']['tg_update'])

def main(params):
    config = vars(parser.parse_args())

    env = make_env(config['env'])
    env.seed(seed)
    
    #env.n = restituisce il numero di agenti
    """
    agent = []
    for i in range(env.n):
        agent.append(DDQN(env, cfg['agent']))
        tag = 'DDQN'
    """
    print(f"{env.action_space[0]} \n {env.action_space}")

    #printf(f"{env.")
    agent = DDQN(env, cfg['agent'])
    tag = 'DDQN'

    # Initiate the tracker for stats
    if env.n==1:
        tracker = Tracker(
            'simple',
            tag,
            seed,
            cfg['agent'], 
            ['Epoch', 'Ep_Reward']
        )
    else:   
        tracker = Tracker(
            'simple',
            tag,
            seed,
            cfg['agent'], 
            ['Epoch', 'Ep_Reward', 'Ep_Adv_Reward']
        )

    # Train the agent
    agent.train(
        tracker,
        n_episodes=config['epochs'], 
        verbose=config['verbose'],
        params=cfg['agent'],
        hyperp=config
    )

if __name__ == "__main__":
    main(cfg)
