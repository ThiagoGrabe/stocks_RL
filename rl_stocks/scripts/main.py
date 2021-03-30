# -*- coding: utf-8 -*-
# Standard Libraries
import os
import argparse
import datetime

# Third parties libraries
import gym
import numpy as np
import pandas as pd
from yahoo import Yahoo
import matplotlib.pyplot as plt
from stable_baselines3 import DQN as DDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from ta import add_all_ta_features


# Personal Code
from rl_stocks.envs.StocksRL import StocksRL
from callbacks import SaveOnBestTrainingRewardCallback


#######################################################
################# PARSE ARGUMENTS #####################
#######################################################
def getArgs():
    "Set all arguments."
    parser = argparse.ArgumentParser(prog='PROG', description="Reinforcement Learning Stock Prices - Peter Gunnarsson Project")

    parser.add_argument("-s", '--stock', required=False, type=str, help="Stock to be analysed.\n")

    parser.add_argument("-i", '--interval', required=False, type=str, help="Interval to be analysed.\n")

    parser.add_argument("-p", '--period', required=False, type=int, help="Period (in days) to be analysed.\n")

    parser.add_argument("-start", '--start_date',required=False, type=str, help="Start date to run the RL algorithm\n")

    parser.add_argument("-end", '--end_date',required=False, type=str, help="End date to run the RL algorithm\n")

    parser.add_argument("-a", '--trade_amount', required=True, type=float, help="Amount of money to be traded.\n")

    parser.add_argument("-w", '--wallet', required=True, type=float, help="Amount of money in the wallet.\n")

    parser.add_argument("-it", '--interest_rate', required=True, type=float, help="Interest Rate value.\n")

    parser.add_argument("-k", '--key', required=True, type=str, help="Dataframe key to use.\n")

    parser.add_argument("-df_train", '--train_dataset', required=False, type=str, help="Dataset to be used.\n")

    parser.add_argument("-df_test", '--test_dataset', required=False, type=str, help="Dataset to be used.\n")

    parser.add_argument("-model", '--model_name', required=True, type=str, help="Model name to be loaded/saved.\n")

    # parser.add_argument("-a", '--algorithm',required=False, type=str, help="Algorithm to be used.\n")

    args = vars(parser.parse_args())
    return args

#######################################################
################# PARSE ARGUMENTS #####################
#######################################################

if __name__ == '__main__':

    # Create log dir
    file_dir        = os.path.dirname(os.path.abspath(__file__))
    log_dir         = file_dir+"/results"
    train_log_dir   = file_dir+"/results/train_log"
    test_log_dir   = file_dir+"/results/test_log"
    tensorboard_log = file_dir+"/tensorboard"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(test_log_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    # Get arguments
    args = getArgs()

    # Set all arguments
    # STOCK          = args['stock'] # stock ticket
    KEY            = args['key'] # stock ticket
    MODEL_NAME     = args['model_name']

    # If both datasets (train/test) is passed in the arguments, the algorithm will learn and test. Otherwise it is indicated by the dataset argument.

    TRAIN_DATASET  = args['train_dataset']
    if TRAIN_DATASET is not None:
        TRAIN = True
    else:
        TRAIN = False

    TEST_DATASET   = args['test_dataset']
    if TEST_DATASET is not None:
        TEST = True
    else:
        TEST = False

    WINDOW         = int(args['period']) # number (int) of days from start date
    TRADE_AMOUNT   = int(args['trade_amount'])
    WALLET         = int(args['wallet'])
    INTEREST_RATE  = float(args['interest_rate'])

    if TRAIN:
        # Training Dataset
        train_df = pd.read_csv(file_dir+'/.data/'+str(TRAIN_DATASET))
        train_df.fillna(method='bfill', inplace=True)

        OBSERVATION_SPACE  =  len(train_df.columns.tolist()[7:])

        # The total timesteps should not be larger than the current dataset
        training_timesteps = len(train_df)-WINDOW-1

    if TEST:
        # Testing Dataset
        test_df = pd.read_csv(file_dir+'/.data/'+str(TEST_DATASET))
        test_df.fillna(method='bfill', inplace=True)

        OBSERVATION_SPACE_TEST = len(test_df.columns.tolist()[7:])

        # The total timesteps should not be larger than the current dataset
        testing_timesteps = len(test_df)-WINDOW-1

    # Stable Baselines obs and action information
    N_DISCRETE_ACTIONS = [0, 1, 2] # Short, Flat, Long
    # OBSERVATION_SPACE  = len(myStockPrices.select_dtypes(include=['float']).iloc[0].values) # Time interval to keep NN entry constant
    
    
    if TEST and TRAIN:
        assert OBSERVATION_SPACE == OBSERVATION_SPACE_TEST, "Datasets must have same number of columns/features."

    env = gym.make('rl_stocks-v0')

    # Create Callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=5, log_dir=log_dir, model_name=MODEL_NAME,verbose=0)

    # Learning...
    if TRAIN:
         # Reseting env before train/test
        env._reset(actions=N_DISCRETE_ACTIONS, observation_space=OBSERVATION_SPACE, data=train_df, 
                trade_amount=TRADE_AMOUNT, key=KEY, wallet=WALLET, window=WINDOW, interest_rate=INTEREST_RATE,
                log_dir=train_log_dir)
        env = Monitor(env=env, filename=log_dir)
        
        model = DDQN("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log)
        model.learn(total_timesteps=training_timesteps, log_interval=1, callback=[callback])

        npz = np.load(train_log_dir+'/log.npz')
        df= pd.DataFrame.from_dict({item: npz[item] for item in npz.files})
        print('Train Profit Factor:', df['Index'].loc[(df['State']!='Flat') & (df['Profit']>0)].count() / df['Index'].loc[(df['State']!='Flat') & (df['Profit']<0)].count())
        df.to_csv(train_log_dir+'/Train_log.csv')
    
    if TEST:
        model = DDQN.load(log_dir+'/best_model/'+MODEL_NAME)
        env = gym.make('rl_stocks-v0')
        env._reset(actions=N_DISCRETE_ACTIONS, observation_space=OBSERVATION_SPACE_TEST, data=test_df, 
               trade_amount=TRADE_AMOUNT, key=KEY, wallet=WALLET, window=WINDOW, interest_rate=INTEREST_RATE,
               log_dir=test_log_dir)

        for step in range(testing_timesteps):
            obs = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # print('action=', action, 'reward=', reward, 'done=', done)

        npz = np.load(test_log_dir+'/log.npz')
        df= pd.DataFrame.from_dict({item: npz[item] for item in npz.files})
        df['State_prime'] = df['State'].shift()
        print('Test Profit Factor:', df['Index'].loc[(df['State_prime']!='Flat') & (df['Profit']>0)].count() / df['Index'].loc[(df['State_prime']!='Flat') & (df['Profit']<0)].count())
        print('Test Profit', df['Profit'].sum())
        df.to_csv(test_log_dir+'/Test_log.csv')

    

    print('Done')

        
