# -*- coding: utf-8 -*-
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import argparse
import numpy as np
import pandas as pd
import datetime

from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import DQN as DDQN

from rl_stocks.envs.StocksRL import StocksRL
from yahoo import Yahoo

#######################################################
################# PARSE ARGUMENTS #####################
#######################################################
def getArgs():
    "Set all arguments."
    parser = argparse.ArgumentParser(prog='PROG', description="Reinforcement Learning Stock Prices - Peter Gunnarsson Project")

    parser.add_argument("-s", '--stock', required=False, type=str, help="Stock to be analysed.\n")

    parser.add_argument("-l", '--length', required=False, type=str, help="State length to be analysed.\n")

    parser.add_argument("-i", '--interval', required=False, type=str, help="Interval to be analysed.\n")

    parser.add_argument("-p", '--period', required=False, type=int, help="Period (in days) to be analysed.\n")

    parser.add_argument("-d", '--day',required=False, type=str, help="Start day to run the RL algorithm\n")

    parser.add_argument("-m", '--month',required=False, type=str, help="Start month to run the RL algorithm\n")

    parser.add_argument("-y", '--year',required=False, type=str, help="Start year to run the RL algorithm\n")

    # parser.add_argument("-a", '--algorithm',required=False, type=str, help="Algorithm to be used.\n")

    args = vars(parser.parse_args())
    return args

#######################################################
################# PARSE ARGUMENTS #####################
#######################################################

if __name__ == '__main__':

    # Create log dir
    file_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir  = file_dir+"/.results"
    os.makedirs(log_dir, exist_ok=True)

    # Get arguments
    args = getArgs()

    # Set all arguments
    STOCK       = args['stock'] # stock ticket
    START_DAY   = args['day'] # day (dd)
    START_MONTH = args['month'] # month (mm)
    START_YEAR  = args['year'] # year (yyyy)
    WINDOW      = int(args['period']) # number (int) of days from start date
    INTERVAL    = args['interval'] # data interval for stock values
    STATE_LEN   = int(args['length']) # state len. It defines the NN input

    # Create a date time string (dd-mm-yyyy)
    date_time_str = START_DAY+'-'+START_MONTH+'-'+START_YEAR

    # Setting the start and end date
    START_DATE = datetime.datetime.strptime(date_time_str, '%d-%m-%Y')
    END_DATE   = START_DATE + datetime.timedelta(days=WINDOW) # Should be date just like start date

    # Create the Yahoo class and get the stock prices ('Close')
    myStock = Yahoo(window=WINDOW, stock=STOCK, start_date=START_DATE, end_date=END_DATE, interval=INTERVAL)
    myStockPrices = myStock.getStockInfo(key='Close')

    # Stable Baselines obs and action information
    N_DISCRETE_ACTIONS = [0,1,2] # Buy, Sell or Hold
    OBSERVATION_SPACE  = STATE_LEN # Time interval to keep NN entry constant

    # Instantiate the gym env
    env = gym.make('rl_stocks-v0')
    env = StocksRL()
    env._reset(actions=N_DISCRETE_ACTIONS, observation_space=OBSERVATION_SPACE, data=myStockPrices)
    
    # It will check your custom environment and output additional warnings if needed
    check_env(env)


        
