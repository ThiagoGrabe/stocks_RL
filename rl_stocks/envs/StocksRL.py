# Standard Libraries
import random
import math
import pickle
import csv

# Third parties libraries
import gym
from gym import spaces
from gym.utils import seeding
from gym import spaces
import numpy as np
import yfinance as yf



class StocksRL(gym.Env):
    """
    Main class for stock trading gym environment. The main functions are: reset and step. Both should not be removed.

    param N_DISCRETE_ACTIONS: list of actions [0,1,2] > [buy, sell, hold]
    param OBS: observation space is the window interval to train. It indicates the input shape of the first layer of the NN.
    param action_space: Is also a gym space object that describes the action space, so the type of action that can be taken
    param observation_space: Is one of the gym spaces (Discrete, Box, ...) and describe the type and shape of the observation
    param data: The stock data
    param index: Simple counter for a dummy env.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(StocksRL, self).__init__()

        
    def _reset(self, 
                actions, 
                observation_space, 
                data, 
                key, 
                log_dir, 
                trade_amount=1e3, 
                wallet=1e5, 
                window=7, 
                interest_rate=-0.001):


        self.N_DISCRETE_ACTIONS = actions
        self.OBS                = observation_space

        self.trade_amount       = trade_amount
        self.wallet             = wallet # Change during simulation
        self.full_wallet        = wallet # Base value
        self.transaction_cost   = 0.01 # Hard coded as requested in the project requirements
        self.interest_rate      = interest_rate
        
        self.data               = data
        self.number_stocks      = 0
        self.profit             = 0
        self.logging            = dict()
        self.transaction_history= dict()

        self.index              = 0
        self.episode            = 1
        self.window             = window
        self.key                = key
        self.game_status        = True
        self.state              = 'Flat'
        self.terminate          = False
        self.log_dir            = log_dir

        self.action_space       = spaces.Discrete(len(self.N_DISCRETE_ACTIONS))
        self.observation_space  = spaces.Box(low=0, high=float('inf'),
                                        shape=(self.window, self.OBS), dtype=np.float32)

        # Logging Lists
        self.log_state       = []
        self.log_idx         = []
        self.log_qty_stocks  = []
        self.log_trade_cost  = []
        self.log_stock_price = []
        self.log_reward      = []
        self.log_action      = []
        self.log_profit      = []

    def reset(self):
        # print('Starting new episode...\n')
        return self.perception()
    
    def step(self, action):
        self.current_action = self._action(action)
        self.reward         = self._reward(action)
        self.info           = {}
        self.index          += 1
        self.perception()
        # print('Action: ', self.current_action, 'Reward: ', self.reward)

        if self.terminate:
            path_ = self.log_dir+"/log.npz"
            np.savez(path_, Index=self.log_idx, State=self.log_state, QtyStocks=self.log_qty_stocks, TradeCost=self.log_trade_cost, StockPrice=self.log_stock_price, Profit=self.log_profit)
            self.episode += 1
            return self.obs, self.reward, self.terminate, self.info
        else:
            return self.obs, self.reward, self.terminate, self.info
    
    def render(self, mode='human'):
        print('render function')

    def perception(self):
        # The current perception/observation is the window set by the argument with the same name. If window = 4, then there will be a numpy array of 4 other lists
        '''
        array([[1.2875000e+02, 1.2937000e+02, 1.2852000e+02, 1.2925000e+02,
        2.2722524e+07],
       [1.2954000e+02, 1.2970000e+02, 1.2854000e+02, 1.2951000e+02,
        2.4534388e+07],
       [1.2866000e+02, 1.2903000e+02, 1.2772000e+02, 1.2901000e+02,
        3.8143520e+07],
       [1.3010000e+02, 1.3032000e+02, 1.2906000e+02, 1.2931000e+02,
        2.7054142e+07]])
        '''
        self.obs = self.data.iloc[self.index:self.index+self.window, 7:].values

        if self.obs.shape != (self.window, self.OBS):

            self.obs = np.concatenate((self.obs, np.zeros((abs(self.window-self.obs.shape[0]), self.OBS))))
        assert self.obs.shape == (self.window, self.OBS)
        # print(self.obs.shape)
        # print(self.obs)
        return self.obs

    def _trade(self):

        if (self.state == 'Go Short' or self.state == 'Go Long') and self.number_stocks == 0:
            # First Trade
            self.current_stock_price = self.data[self.key].iloc[min(self.index+self.window-1, len(self.data))]
            self.history_stock_price = self.current_stock_price
            self.number_stocks       = math.floor(self.trade_amount/self.current_stock_price) # Can't buy half stock
            self.actual_trade_cost   = self.number_stocks*self.current_stock_price*(1+self.transaction_cost)
            
            if self.wallet > self.actual_trade_cost:
                self.wallet              -= self.actual_trade_cost
                self.transaction_history = {'State': self.state,
                                            'When':min(self.index+self.window-1, len(self.data)),
                                            'Qty Stocks': self.number_stocks,
                                            'Trade Cost': self.actual_trade_cost,
                                            'Stock Price': self.current_stock_price}
                self.log_state.append(self.state)
                self.log_idx.append(min(self.index+self.window-1, len(self.data)))
                self.log_qty_stocks.append(self.number_stocks)
                self.log_trade_cost.append(self.actual_trade_cost)
                self.log_stock_price.append(self.current_stock_price)
                self.log_profit.append(0)
                self.terminate = False
            else:
                # print('Trade cost higher than wallet at index:', self.index+self.window)
                self.terminate = True
                self.wallet    = self.full_wallet
                return 0

        elif (self.state == 'Go Short' or self.state == 'Go Long') and self.number_stocks != 0:
            # Coming from Go Short/Go Long. We sell the stocks, take profit and buy again in go short/go long state.
            self.current_stock_price = self.data[self.key].iloc[min(self.index+self.window-1, len(self.data))]
            self.history_stock_price = float(self.transaction_history['Stock Price'])
            self.number_stocks_sell  = int(self.transaction_history['Qty Stocks'])
            self.profit              = self.number_stocks_sell * (self.current_stock_price - self.history_stock_price) * (1+self.transaction_cost)
            self.number_stocks       -= self.number_stocks_sell
            self.wallet              += self.profit

            # After selling, let's start short/long!
            self.number_stocks       += math.floor(self.trade_amount/self.current_stock_price) # Can't buy half stock
            self.actual_trade_cost   = self.number_stocks*self.current_stock_price*(1+self.transaction_cost)
            
            if self.wallet > self.actual_trade_cost:
                self.wallet              -= self.actual_trade_cost
                self.transaction_history = {'State': self.state,
                                            'When':min(self.index+self.window-1, len(self.data)),
                                            'Qty Stocks': self.number_stocks,
                                            'Trade Cost': self.actual_trade_cost,
                                            'Stock Price': self.current_stock_price}
                self.log_state.append(self.state)
                self.log_idx.append(min(self.index+self.window-1, len(self.data)))
                self.log_qty_stocks.append(self.number_stocks)
                self.log_trade_cost.append(self.actual_trade_cost)
                self.log_stock_price.append(self.current_stock_price)
                self.log_profit.append(self.profit)
                self.terminate = False
            else:
                self.terminate = True
                self.wallet    = self.full_wallet
                return 0

        elif self.state == 'Flat' and self.number_stocks > 0:
            # Sell and take profit
            self.current_stock_price = self.data[self.key].iloc[min(self.index+self.window-1, len(self.data))]
            self.history_stock_price = float(self.transaction_history['Stock Price'])
            self.number_stocks_sell  = int(self.transaction_history['Qty Stocks'])
            self.profit              = self.number_stocks_sell * (self.current_stock_price - self.history_stock_price) *(1+self.transaction_cost)
            self.number_stocks       -= self.number_stocks_sell

            self.wallet              += self.profit

            self.log_state.append(self.state)
            self.log_idx.append(min(self.index+self.window-1, len(self.data)))
            self.log_qty_stocks.append(self.number_stocks_sell)
            self.log_trade_cost.append(0)
            self.log_stock_price.append(self.current_stock_price)
            self.log_profit.append(self.profit)
            self.terminate = True
            self.wallet   = self.full_wallet
        else:
            # print('State/Condition not found!', self.state)
            self.log_state.append(self.state)
            self.log_idx.append(min(self.index+self.window-1, len(self.data)))
            self.log_qty_stocks.append(0)
            self.log_trade_cost.append(0)
            self.log_stock_price.append(self.data[self.key].iloc[min(self.index+self.window-1, len(self.data))])
            self.log_profit.append(0)
            return 

    def _action(self, action):

        # [0, 1, 2] # Short, Flat, Long - Reminder!

        self.log_action.append(action)

        if action == 1 and self.state == 'Flat':
            self._trade()
            return self.state
        
        elif action == 1 and self.state != 'Flat':
            self.state = 'Flat'
            self._trade()
            return self.state

        elif action == 0 and self.state != 'Go Short':
            self.state = 'Go Short'
            self._trade()
            return self.state

        elif action == 2 and self.state != 'Go Long':
            self.state = 'Go Long'
            self._trade()
            return self.state

        else:
            return self.state

    def _reward(self, action):
        if  self.number_stocks > 0:
            curr  = self.current_stock_price
            entry = self.history_stock_price # Can open, close and it is an arguement
            pos   = action-1
            tc    = self.trade_amount * self.transaction_cost

            self.reward = ((((curr-entry)/entry)*pos)*self.trade_amount)
            self.log_reward.append(self.reward)
            return self.reward
        else:
            self.reward = self.interest_rate
            # self.log_reward.append(self.reward)
            return self.reward




