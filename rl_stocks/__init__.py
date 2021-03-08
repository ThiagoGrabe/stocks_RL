from gym.envs.registration import register

register(
    id='rl_stocks-v0',
    entry_point='rl_stocks.envs:StocksRL',
)
