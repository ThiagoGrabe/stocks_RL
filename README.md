# Stock Prices Reinforcement Learning - Gym Environment

## Dependencies

This project uses the [Pipenv](https://github.com/pypa/pipenv) to packages version control

## Installation

```bash
cd rl_stocks
pip install -e .
```

## How to

As the project is still in progress, some features are simplified. You may find an example of how to run the project below:

```bash
python3.8 main.py -s MSFT -l 24 -i 1m -p 7 -d 01 -m 03 -y 2021

Arguments:
  -h, --help            show this help message and exit
  -s STOCK, --stock STOCK
                        Stock to be analysed.
  -l LENGTH, --length LENGTH
                        State length to be analysed.
  -i INTERVAL, --interval INTERVAL
                        Interval to be analysed.
  -p PERIOD, --period PERIOD
                        Period (in days) to be analysed.
  -d DAY, --day DAY     Start day to run the RL algorithm
  -m MONTH, --month MONTH
                        Start month to run the RL algorithm
  -y YEAR, --year YEAR  Start year to run the RL algorithm

```