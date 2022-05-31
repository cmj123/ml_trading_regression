"""
    Title: Machine learning Regression Strategy Template
    Description: This strategy will implement a linear regression model on 
    the selected asset: The model will predict the day's High and day's Low,
    given the day's Open. The model is retrained monthly, and trades are 
    placed only if the predictions are better than a random guess.
    Dataset: US Equities

    ####################### DISCLAIMER #################################
    This is a strategy template only and should not be
    used for live trading without appropriate backtesting and tweking of 
    the strategy parameters.
    ####################################################################
"""

# Machine learning libraries
from os import times_result
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Pipeline
from sklearn.pipeline import Pipeline

# Import numpy
import numpy as np

# Import blueshift libraries
from blueshift.api import symbol, order_target_percent, schedule_function, date_rules, time_rules, get_datetime

# The strategy requires context.lookback number of days

def initalize(context):
    # Define Symbols 
    context.security = symbol('MSFT')

    # The lookback for historical data
    context.lookback = 300

    # The lookback for correlation
    context.correl_lookback = 10

    # The lookback for MA
    context.MA_lookback_short = 3
    context.MA_lookback_medium = 15
    context.MA_lookback_long = 60

    # The train-test split
    context.split_ratio = 0.7

    # The machine learning regressor for Std_U
    context.reg_U =  None 

    # The machine learning regressor for Std_D
    context.reg_D = None 

    # The machine learning regressor accuracy 
    context.accuracy - None 

    # The flag variable is used to check if model needs to be retrained
    context.retrain_flag = True 

    # Schedule the retrain_model function every month
    schedule_function(
                        retrain_model,
                        date_rules = date_rules.month_start(),
                        time_rules = time_rules.market_close(minutes=5)
    )

    # Schedule the rebalance function every day
    schedule_function(
                        rebalance,
                        date_rule = date_rules.every_day(),
                        time_rule = time_rules.market_close(minutes=5)
    )