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
from curses import window
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


def retrain_model(context, data):
    """
        A function to retrain the regression. This function is called by the 
        schedule_function in the initialize function.
    """
    context.retrain_flag = True 


def rebalance(context, data):
    """
        A function to rebalance the portfolio. This function is called by the 
        schedule_function in the initialize function.
    """

    # Fetch lookback no. days data for the security
    try:
        Df = data.history(
            context.security,
            ['open','high','low', 'close'],
            context.lookback + 1,
            '1d'
        )
    except IndexError:
        return

    # Calculate short moving average of close prices
    Df['S_short'] = Df['close'].shift(1).rolling(
        window=context.MA_lookback_short).mean()

    # Calculate medium moving average of close prices
    Df['S_medium'] = Df['close'].shift(1).rolling(
        window=context.MA_lookback_medium).mean()

    # Calculate long moving average of close prices
    Df['S_long'] = Df['close'].shift(1).rolling(
        window=context.MA_lookback_long).mean()

    # Calculate the correlation between close price and short moving average
    Df['Corr'] = Df['close'].shift(1).rolling(
        window=context.correl_lookback).corr(Df['S_short'].shift(1))

    Df['Std_U'] = Df['high'] - Df['open']
    Df['Std_D'] = Df['open'] - Df['low']

    Df['OD'] = Df['open'] - Df['open'].shift(1)
    Df['OL'] = Df['open'] - Df['close'].shift(1)
    Df.dropna(inplace=True)

    X_U = Df[['open', 'S_short', 'S_medium', 'S_long', 'OD', 'OL', 'Corr']]
    X_D = X_U
    yU = Df['Std_U']
    yD = Df['Std_D']

    # Calculate the split ratio
    t = context.split_ratio
    split = int(t*len(Df))

    if context.retrain_flag:
        context.retrain_flag = False 
        steps = [('scaler', StandardScaler()), 
                 ('linear', LinearRegression())]
        
        pipeline = Pipeline(steps)
        parameters = {'linear_fit_intercept':[0,1]}

        context.reg_U = GridSearchCV(pipeline, parameters, cv=5)
        context.reg_U.fit(X_U[:split], yU[:split])
        best_fit_U = context.reg_U.best_params_['linear__fit_intercept']
        context.reg_U = LinearRegression(fit_intercept=best_fit_U)
        context.reg_U.fit(X_U[:split], yU[:split])

        context.reg_D = GridSearchCV(pipeline, parameters, cv=5)
        context.reg_D.fit(X_D[:split], yD[:split])
        best_fit_D = context.reg_D.best_params_['linear__fit_intercept']
        context.reg_D = LinearRegression(fit_intercept=best_fit_D)
        context.reg_D.fit(X_D[:split], yD[:split])

    yU_predict = context.reg_U.predict(X_U[split:])
    # Assign the predicted values to a new column in the dataframe
    Df.reset_index(inplace=True)
    Df['Max_U'] = 0 
    Df.loc[Df.index >= split, 'Max_U'] = yU_predict 
    Df.loc[Df['Max_U'] < 0, 'Max_U'] = 0

    yD_predict = context.reg_D.predict(X_D[split:])
    # Assign the predicted values to a new column in the data frame
    Df['Max_D'] = 0
    Df.loc[Df.index >= split, 'Max_D'] = yD_predict 
    Df.loc[Df['Max_D'] < 0, 'Max_D'] = 0

    # Use the predicted upside deviation values to calculate the high price
    Df['P_H'] = Df['open'] + Df['Max_U']
    Df['P_L'] = Df['open'] - Df['Max_D']

    context.accuracy = len(
        Df[(Df['P_H'] >= Df['high']) & 
        (Df['P_L'] <= Df['low'])]) * 1.0 / len(X_U[split:])

    # Trading signal
    sell_sig = list((Df['high'] > Df['P_H']) & (Df['low'] > Df['P_L']))[-1]
    buy_sig  = list((Df['high'] < Df['P_H']) & (Df['low'] < Df['P_L']))[-1]

    # Long, short and exit conditions
    long_entry = buy_sig and context.accuracy > 0.3
    short_entry = sell_sig and context.accuracy > 0.3
    exit_position = context.accuracy < 0.3

    # Place the orders
    if long_entry:
        print("{} Long entry condition is {}".format(get_datetime(), long_entry))
        print("{} Going long in: {}".format(get_datetime(), context.security.symbol))
        order_target_percent(context.security, 1)

    elif short_entry:
        print("{} Short entry condition is {}".format(get_datetime(), short_entry))
        print("{} Going short in: {}".format(get_datetime(), context.security.symbol))
        order_target_percent(context.security, -1)

    elif exit_position:
        order_target_percent(context.security, 0)

    