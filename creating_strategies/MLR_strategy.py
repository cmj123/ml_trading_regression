"""
    Title: Machine learning Regression Strategy Template
    Description: This strategy will implement a linear regression model on 
    the selected asset: The model will predict the day's High and day's Low,
    given the day's Open. The model is retrained monthly, and trades are 
    placed only if the predictions are better than a random guess.
    Dataset: US Equities

    ####################### DISCLAIMER #############################
    This is a strategy template only and should not be
    used for live trading without appropriate backtesting and tweking of 
    the strategy parameters.
"""