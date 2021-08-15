import pandas as pd
df_A = pd.read_csv('Dates.csv', sep=";")      # Big
df_A = pd.read_csv('Dates2.csv', sep=";")     # Small
df_B = pd.read_csv('GSPC.csv', sep=";") 
df_C = df_A.merge(df_B, how='left', on=['Date'])
print(df_C)


### New Variables ###
#####################
import pandas as pd

# Load the specific market dataset
df_raw = pd.read_csv('DJI_TN.csv', index_col='Date') 
df_raw = pd.read_csv('NASDAQ_TN.csv', index_col='Date') 
df_raw = pd.read_csv('NYSE_TN.csv', index_col='Date') 
df_raw = pd.read_csv('RUSSELL_TN.csv', index_col='Date') 
df_raw = pd.read_csv('SP_TN.csv', index_col='Date') 

# assign only technical columns to new dataframe
df_market = df_raw[["Close", "Volume"]]

### Add new technical features 
import talib

# Moving Average Convergence/Divergence
df_market['RSI']  = talib.RSI(df_market['Close'], timeperiod=10)

# Simple Moving Average (5-10-15-20 days) 
df_market['SMA_5']  = talib.SMA(df_market['Close'], timeperiod=5)
df_market['SMA_10'] = talib.SMA(df_market['Close'], timeperiod=10)
df_market['SMA_15'] = talib.SMA(df_market['Close'], timeperiod=15)
df_market['SMA_20'] = talib.SMA(df_market['Close'], timeperiod=20)

# Weighted Moving Average (10-20 days)
df_market['WMA_10'] = talib.WMA(df_market['Close'], timeperiod=10)
df_market['WMA_20'] = talib.WMA(df_market['Close'], timeperiod=20)

# Triple exponential moving average
df_market['TEMA_20'] = talib.TEMA(df_market['Close'], timeperiod=20)

# Chande Momentum Oscillator
df_market['CMO_10'] = talib.CMO(df_market['Close'], timeperiod=10)

# Percentage Price Oscillator
df_market['PPO'] = talib.PPO(df_market['Close'], fastperiod=12, slowperiod=26, matype=0)


# Remove the redundant columns in df_market
del df_market['Close']
del df_market['Volume']

# Merge the datasets to form one final dataset
df_C = df_raw.merge(df_market, how='left', on=['Date'])

# Output the new dataset
df_C.to_csv('DJI_TN.csv', index_label='Date')
df_C.to_csv('NASDAQ_TN.csv', index_label='Date')
df_C.to_csv('NYSE_TN.csv', index_label='Date')
df_C.to_csv('RUSSELL_TN.csv', index_label='Date')
df_C.to_csv('SP_TN.csv', index_label='Date')

# In case of inspecting the newly created dataset
df_A = pd.read_csv('SP_TN.csv', index_col='Date') 


