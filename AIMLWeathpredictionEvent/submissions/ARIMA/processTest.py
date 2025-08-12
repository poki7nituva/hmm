import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Data Preprocessing
df = pd.read_csv('./test.csv')
pd.set_option('display.max_rows',200)

dateList = pd.date_range(start='2010-01-01', end='2020-10-24').tolist()
df['Date'] = dateList

dates = []
months = []
years = []
for x in df['Date'] :
    temp = str(x).split("-")
    years.append(temp[0])
    months.append(temp[1])
    dates.append(temp[2][:2])
df['Full date'] = df['Date']
df['Year'] = years 
df['Month'] = months
df['Date'] = dates
df.to_csv('processedTest.csv', index=False) #make prototypes in knime