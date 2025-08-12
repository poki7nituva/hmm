import pandas as pd

# Data Preprocessing
df = pd.read_csv('./test.csv')
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
print(df)
df.to_csv('processedTest.csv', index=False) #make prototypes in knime