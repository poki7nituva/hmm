import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
df = pd.read_csv('./processedData.csv')

# Parse the 'Full date' column and set it as index
df['Full date'] = pd.to_datetime(df['Full date'])
df.set_index('Full date', inplace=True)

# Display the first few rows of the DataFrame
print(df.head())

# Visualize the temperature data
plt.figure(figsize=(12, 6))
plt.plot(df['Temp Max'], label='Max Temperature', color='orange')
plt.title('Max Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Max Temperature')
plt.legend()
plt.show()

# Feature Engineering: Extract features from the date
df['DayOfYear'] = df.index.dayofyear
df['Month'] = df.index.month
df['Day'] = df.index.day
df['Year'] = df.index.year

# Split the data into training and testing sets (80-20 split)
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Define features and target variable
X_train = train[['DayOfYear', 'Month', 'Day', 'Year']]
y_train = train['Temp Max']
X_test = test[['DayOfYear', 'Month', 'Day', 'Year']]
y_test = test['Temp Max']

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
r_squared = r2_score(y_test, predictions)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r_squared:.2f}')

# Visualize actual vs predicted temperatures
plt.figure(figsize=(12, 6))
plt.plot(test['Temp Max'], label='Actual Max Temperature', color='orange')
plt.plot(test.index, predictions, label='Predicted Max Temperature', color='blue', linestyle='--')
plt.title('Actual vs Predicted Max Temperature')
plt.xlabel('Date')
plt.ylabel('Max Temperature')
plt.legend()
plt.show()

# Function to predict temperature for a new date
def predict_temperature(date):
    date = pd.to_datetime(date)
    features = np.array([[date.dayofyear, date.month, date.day, date.year]])
    return model.predict(features)[0]

# Example usage
new_date = '2009-11-25'
predicted_temp = predict_temperature(new_date)
print(f'Predicted Max Temperature for {new_date}: {predicted_temp:.2f}')
