import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# Load the data
df = pd.read_csv('./processedData.csv')

# Parse the 'Full date' column and set it as index
df['Full date'] = pd.to_datetime(df['Full date'])
df.set_index('Full date', inplace=True)

df['Month'] = df.index.month
df['Day'] = df.index.day
df['Year'] = df.index.year

# Prepare the data for modeling
X = df[['Month', 'Day', 'Year']]
y = df['Temp Max']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R² and MAE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the results
print(f'R² Score: {r2:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')

# Function to predict temperature for a specific date
def predict_temperature(date_str):
    date = pd.to_datetime(date_str)
    features = {
        'Month': date.month,
        'Day': date.day,
        'Year': date.year
    }
    input_df = pd.DataFrame(features, index=[0])
    predicted_temp = model.predict(input_df)
    return predicted_temp[0]

# Predict temperature for a specific date
predict_date = '2010-12-31'  # Change this to the specific date you want to predict
predicted_temperature = predict_temperature(predict_date)
print(f'Predicted Max Temperature for {predict_date}: {predicted_temperature:.2f}°C')
