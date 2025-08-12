# %% Import necessary libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error

# %% Load the training data
df = pd.read_csv('./processedData.csv')

# %% Create lag features for the past 7 days
def create_lagged_features(df, lag_days=7):
    for lag in range(1, lag_days + 1):
        df[f'Temp_Max_Lag_{lag}'] = df['Temp Max'].shift(lag)
    return df

# Apply lagging on the training data
df = create_lagged_features(df)

# Drop rows with NaN values created by shifting
df = df.dropna()

# %% Create a Date column in the training data
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Date'].astype(str))

# Convert Date to numeric timestamp
df['Date'] = df['Date'].astype(np.int64) // 10**9  # Convert to seconds since epoch

# %% Update feature and target variables
X = df[['Month', 'Date', 'Year'] + [f'Temp_Max_Lag_{lag}' for lag in range(1, 8)]]
y = df['Temp Max']

# %% Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.1)
model.fit(X_train, y_train)

# %% Make predictions and evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# %% Print evaluation metrics
print('R² Score:', r2)
print('Mean Absolute Error:', mae)

# %% Load the test data
test_df = pd.read_csv('./processedTest.csv')

# %% Create a datetime column for merging
test_df['Date'] = pd.to_datetime(test_df['Year'].astype(str) + '-' + test_df['Month'].astype(str) + '-' + test_df['Date'].astype(str))

# Convert Date to numeric timestamp
test_df['Date'] = test_df['Date'].astype(np.int64) // 10**9  # Convert to seconds since epoch

# %% Print shape of test_df
print("Shape of test_df:", test_df.shape)

# %% Merge test data with training data to include lag features
merged_df = pd.merge(test_df, df[['Date', 'Temp Max']], on='Date', how='left')

# %% Print shape of merged_df
print("Shape of merged_df after merge:", merged_df.shape)

# Print the merged DataFrame before dropping NaN values
print("Merged DataFrame sample before dropping NaNs:\n", merged_df.head())

# Create lag features for the merged test data (may not be sufficient rows for lagging)
merged_df = create_lagged_features(merged_df)

# %% Print the merged DataFrame after creating lag features
print("Merged DataFrame after creating lag features:\n", merged_df.head())

# Drop rows with NaN values
merged_df = merged_df.dropna()

# %% Print shape after dropping NaN values
print("Shape after dropping NaN values:", merged_df.shape)

# Check if merged_df is empty
if merged_df.empty:
    print("Warning: merged_df is empty after dropping NaN values. Check earlier steps.")

# %% Prepare features for prediction
X_test_final = merged_df[['Month', 'Date', 'Year'] + [f'Temp_Max_Lag_{lag}' for lag in range(1, 8)]]

# Make predictions only if X_test_final is not empty
if not X_test_final.empty:
    y_pred = model.predict(X_test_final)

    # Check predictions
    print("Predictions:", y_pred)

    # %% Add predictions to the original test dataframe
    merged_df['maxTemp'] = y_pred

    # %% Print final DataFrame
    print("Final DataFrame sample:\n", merged_df[['Date', 'maxTemp']].head())

    # %% Save the output to a CSV file
    merged_df.to_csv("maxTempOutput.csv", columns=['Date', 'maxTemp'], index=False)
    print("Output CSV saved.")
else:
    print("No predictions made, as the test features DataFrame is empty.")
