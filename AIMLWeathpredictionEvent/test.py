import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
sns.set_style('darkgrid')


# Load the data
data = pd.read_csv('processedData.csv')

# Convert the 'Full date' column to datetime format if not already
data['Full date'] = pd.to_datetime(data['Full date'])

# Split data into training and testing sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Set up training features and target variable
train_X = train[['Year', 'Month', 'Day']]
train_y = train['Temp Max']

# Set up testing features
test_X = test[['Year', 'Month', 'Day']]
test_y = test['Temp Max']  # True values for comparison

# Visualization of temperature and rainfall over the years
fig, ax = plt.subplots(8, 2, figsize=(12, 30))
ax = ax.flatten()

for i in range(0, 16, 2):
    train_Year_i = train[train['Year'] == 1951 + i]
    ax[i].plot(train_Year_i.index, train_Year_i['Temp Max'], label='Max Temp', color='orange')
    ax[i].plot(train_Year_i.index, train_Year_i['Temp Min'], label='Min Temp', color='blue')
    ax[i + 1].plot(train_Year_i.index, train_Year_i['Rain'], label='Rainfall', color='green')
    ax[i].set_title(f'Temperatures of Year {1951 + i}')
    ax[i + 1].set_title(f'Rainfall of Year {1951 + i}')
    ax[i].legend()
    ax[i + 1].legend()

plt.tight_layout()
plt.show()


# Function to train, evaluate, and plot results
def fit_train_evaluate_plot(model):
    model.fit(train_X, train_y)
    
    # Make predictions
    temp_max_preds = model.predict(test_X)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true=test_y, y_pred=temp_max_preds)
    r2 = r2_score(y_true=test_y, y_pred=temp_max_preds)
    
    print(f'Mean Absolute Error = {mae}')
    print(f'R² Score = {r2}')
    
    # Plot the results
    plot_data = pd.DataFrame({
        'Year': test['Year'],
        'Temp Max True': test_y,
        'Temp Max Pred': temp_max_preds
    })
    
    fig, ax = plt.subplots(8, 2, figsize=(12, 30))
    ax = ax.flatten()
    
    for i in range(0, 16, 2):
        plot_Year_i = plot_data[plot_data['Year'] == 2000 + i]
        ax[i].plot(plot_Year_i.index, plot_Year_i['Temp Max True'], label='True', color='red')
        ax[i].plot(plot_Year_i.index, plot_Year_i['Temp Max Pred'], label='Predictions', color='blue')
        ax[i].set_title(f'Temperatures of Year {2000 + i}')
        ax[i].set_xlabel('Index')
        ax[i].set_ylabel('Temperature (°C)')
        ax[i].legend()
        
        plot_Year_i = plot_data[plot_data['Year'] == 2000 + i + 1]
        ax[i+1].plot(plot_Year_i.index, plot_Year_i['Temp Max True'], label='True', color='red')
        ax[i+1].plot(plot_Year_i.index, plot_Year_i['Temp Max Pred'], label='Predictions', color='blue')
        ax[i+1].set_title(f'Temperatures of Year {2000 + i + 1}')
        ax[i+1].set_xlabel('Index')
        ax[i+1].set_ylabel('Temperature (°C)')
        ax[i+1].legend()

    plt.tight_layout()
    plt.show()


# Instantiate and evaluate the model
model = XGBRegressor()
fit_train_evaluate_plot(model)
