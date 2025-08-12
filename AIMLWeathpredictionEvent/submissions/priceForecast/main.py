# %% [markdown]
# # Recurrent Neural Network with Keras

# %% [markdown]
# ## Part 1 - Data Preprocessing

# %% [markdown]
# ### Importing the libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# %% [markdown]
# ### Importing the training set

# %%
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# %% [markdown]
# ### Feature Scaling

# %%
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# %% [markdown]
# ### Creating a data structure with 60 timesteps and 1 output

# %%
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# %% [markdown]
# ### Reshaping

# %%
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %% [markdown]
# ## Part 2 - Building and Training the RNN

# %% [markdown]
# ### Initialising the RNN

# %%
regressor = Sequential()

# %% [markdown]
# ### Adding the LSTM layers and Dropout regularisation

# %%
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# %% [markdown]
# ### Adding the output layer

# %%
regressor.add(Dense(units=1))

# %% [markdown]
# ### Compiling the RNN

# %%
regressor.compile(optimizer='adam', loss='mean_squared_error')

# %% [markdown]
# ### Fitting the RNN to the Training set

# %%
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# %% [markdown]
# ## Part 3 - Making the predictions and visualising the results

# %% [markdown]
# ### Getting the real stock price of 2017

# %%
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# %% [markdown]
# ### Getting the predicted stock price of 2017

# %%
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# %% [markdown]
# ### Making predictions

# %%
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# %% [markdown]
# ### Visualising the results

# %%
plt.plot(real_stock_price, color='#8B008B', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.title('Google Stock Price Prediction')
plt.legend()
plt.show()
