import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Data Preprocessing
df = pd.read_csv('./processedData.csv')
df['Date'] = pd.to_datetime(df['Full date'], format="%Y-%m-%d")

df['day_of_year'] = df['Date'].dt.dayofyear
df['sine_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['cosine_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

df = df.drop(columns=['Date', 'day_of_year'])

features = ['Rain', 'sine_day', 'cosine_day']
target = ['Temp Max', 'Temp Min']

scaler = MinMaxScaler()
df[features + target] = scaler.fit_transform(df[features + target])

sequence_length = 7
X, y = [], []
for i in range(sequence_length, len(df)):
    X.append(df[features].iloc[i-sequence_length:i].values)
    y.append(df[target].iloc[i].values)
X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the PyTorch model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output from the LSTM for the dense layers
        x = self.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = X_train.shape[2]
hidden_size = 64
output_size = 2  # 'Temp Max' and 'Temp Min'
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 50
batch_size = 16

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluating the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print("Test Loss:", test_loss.item())

# Making predictions
with torch.no_grad():
    predictions = model(X_test).numpy()
