# Import necessary libraries
import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Create a sample time series data
def create_time_series_data():
    date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    data = pd.DataFrame(date_rng, columns=['date'])
    data['value'] = (data['date'].dt.dayofyear + (data['date'].dt.dayofyear % 10)).astype(float)
    return data

# Function to fit a Prophet model
def fit_prophet_model(data):
    data.rename(columns={'date': 'ds', 'value': 'y'}, inplace=True)
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the neural network
def train_neural_network(X, y):
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    writer = SummaryWriter()  # Create a SummaryWriter for TensorBoard

    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch)

    writer.close()
    return model

if __name__ == "__main__":
    # Create and print time series data
    time_series_data = create_time_series_data()
    print("Time Series Data:\n", time_series_data.head())

    # Fit and print forecast from Prophet
    forecast = fit_prophet_model(time_series_data)
    print("Prophet Forecast:\n", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Prepare data for the neural network
    X = time_series_data['value'].values[:-1].reshape(-1, 1)
    y = time_series_data['value'].values[1:].reshape(-1, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Train the neural network
    model = train_neural_network(X_tensor, y_tensor)
    print("Neural Network trained successfully.")
