import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Loading the data
df = pd.read_csv('INTERNSHIP_DATA.csv')

# Cleaning and converting CLOCK to datetime
df['CLOCK'] = df['CLOCK'].str.strip()
df['CLOCK'] = pd.to_datetime(df['CLOCK'], format='mixed', dayfirst=True, errors='coerce')
df = df.dropna(subset=['CLOCK'])

# Setting CLOCK as index
df.set_index('CLOCK', inplace=True)

# Creating lagged features for output (10, 20, 30 minutes ago)
for lag in [1, 2, 3]:  # Assuming 10-minute intervals
    df[f'output_lag_{lag*10}min'] = df['output'].shift(lag)

# Dropping rows with NaN due to lagging
df = df.dropna()

# Features (X1–X14 + lagged outputs)
features = [col for col in df.columns if col.startswith('X') or col.startswith('output_lag')]
X = df[features]
y = df['output']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data (preserve time order)
train_size = int(0.8 * len(X))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Training XGBoost
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Forecasting
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"Time Series Forecasting Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Plotting actual vs forecasted
plt.figure(figsize=(15, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Forecasted')
plt.title('Actual vs Forecasted Stack Emission (20-min Ahead)')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.savefig('forecast_plot.png')
plt.close()

# Saving results
results = {'RMSE': rmse, 'MAE': mae}
pd.Series(results).to_csv('forecast_results.csv')