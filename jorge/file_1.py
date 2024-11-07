from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Load the dataset
ethereum_data = pd.read_csv('/Users/ditsy/PythonProjects/MachineLearning/jorge/Ethereum_4_7_2018-6_6_2018_historical_data_coinmarketcap.csv')

# Select relevant variables
columns_to_use = ['open', 'high', 'low', 'volume', 'marketCap']
target_column = 'close'

# Filter the dataset for selected variables
data = ethereum_data[columns_to_use + [target_column]].dropna()

# Split features (X) and target (y)
X = data[columns_to_use]
y = data[target_column]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Save predictions alongside actual values for comparison
predictions = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

# Save to a CSV or display
predictions.to_csv('/Users/ditsy/PythonProjects/MachineLearning/jorge/ethereum_predictions.csv', index=False)
print(predictions.head())