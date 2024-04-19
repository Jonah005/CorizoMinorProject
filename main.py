# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("california_housing_prices.csv")  # Replace with your dataset path

# Separate features and target variable
features = data.drop("median_house_value", axis=1)  # All columns except target
target = data["median_house_value"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2:.2f}")

# Make a prediction for a new house (replace values with your own data)
new_house = [[8, 30, 20000, 0.5]]  # Example features (replace with actual values)
predicted_price = model.predict(new_house)[0]
print(f"Predicted price for the new house: ${predicted_price:.2f}")
