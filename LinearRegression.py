import numpy as np 

class LinearRegressionCustom:
    # def __init__(self, learning_rate=0.01, epochs=1000):
    #  self.learning_rate = learning_rate
    #   self.epochs = epochs
    #   self.weights = None
    #   self.bias = None
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

   # initialize weights & bias to zeros
    # num_features passed to determine size of weight vector
    def _initialize_weights(self, num_features):
        # initialize_weights & bias with zero
        self.weights = np.zeros(num_features)
        self.bias = 0 

    # calculates mean square error (measures how well the model is performing) 
    # uses current weights and bias & compares them to the actual values
    def _compute_cost(self, X, y):
        m = len(y)
        predictions = np.dot(X, self.weights) + self.bias
        cost = np.sum((predictions - y) ** 2) / (2*m)
        return cost
    
    # here model is trained using gradient descnet
    # adds column of ones to input data to account for bias term
    # weights -> initialized
    # iteratively update weights & bias -> to minimise mean square error
    def fit(self, X, y):
        # adding column of zeros for bias term
        X = np.insert(X, 0, 1, axis = 1)

        # extracting number of features in input data 'X'
        num_features = X.shape[1]

        # initializing weights
        self._initialize_weights(num_features)

        # Gradient Descent calculation
        m = len(y)
        for epoch in range(self.epochs):
            predictions = np.dot(X, self.weights) + self.bias
            errors = predictions - y

            # update weights and bias
            self.weights -= self.learning_rate * np.dot(X.T, errors) / m 
            self.bias -= self.learning_rate * np.sum(errors) / m 

            if epoch % 100 == 0:
                cost = self._compute_cost(X, y)
                # print(f'Epoch {epoch}, Cost: {cost}')

    # adds a column of 1s to input data 
    # kind of like fit
    # uses learned weights and bias to make predictions
    def predict(self, X):
        # add column of ones for bias
        X = np.insert(X, 0, 1, axis = 1)

        # making prediction 
        return np.dot(X, self.weights) + self.bias




from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the California Housing dataset
california_housing = fetch_california_housing()
X, y = california_housing.data, california_housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but often recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the scikit-learn Linear Regression model
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

# Make predictions on the test set
sklearn_predictions = sklearn_model.predict(X_test)

# Evaluate the scikit-learn model
mse = mean_squared_error(y_test, sklearn_predictions)
r2 = r2_score(y_test, sklearn_predictions)

print(f"Scikit-learn Linear Regression Model:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Create and fit our custom Linear Regression model
custom_model = LinearRegressionCustom()
custom_model.fit(X_train, y_train)

# Make predictions on the test set
custom_predictions = custom_model.predict(X_test)

# Evaluate our custom model
mse_custom = mean_squared_error(y_test, custom_predictions)
r2_custom = r2_score(y_test, custom_predictions)

print("\nCustom Linear Regression Model:")
print(f"Mean Squared Error (MSE): {mse_custom}")
print(f"R-squared (R2): {r2_custom}")
