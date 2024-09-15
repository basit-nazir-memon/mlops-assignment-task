# main.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

# Function to load and process data
def load_data(file_path):
    HouseDF = pd.read_csv(file_path)
    return HouseDF

# Function for exploratory data analysis
def perform_eda(HouseDF):
    
    # Pairplot, distribution of prices, heatmap of correlations
    sns.pairplot(HouseDF)
    plt.show()
    
    sns.distplot(HouseDF['Price'])
    plt.show()
    
    sns.heatmap(HouseDF.corr(), annot=True)
    plt.show()

# Function to train and save the Linear Regression model
def train_model(HouseDF):
    # Define X and y
    X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                 'Avg. Area Number of Bedrooms', 'Area Population']]
    y = HouseDF['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

    # Create and train the Linear Regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Save the model using pickle
    with open('lmmodel.pkl', 'wb') as file:
        pickle.dump(lm, file)

    # Print intercept and coefficients
    print("Intercept: ", lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)

    # Predictions and evaluation
    predictions = lm.predict(X_test)

    # Plot the results
    plt.scatter(y_test, predictions)
    plt.show()
    
    sns.distplot((y_test - predictions), bins=50)
    plt.show()

    # Regression Evaluation Metrics
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Function to load the saved model and make predictions
def load_and_predict(input_data):
    with open('lmmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict([input_data])
    return prediction

# Main function to execute the script
if __name__ == "__main__":
    # Load the dataset
    data_file = 'USA_Housing.csv'
    HouseDF = load_data(data_file)

    # Perform exploratory data analysis
    perform_eda(HouseDF)

    # Train the model and evaluate
    train_model(HouseDF)

    # Example input for prediction
    example_input = [68707.067178656784, 5.1865889840310001, 8.1512727430375099, 4.51, 38259.39970458]  # Adjust accordingly
    prediction = load_and_predict(example_input)

    print(f"Predicted house price: {prediction[0]}")
