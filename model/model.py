import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the data into a pandas DataFrame
data = pd.read_csv('Delhi_v2.csv')
data = data.drop(['Landmarks','Furnished_status','Price_sqft','Address','Status','neworold','type_of_building','desc','Unnamed: 0', 'longitude','latitude'], axis=1)
data.fillna(0, inplace=True)


# Split the data into training and test sets
x = data.drop(['price'], axis = 1)
y = data['price']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)


# Create a RandomForestRegressor model
forest = RandomForestRegressor()

# Fit the model to the training data
forest.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = forest.predict(X_test)

# Evaluate the model using mean squared error
# mse = mean_squared_error(y_test, y_pred)
# print('Mean Squared Error:', mse)
import pickle
with open('model.pkl', 'bw') as f:
    pickle.dump(forest, f)