import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Read data 
file_data = './Data/HousePricePrediction/data.csv'
df = pd.read_csv(file_data)
df.info()

columns = ['date', 'price', 'bedrooms', 
'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view',
        'condition', 'sqft_above',
       'sqft_basement', 
       'yr_built', 'yr_renovated', 'street', 'city',
       'statezip', 'country']

numeric_columns = ['date', 'price', 'bedrooms', 
'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view',
        'condition', 'sqft_above',
       'sqft_basement', 
       'yr_built', 'yr_renovated' ]
numeric_columns_without_price = ['date',  'bedrooms', 
'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view',
        'condition', 'sqft_above',
       'sqft_basement', 
       'yr_built', 'yr_renovated' ]

# we are going to select 
X = df[numeric_columns_without_price].to_numpy
y = df.price.to_numpy

# Split data

test_ratio = 0.33
inside_test = 0.33

X_train_total, X_test_total, y_train_total, y_test = train_test_split(
X, y, test_size = test_ratio, random_state=42)

X_train, X_test_inside, y_train, y_test_inside = train_test_split(
X_train_total, y_train_total, test_size = test_ratio, random_state=42)


scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)


# Training the model 

from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(solver='sgd',
batch_size=32,
 alpha=0,
activation="tanh").fit(X_train, y_train)



