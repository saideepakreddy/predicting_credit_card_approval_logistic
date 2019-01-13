# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:54:18 2019

@author: saide
"""
#%% importing the required packages
import pandas as pd
import numpy as np
cr_data=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data',header=None)
cr_data.head()

#%% chcking out the data
cr_data.describe()

#%% checking out the data 2
cr_data.info() 
cr_data.tail(17)
#%% replacing missing values '?' with NaN



cr_data= cr_data.replace('?', np.NaN)
print(cr_data.isnull().sum())
cr_data.tail(70)
#%% using mean imputatioons to handling missing data
cr_data.fillna(cr_data.mean(),inplace=True)
cr_data.tail(70)
#%% Removing the NaN in non numeric values using mode imputation
# Iterate over each column of cr_data
for col in cr_data.columns:
    # Check if the column is of object type
    if cr_data[col].dtypes == 'object':
        # Impute with the most frequent value
        cr_data = cr_data.fillna(cr_data[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify


#%% converting all non numeric attributes to Numeric using LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cr_data.columns:
    # Compare if the dtype is object
    if cr_data[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
        cr_data[col]=le.fit_transform(cr_data[col])


#%% Scaling the values
from sklearn.preprocessing import MinMaxScaler

# Drop features 10 and 13 and convert the DataFrame to a NumPy array
cr_data = cr_data.drop([10, 13], axis=1)
cr_data = cr_data.values

# Segregate features and labels into separate variables
X,y = cr_data[:,0:12] , cr_data[:,13]

# Instantiate MinMaxScaler and use it to rescale
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X,[y])

#%%
from sklearn.model_selection import train_test_split

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.25,
                                random_state=143)

#%%
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
# Fit logreg to the train set
logreg.fit(X_train, y_train)

#%%
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(X_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(X_test,y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test,y_pred)

#%%
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol,max_iter=max_iter)

#%%
# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX,y)

# Summarize results
best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_
print("Best: %f using %s" % (best_score,best_params))
