# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:28:48 2021

@author: Amarnadh Tadi
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
car = pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign11\ToyotaCorolla.csv",encoding= 'unicode_escape')
car.dtypes
car.columns
car.drop(['Model'],axis='columns')
color=pd.get_dummies(car['Color'])
fuel_type=pd.get_dummies(car['Fuel_Type'])
cars_new=pd.concat([car,color,fuel_type],axis=1)
cars_new.drop(['Color','Fuel_Type','Model'],axis='columns',inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# price

plt.hist(cars_new['Price']) #histogram
plt.boxplot(cars_new['Price']) #boxplot

#'Sport_Model'

plt.hist(cars_new['Sport_Model']) #histogram
plt.boxplot(cars_new['Sport_Model']) #boxplot
##relavancy check of dependent variable with independent variables by scatter plot
#scatterplot R&D Spend(independent) vs profit
%matplotlib inline
plt.scatter(x=cars_new['Sport_Model'],y=cars_new['Price'],color="red")
plt.scatter(x=cars_new['HP'],y=cars_new['Price'],color="red")
plt.scatter(x=cars_new['KM'],y=cars_new['Price'],color="red")



# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cars_new['Price'], dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars_new.iloc[:, :])
                             
# Correlation matrix 
cars_new.corr()

X = cars_new
X.head()
y = car.iloc[:,[2]]



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
score