# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:16:29 2021

@author: Amarnadh Tadi
"""



# Multilinear Regression
import pandas as pd
import numpy as np
avacado=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign11\Avacado_Price.csv")
avacado.describe()
avacado.columns
avacado['type'].value_counts()
avacado['region'].value_counts()
avacado.dtypes
# Exploratory data analysis
avacado.drop(['region'],axis=1,inplace=True)
type_dummies=pd.get_dummies(avacado['type'])
avacado_new=pd.concat([avacado,type_dummies],axis=1)
avacado_new.drop(['type'],axis='columns',inplace=True)
    



#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#Average price

plt.hist(avacado_new['AveragePrice']) #histogram
plt.boxplot(avacado_new['AveragePrice']) #boxplot

#large bags

plt.hist(avacado_new['Large_Bags']) #histogram
plt.boxplot(avacado_new['Large_Bags']) #boxplot
##relavancy check of dependent variable with independent variables by scatter plot
#scatterplot R&D Spend(independent) vs profit
%matplotlib inline
plt.scatter(x=avacado_new['Large_Bags'],y=avacado_new['AveragePrice'],color="red")
plt.scatter(x=avacado_new['Total_Volume'],y=avacado_new['AveragePrice'],color="red")
plt.scatter(x=avacado_new['tot_ava3'],y=avacado_new['AveragePrice'],color="red")
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
scale.fit(avacado_new)

avacado_new.describe()
x=avacado_new.iloc[:,1:12]
y=avacado_new.iloc[:,[0]]


# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(avacado_new['AveragePrice'], dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(avacado_new.iloc[:, :])
                             
# Correlation matrix 
avacado_new.corr()

## exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('y~x ', data = avacado_new).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 46,48,49 is showing high influence so we can exclude that entire row

avacado_final = avacado_new.drop(avacado_new.index[[15560,17468]])
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)



# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
score
