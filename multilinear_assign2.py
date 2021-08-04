# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:41:26 2021

@author: Amarnadh Tadi
"""

# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
computers = pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign11\Computer_Data.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

computers.describe()
computers.drop(['Unnamed: 0'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
computers['cd']=le.fit_transform(computers['cd'])
computers['multi']=le.fit_transform(computers['multi'])
computers['premium']=le.fit_transform(computers['premium'])
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# price
plt.bar(height = computers.price, x = np.arange(1, 6260, 1))
plt.hist(computers.price) #histogram
plt.boxplot(computers.price) #boxplot

# speed
plt.bar(height = computers.speed, x = np.arange(1, 6260, 1))
plt.hist(computers.speed) #histogram
plt.boxplot(computers.speed) #boxplot

# Jointplot
%matplotlib inline
import seaborn as sns
sns.jointplot(x=computers['speed'], y=computers['price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(computers['speed'])
computers['speed'].value_counts
# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(computers.price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(computers.iloc[:, :])
                             
# Correlation matrix 
computers.corr()

##relavancy check of dependent variable with independent variables by scatter plot
#scatterplot speed vs price
plt.scatter(x=computers['speed'],y=computers['price'],color='red')
plt.scatter(x=computers['trend'],y=computers['price'],color='red')



# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price~speed +hd +ram +screen +ads +trend', data = computers).fit() # regression model

# Summary
ml1.summary()
# p-values 

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals


computers_new = computers.drop(computers.index[[1400]])

# Preparing model                  
ml_new = smf.ols('price~ hd +ram +screen +ads +trend', data = computers_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed~hd +ram +screen +ads +trend', data = computers).fit().rsquared  
vif_speed = 1/(1 - rsq_speed) 

rsq_hd = smf.ols('hd~speed +ram +screen +ads +trend', data = computers).fit().rsquared  
vif_hd = 1/(1 - rsq_hd)

rsq_ram = smf.ols('ram~hd  +screen +ads +trend', data = computers).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 

rsq_screen = smf.ols('screen~hd +ram  +ads +trend', data = computers).fit().rsquared  
vif_screen = 1/(1 - rsq_screen) 

# Storing vif values in a data frame
d1 = {'Variables':['speed','hd','ram' ,'screen'], 'VIF':[vif_speed, vif_hd, vif_ram, vif_screen]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price~ hd +ram +screen +ads +trend', data = computers).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(computers)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = computers.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
computers_train, computers_test = train_test_split(computers, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price~ hd +ram +screen +ads +trend', data = computers).fit()

# prediction on test data set 
test_pred = model_train.predict(computers_test)

# test residual values 
test_resid = test_pred - computers_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(computers_train)

# train residual values 
train_resid  = train_pred - computers_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
