# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:53:02 2021

@author: Amarnadh Tadi
"""

# Multilinear Regression
import pandas as pd
import numpy as np
startup=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign11\50_Startups.csv")
startup.describe()
startup.columns
# Exploratory data analysis:



#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# R&D Spend
plt.bar(height = startup['R&D Spend'], x = np.arange(1, 51, 1))
plt.hist(startup['R&D Spend']) #histogram
plt.boxplot(startup['R&D Spend']) #boxplot

# Marketing Spend
plt.bar(height = startup['Marketing Spend'], x = np.arange(1, 51, 1))
plt.hist(startup['Marketing Spend']) #histogram
plt.boxplot(startup['Marketing Spend']) #boxplot
##relavancy check of dependent variable with independent variables by scatter plot
#scatterplot R&D Spend(independent) vs profit
%matplotlib inline
plt.scatter(x=startup['rd'],y=startup['Profit'],color="red")
plt.scatter(x=startup['admin'],y=startup['Profit'],color="red")
plt.scatter(x=startup['ms'],y=startup['Profit'],color="red")

startup.drop(['State'],axis=1,inplace=True)
startup.head()
startup.columns
# Jointplot
import seaborn as sns
sns.jointplot(x=startup['R&D Spend'], y=startup['Marketing Spend'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(startup['Marketing Spend'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startup['Profit'], dist = "norm", plot = pylab)
plt.show()
startup.rename(columns={'R&D Spend':'rd','Administration':'admin','Marketing Spend':'ms'},inplace=True)
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup.iloc[:, :])
                             
# Correlation matrix 
startup.corr()

## exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit ~ rd + admin + ms ', data = startup).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 46,48,49 is showing high influence so we can exclude that entire row

startup_new = startup.drop(startup.index[[46,48,49]])

# Preparing model                  
ml_new = smf.ols('Profit ~ rd + admin + ms + c+f+n', data = startup_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_rd = smf.ols('rd ~  admin + ms', data = startup).fit().rsquared  
vif_rd = 1/(1 - rsq_rd) 

rsq_admin = smf.ols('admin ~  rd + ms', data = startup).fit().rsquared  
vif_admin = 1/(1 - rsq_admin)

rsq_ms = smf.ols('ms ~  admin + rd', data = startup).fit().rsquared  
vif_ms = 1/(1 - rsq_ms) 


# Storing vif values in a data frame
d1 = {'Variables':['rd', 'admin', 'ms'], 'VIF':[vif_rd, vif_admin, vif_ms]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Profit ~ rd + admin + ms', data = startup).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(startup)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startup.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train, startup_test = train_test_split(startup, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Profit ~ rd + admin + ms", data = startup_train).fit()

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid = test_pred - startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
