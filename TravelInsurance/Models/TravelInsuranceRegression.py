#!/usr/bin/env python
# coding: utf-8

# # 3) Additional Comparative Study
# Thanks to the good results obtained in the comparative study, the company has deployed your
# system and is obtaining good profit. Now a competitor would like to hire you to design a similar
# system for them but, unlike the first system, they would like you to predict not only if the insured
# files a claim but also the value of the claim.
# They provide you with a training set of historical data containing features of each customer
# and a numerical value representing the value of the claim (which may be zero). These data are
# available in the TravelInsurance_Regression.zip archive. In this part of
# the project, you are asked to perform the following two tasks.
# 
# a) Investigate the performance of a number of machine learning procedures on this
# dataset. Using the data in the file TravelInsurance_Regr.csv contained in the TravelInsurance_Regression.zip
# archive, you are required to perform a comparative study of the following machine learning procedures:
# - Linear Regression;
# - at least two more ML technique to predict the value of the claim.
# This company too uses Python internally and therefore Python with scikit-learn is the required
# language and machine learning library for the problem. For this task, you are expected to submit a
# Jupyter Notebook called TravelInsuranceRegression.ipynb containing the Python code used to perform
# the comparative analysis and produce the results as well as the code used to perform the predictions
# described in task “b” below.
# 
# b) Prediction on a hold-out test set. An additional dataset, TravelInsuranceRegr_Test.csv, is provided
# inside the TravelInsurance_Regression.zip archive. Target values are withheld for this test set (i.e. the “Value”
# column is empty). In this second task you are required to produce predictions of the records in
# the test set using one approach of your choice among those tested in task “a” (for example the one
# achieving the best performance). These data must not be used other than to test the algorithm
# trained on the training data.
# 
# As part of your submission you should submit a new version of the file TravelInsurance_Regr.csv in
# CSV format with the missing “Value” column replaced with the output predictions obtained using
# the approach chosen. This second task will be marked based on the mean squared error on the test
# set

# # Multiple Linear Regression
# We will use three regression techniques to solve this problem: multiple linear regression, polynomial regression, and bayesian regression. Remember that in this task we are predicting the value of claim filed by the customer.

# In[152]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load the travel insurance dataset

# In[153]:


data=pd.read_csv("TravelInsurance_Regression\TravelInsurance_Regr.csv")
data.head()


# ### One-hot encoding
# Get dummies is also referred to as one-hot encoding, this is a special feature for our categorical variable.

# In[154]:


dummies=pd.get_dummies(data)
dummies


# ### Pass all columns except 'Target' to the variable X.

# In[155]:


X=dummies.drop(['Target'], axis=1)
X.head()


# ### Use the variable y to store the 'Target' value, also known as the target variable that we will be predicting.

# In[156]:


y=dummies[['Target']]
y.head()


# ### Standardize the dataset

# In[157]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### Train Test Split divides the dataset into 70 percent for training and 30 percent for testing.

# In[158]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ### To build the model, import Linear Regression from the sci-kit learn library

# In[159]:


from sklearn.linear_model import LinearRegression
ml = LinearRegression().fit(x_train,y_train)
print ('Coefficients: ', ml.coef_)
print ('Intercept: ', ml.intercept_)


# ### Let us now apply the predict to our test set.

# In[160]:


y_pred=ml.predict(x_test)
print(y_pred)


# ### Using the different regression metrics, summarise the model's results.
# Note: As one of the requirements mentioned above, the MSE is treated seriously.

# In[161]:


print("Mean absolute error: %.2f" % np.mean(np.absolute(y_test - y_pred)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_test - y_pred) ** 2))
print('Variance score: %.2f' % ml.score(x,y))


# ### Another performance metric used to evaluate the model

# In[162]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# ### Plot a graph to see how accurately the claim values are estimated.

# In[163]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# # Polynomial Regression

# ### Load the travel insurance dataset

# In[227]:


polynomial=pd.read_csv("TravelInsurance_Regression\TravelInsurance_Regr.csv")
polynomial.head()


# ### One-hot encoding

# In[228]:


dummies=pd.get_dummies(polynomial)
dummies


# ### Pass all columns except 'Target' to the variable X.

# In[229]:


X=dummies.drop(['Target'], axis=1)
X.head()


# ### Use the variable y to store the 'Target' value, also known as the target variable that we will be predicting.

# In[230]:


y=dummies[['Target']]
y.head()


# ### Normalize the dataset

# In[231]:


X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### Train Test Split divides the dataset into 80 percent for training and 20 percent for testing.

# In[232]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ### To build the model, import Polynomial Features from the sci-kit learn library

# In[233]:


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures()
x_poly=poly.fit_transform(x_train)
poly.fit(x_train,y_train)


# ### To fit the polynomial features, we must also use the Linear Regression method.

# In[234]:


lin_reg = LinearRegression().fit(x_poly,y_train)


# ### Examine the coeffiecients and intercept

# In[235]:


lin_reg2 = LinearRegression().fit(x_train,y_train)
print ('Coefficients: ', lin_reg2.coef_)
print ('Intercept: ', lin_reg2.intercept_)


# ### Let us now apply the predict to our test set.

# In[236]:


y_pred=lin_reg.predict(poly.fit_transform(x_test))
y_pred


# ### Using the different regression metrics, summarise the model's results.

# In[237]:


print("Mean absolute error: %.2f" % np.mean(np.absolute(y_test - y_pred)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_test - y_pred) ** 2))
print('Variance score: %.2f' % lin_reg2.score(x,y))


# In[238]:


r2_score(y_test,y_pred)


# ### Plot a graph to see how accurately the claim values are estimated.

# In[176]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# # Bayesian Regression

# ### Load the travel insurance dataset

# In[295]:


bayesian=pd.read_csv("TravelInsurance_Regression\TravelInsurance_Regr.csv")
bayesian.head()


# ### One-hot encoding

# In[296]:


dummies=pd.get_dummies(bayesian)
dummies


# ### Pass all columns except 'Target' to the variable X.

# In[297]:


X=dummies.drop(['Target'], axis=1)
X.head()


# ### Use the variable y to store the 'Target' value, also known as the target variable that we will be predicting.

# In[298]:


y=dummies[['Target']]
y.head()


# ### Standardize the dataset

# In[299]:


X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### Train Test Split divides the dataset into 80 percent for training and 20 percent for testing.

# In[300]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ### Examine the coeffiecients and intercept

# In[301]:


from sklearn.linear_model import BayesianRidge
clf = linear_model.BayesianRidge().fit(x_train,y_train)
print ('Coefficients: ', ml.coef_)
print ('Intercept: ', ml.intercept_)


# ### Let us now apply the predict to our test set.

# In[302]:


y_pred=clf.predict(x_test)
print(y_pred)


# In[303]:


y_pred=y_pred.reshape(300,1)


# ### Using the different regression metrics, summarise the model's results.

# In[304]:


print("Mean absolute error: %.2f" % np.mean(np.absolute(y_test - y_pred)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_test - y_pred) ** 2))
print('Variance score: %.2f' % clf.score(x,y))


# In[305]:


r2_score(y_test,y_pred)


# ### Plot a graph to see how accurately the claim values are estimated.

# In[306]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# # Prediction on a hold-out test set (Polynomial Regression)

# ### Load the travel insurance hold-out test set

# In[317]:


pn=pd.read_csv("TravelInsurance_Regression\TravelInsuranceRegr_TestSet.csv")
pn.head()


# ### One-hot Encoding 

# In[318]:


dummies=pd.get_dummies(pn)
dummies


# ### Pass all columns except 'Target' to the variable X.

# In[319]:


X=dummies.drop(['Target'], axis=1)
X.head()


# ### Use the variable y to store the 'Target' value, also known as the target variable that we will be predicting.¶

# In[320]:


y=dummies[['Target']]
y.head()


# ### Normalize the dataset

# In[321]:


X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### Let us now apply the predict to our test set. Remember that polynomial regression yielded the lowest Mean Square Error

# In[322]:


y_pred=lin_reg.predict(poly.fit_transform(x_test))
y_pred


# ### Prepare the result 

# In[324]:


s = pd.DataFrame(y_pred)


# ### Save the model result to file

# In[325]:


s.to_csv("TravelInsurance_Regression\TravelInsuranceRegr_result.csv", index=False)


# ### Summary of the model performance

# In[326]:


result=[{'Mean Absolute Error':396.32,
         'Residual Sum of Squares (MSE)': 251025,
         'R2 Score': 0.79},
        {'Mean Absolute Error':265.77,
         'Residual Sum of Squares (MSE)': 161513.33,
         'R2 Score': 0.86},
        {'Mean Absolute Error': 387.40,
         'Residual Sum of Squares (MSE)': 253833,
         'R2 Score': 0.79}]
       
df=pd.DataFrame(result, index=['Multiple Linear Regression', 'Polynomial Regression','Bayesian Ridge Regression'])
df.head()


# In[ ]:




