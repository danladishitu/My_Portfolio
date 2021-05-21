#!/usr/bin/env python
# coding: utf-8

# # Travel Insurance Predictive Models

# ### 1) Pilot Study
# 
# Imagine that you work as Machine Learning independent consultant, providing scientific advisory and consulting services to companies seeking to apply data analytics to their business activities. A travel insurance company would like to offer a discounted premium (for the same cover requested) to customers that are less likely to make a future claim. The manager contacts you to investigate the feasibility of using machine learning to predict whether a customer will file a claim on their travel. The manager has access to historical data of past policies and she offers to provide you with information about the insured, the purpose and destination of the travel, and whether a claim was filed.
# 
# In the first part of your project, you are asked to write a detailed proposal for a pilot study to investigate whether machine learning procedures could be used to successfully solve this problem.
# 
# Your report should discuss several aspects of the problem, including the following main points:
# - the type of predictive task that must be performed (e.g., classification, regression, clustering, rules mining, ...);
# - examples of possibly informative features that you would like to be provided with (what type of information that the company could obtain from/about the customer is likely to be a good predictor?);
# - the learning procedure or procedures (e.g., DTs, k-NN, k-means, linear regression, Apriori, SVMs, ...) you would choose and the reason for your choice;
# - how you would evaluate the performance of your system before deploying it.
# 
# You can assume that the manager has some knowledge of machine learning and you do not need to explain how the recommended learning method works. Simply discuss your recommendation and back it with sound arguments and/or references.

# ### 2) Comparative Study
# Thanks to the convincing arguments in your pilot-study proposal, the company decides to collect the data that you suggested and to hire you to perform the proposed study. They provide you with a training set of historical data containing features of each customer and a label representing whether
# the insured filed a claim. These data are available shared in Github repository. In this part of the project, you are asked to perform the following two tasks.
# 
# a) Investigate the performance of a number of machine learning procedures on this dataset. Using the data in the file TravelInsurance.csv contained in the TravelInsurance_Classification.zip archive, you are required to perform a comparative study of the following machine learning procedures:
# - a Decision Tree classifier;
# - at least two more ML technique to predict if the insured will file a claim.
# You will notice that one of the features, is missing for some of the instances. You are therefore required to deal with the problem of missing features before you can proceed with the prediction step. As a baseline approach you may try to discard the feature altogether and train on the remaining
# features. You are then encouraged to experiment with different inputation methods.
# The company uses Python internally and therefore Python with scikit-learn is the required
# language and machine learning library for the problem. For this task, you are expected to submit a
# Jupyter Notebook called TravelInsurance_Classification.ipynb containing the Python code used to perform
# the comparative analysis and produce the results, as well as the code used to perform the predictions
# described in task “b” below.
# 
# b) Prediction on a hold-out test set. An additional dataset, TravelInsurance_TestSet.csv, is provided
# inside the TravelInsurance_Classification.zip archive. Binary outcomes are withheld for this test set (i.e. the
# “Class” column is empty). In this second task you are required to produce class predictions of
# the records in the test set using one approach of your choice among those tested in task “a” (for
# example the one achieving the best performance). These data must not be used other than to test
# the algorithm trained on the training data.
# As part of your submission you should submit a new version of the file TravelInsurance_TestSet.csv in
# CSV format with the missing class replaced with the output predictions obtained using the approach
# chosen. This second task will be marked based on the prediction accuracy on the test set.
# 

# # Decision Tree classifier
# We will build a model using four different classification algorithms which are **decision tree**, **support vector machine**, **logistic regression** and **random forest**. This is the very first machine learning technique we will use to approach the problem and compare the various performance metrics.

# ### Import all the libraries from Sci-kit learn

# In[505]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline


# In[506]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score


# ### Load the travel insurance dataset

# In[507]:


data=pd.read_csv("TravelInsurance_Classification\TravelInsurance.csv")
data.head(7)


# ### Examine the number of rows and columns in the dataset

# In[508]:


data.shape


# ### Explore the dataset and see if there are any missing values.

# In[509]:


data.isnull().sum()


# ### Drop the column 'F15,' which contains 750 missing values.

# In[510]:


data.drop(['F15'], axis=1, inplace=True)


# ### Examine the dataset to ensure that the modifications were made right.

# In[511]:


data.head()


# ### Pass all columns except 'class' to the variable X.

# In[512]:


X = data[['F1','F2','F3', 'F4', 'F5', 'F6', 'F7','F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']].values 
X[0:5]


# ### Use the variable y to store the 'class' value, also known as the target variable that we will be predicting.

# In[513]:


y=data["Class"] 
y[0:5]


# ### Normalize the dataset 

# In[514]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X= scale.fit_transform(X)
X


# ### Over-sampling and under-sampling on unbalanced data

# In[515]:


import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE


# In[516]:


print(imblearn.__version__)

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import EditedNearestNeighbours 

oversample = SMOTE()
enn = EditedNearestNeighbours()
# label encode the target variable

y = LabelEncoder().fit_transform(y)

X, y = enn.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
    per = v / len(y) * 100

    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
#plot the distribution
plt.bar(counter.keys(), counter.values())
plt.show()


# ### Train Test Split divides the dataset into 80 percent for training and 20 percent for testing.

# In[517]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ### To build the model, use the decision tree classifier.

# In[518]:


fileclaimedTree = DecisionTreeClassifier()
fileclaimedTree


# ### Grid search cross validation hyperparameter tuning will be used to improve our model's performance accuracy.

# In[519]:


#pass the parameters for decision tree classifier
param_dict={"criterion":['gini', 'entropy'],
           "max_depth":range(1,12),
           "min_samples_split": range(1,12),
           "min_samples_leaf": range(1,5)}


# ### This informs us of the best parameter for our model's efficiency.

# In[520]:


grid= GridSearchCV(fileclaimedTree, param_grid=param_dict, cv=10, verbose=1, n_jobs=-1)
grid.fit(x_train, y_train)


# ### This will yield the most effective parameters for our decision tree classifier.

# In[521]:


grid.best_params_


# ### Examine the estimator 

# In[522]:


grid.best_estimator_


# ### Investigate the optimal score for the decision tree classifier's parameters.

# In[523]:


grid.best_score_


# ### Now, apply the above-mentioned parameters for the decision tree classifier

# In[ ]:


fileclaimedTree = DecisionTreeClassifier(criterion= 'entropy', max_depth=5, min_samples_split=6, min_samples_leaf=3)
fileclaimedTree


# ### To train the model, use the fit function.

# In[525]:


fileclaimedTree.fit(x_train,y_train)


# ### Let us now apply the predict to our test set.

# In[526]:


y_pred = fileclaimedTree.predict(x_test)


# ### We can use cross validation to further analyse the model's performance on its test set.

# In[527]:


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

print('Accuracy of Decision Tree classifier on the training set: {:.2f}'.format(fileclaimedTree.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on the test set: {:.2f}'.format(fileclaimedTree.score(x_test, y_test)))

#Decision Trees are very prone to overfitting as shown in the scores

score = cross_val_score(fileclaimedTree, x_train, y_train, cv=10) 
print('Cross-validation score: ',score)
print('Cross-validation mean score: ',score.mean())


# ### Let's build a function that will return the results of various metrics.

# In[528]:


def summarize_classification(y_test,y_pred,avg_method='weighted'):
    acc = accuracy_score(y_test, y_pred,normalize=True)
    num_acc = accuracy_score(y_test, y_pred,normalize=False)
    f1= f1_score(y_test, y_pred, average=avg_method)
    prec = precision_score(y_test, y_pred, average=avg_method)
    recall = recall_score(y_test, y_pred, average=avg_method)
    jaccard = jaccard_score(y_test, y_pred, average=avg_method)
    
    print("Length of testing data: ", len(y_test))
    print("accuracy_count : " , num_acc)
    print("accuracy_score : " , acc)
    print("f1_score : " , f1)
    print("precision_score : " , prec)
    print("recall_score : ", recall)
    print("jaccard_score : ", jaccard)


# ### Summarize the model's performance using the various classification metrics.

# In[529]:


summarize_classification(y_test, y_pred)


# In[530]:


y_pred = fileclaimedTree.predict(x_test)


# ### Let us now compare how well our model predicts on unseen data.

# In[615]:


#Accuracy performance
pred_results = pd.DataFrame({'y_test': pd.Series(y_test),
                             'y_pred': pd.Series(y_pred)})

pred_results.sample(15)


# In[532]:


from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[533]:


import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


# ### Plot the graph of a Decision Tree classifier

# In[534]:


dot_data = StringIO()
filename = "fileclaimedTree.png"
featureNames = data.columns[0:14]
targetNames = data["Class"].unique().tolist()
dotout=tree.export_graphviz(fileclaimedTree,feature_names=featureNames, out_file=dot_data, filled=True, special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(300, 300))
plt.imshow(img,interpolation='nearest')


# # Support Vector Machine
# This is another classification technique that is renowned for producing high model performance precision. We will use the same approach as we did with the decision tree above. The only feature that will be change here is just the use of a different classifier.

# ### Load the travel insurance dataset

# In[535]:


datasvm=pd.read_csv("TravelInsurance_Classification\TravelInsurance.csv")
datasvm.head(7)


# ### Explore the dataset and see if there are any missing values.

# In[536]:


datasvm.isnull().sum()


# ### Drop the column 'F15,' which contains 750 missing values.

# In[537]:


datasvm.drop(['F15'], axis=1, inplace=True)


# ### Examine the dataset to ensure that the modifications were made right.

# In[538]:


datasvm.head()


# ### Pass all columns except 'class' to the variable X.

# In[539]:


datasvm=data[['F1','F2','F3', 'F4', 'F5', 'F6', 'F7','F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']].values 
X= np.asarray(datasvm)
X[0:5]


# ### Use the variable y to store the 'class' value, also known as the target variable that we will be predicting.

# In[540]:


datasvm=data["Class"] 
y= np.asarray(datasvm)
y[0:5]


# ### Standardize the dataset

# In[541]:


scale=StandardScaler()
X= scale.fit_transform(X)
X


# ### Over-sampling and under-sampling on unbalanced data

# In[542]:


import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE


# In[543]:


print(imblearn.__version__)

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import EditedNearestNeighbours 

oversample = SMOTE()
enn = EditedNearestNeighbours()
# label encode the target variable

y = LabelEncoder().fit_transform(y)

X, y = enn.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
    per = v / len(y) * 100

    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
#plot the distribution
plt.bar(counter.keys(), counter.values())
plt.show()


# ### Train Test Split divides the dataset into 70 percent for training and 30 percent for testing.

# In[544]:


x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# ### To build the model, import SVM from the sci-kit learn library

# In[546]:


from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train, y_train) 


# ### Grid search cross validation hyperparameter tuning will be used to improve our model's performance accuracy.

# In[547]:


# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(x_train, y_train)


# ### This will yield the most effective parameters for our svm classifier.

# In[548]:


# print best parameter after tuning
print(grid.best_params_)


# ### Explore the estimator 

# In[549]:


# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)


# ### Investigate the optimal score for the svm classifier's parameters.

# In[550]:


grid.best_score_


# ### Now, apply the above-mentioned parameters for the svm classifier

# In[551]:


clf = SVC(C=100, gamma=0.1, kernel= 'rbf')
clf.fit(x_train, y_train) 


# ### Let us now apply the predict to our test set.

# In[552]:


y_pred = clf.predict(x_test)


# ### We can use cross validation to further analyse the model's performance on its test set.

# In[553]:


print(classification_report(y_test, y_pred))

print('Accuracy of Decision Tree classifier on the training set: {:.2f}'.format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on the test set: {:.2f}'.format(clf.score(x_test, y_test)))

#Decision Trees are very prone to overfitting as shown in the scores

score = cross_val_score(clf, x_train, y_train, cv=5) 
print('Cross-validation score: ',score)
print('Cross-validation mean score: ',score.mean())


# ### Let's build a function that will return the results of various metrics.
# 

# In[554]:


def summarize_classification(y_test,y_pred,avg_method='weighted'):
    acc = accuracy_score(y_test, y_pred,normalize=True)
    num_acc = accuracy_score(y_test, y_pred,normalize=False)
    f1= f1_score(y_test, y_pred, average=avg_method)
    prec = precision_score(y_test, y_pred, average=avg_method)
    recall = recall_score(y_test, y_pred, average=avg_method)
    jaccard = jaccard_score(y_test, y_pred, average=avg_method)
    
    print("Length of testing data: ", len(y_test))
    print("accuracy_count : " , num_acc)
    print("accuracy_score : " , acc)
    print("f1_score : " , f1)
    print("precision_score : " , prec)
    print("recall_score : ", recall)
    print("jaccard_score : ", jaccard)


# ### Using the different classification metrics, summarise the model's results.

# In[555]:


summarize_classification(y_test, y_pred)


# ### Let us now compare how well our model predicts on unseen data.

# In[616]:


#Accuracy
pred_results = pd.DataFrame({'y_test': pd.Series(y_test),
                             'y_pred': pd.Series(y_pred)})

pred_results.sample(10)


# ### To further evaluate our performance findings, let's build a confusion matrix that describes the false positive, true negative, true positive, and false negative.

# In[557]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[558]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[559]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_pred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False(0)','True(1)'],normalize= False,  title='Confusion matrix')


# # Logistic Regression Classifier

# In[560]:


import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


# ### Load the travel insurance dataset

# In[561]:


datalr=pd.read_csv("TravelInsurance_Classification\TravelInsurance.csv")
datalr.head(7)


# ### Drop the column 'F15,' which contains 750 missing values.

# In[562]:


datalr.drop(['F15'], axis=1, inplace=True)
datalr.head()


# ### Pass all columns except 'class' to the variable X.

# In[563]:


datalr=data[['F1','F2','F3', 'F4', 'F5', 'F6', 'F7','F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']]
X= datalr
X.head()


# ### Use the variable y to store the 'class' value, also known as the target variable that we will be predicting.

# In[564]:


datalr=data["Class"] 
y= datalr
y


# ### Normalize the dataset

# In[565]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### Over-sampling and under-sampling on unbalanced data

# In[566]:


import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE


# In[567]:


print(imblearn.__version__)

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import EditedNearestNeighbours 

oversample = SMOTE()
enn = EditedNearestNeighbours()
# label encode the target variable

y = LabelEncoder().fit_transform(y)

X, y = enn.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
    per = v / len(y) * 100

    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
#plot the distribution
plt.bar(counter.keys(), counter.values())
plt.show()


# ### Train Test Split divides the dataset into 80 percent for training and 20 percent for testing.

# In[568]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# ### To build the model, import Logistic regression from the sci-kit learn library

# In[569]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression().fit(x_train,y_train)
LR


# ### Grid search cross validation hyperparameter tuning will be used to improve our model's performance accuracy.

# In[570]:


model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]


# In[571]:


# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_train, y_train)


# ### This will yield the most effective parameters for our Logistic Regression classifier.

# In[572]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# ### Now, apply the above-mentioned parameters for the Logistic Regression classifier

# In[573]:


model=LogisticRegression(C=100, penalty='l2', solver='newton-cg')
model=model.fit(x_train,y_train)


# ### Let us now apply the predict to our test set.

# In[574]:


y_pred = model.predict(x_test)


# ### We can use cross validation to further analyse the model's performance on its test set.

# In[575]:


print(classification_report(y_test, y_pred))

print('Accuracy of Decision Tree classifier on the training set: {:.2f}'.format(model.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on the test set: {:.2f}'.format(model.score(x_test, y_test)))

#Decision Trees are very prone to overfitting as shown in the scores

score = cross_val_score(model, x_train, y_train, cv=5) 
print('Cross-validation score: ',score)
print('Cross-validation mean score: {:.2f}',score.mean())


# ### Let's build a function that will return the results of various metrics.

# In[576]:


def summarize_classification(y_test,y_pred,avg_method='weighted'):
    acc = accuracy_score(y_test, y_pred,normalize=True)
    num_acc = accuracy_score(y_test, y_pred,normalize=False)
    f1= f1_score(y_test, y_pred, average=avg_method)
    prec = precision_score(y_test, y_pred, average=avg_method)
    recall = recall_score(y_test, y_pred, average=avg_method)
    jaccard = jaccard_score(y_test, y_pred, average=avg_method)
    
    print("Length of testing data: ", len(y_test))
    print("accuracy_count : " , num_acc)
    print("accuracy_score : " , acc)
    print("f1_score : " , f1)
    print("precision_score : " , prec)
    print("recall_score : ", recall)
    print("jaccard_score : ", jaccard)


# ### Using the different classification metrics, summarise the model's results.

# In[577]:


summarize_classification(y_test, y_pred)


# ### Let us now compare how well our model predicts on unseen data.

# In[617]:


#Accuracy
pred_results = pd.DataFrame({'y_test': pd.Series(y_test),
                             'y_pred': pd.Series(y_pred)})

pred_results.sample(10)


# ### To further evaluate our performance findings, let's build a confusion matrix that describes the false positive, true negative, true positive, and false negative.

# In[579]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def ConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, y_pred, labels=[1,0]))


# In[580]:


cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_pred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['TRUE(1)','FALSE(0)'],normalize= False,  title='Confusion matrix')


# # Random Forest Classifier

# In[584]:


from sklearn.ensemble import RandomForestClassifier


# ### Load the travel insurance dataset

# In[585]:


datarand=pd.read_csv("TravelInsurance_Classification\TravelInsurance.csv")
datarand.head(7)


# ### Drop the column 'F15,' which contains 750 missing values.

# In[586]:


datarand.drop(['F15'], axis=1, inplace=True)
datarand.head()


# ### Pass all columns except 'class' to the variable X.

# In[587]:


datarand=data[['F1','F2','F3', 'F4', 'F5', 'F6', 'F7','F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']]
X= datarand
X.head()


# ### Use the variable y to store the 'class' value, also known as the target variable that we will be predicting.

# In[588]:


datalr=data["Class"] 
y= datalr
y


# ### Standardize the dataset

# In[589]:


X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### Over-sampling and under-sampling on unbalanced data

# In[590]:


import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE


# In[591]:


print(imblearn.__version__)

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import EditedNearestNeighbours 

oversample = SMOTE()
enn = EditedNearestNeighbours()
# label encode the target variable

y = LabelEncoder().fit_transform(y)

X, y = enn.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
    per = v / len(y) * 100

    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
#plot the distribution
plt.bar(counter.keys(), counter.values())
plt.show()


# ### Train Test Split divides the dataset into 80 percent for training and 20 percent for testing.

# In[592]:


x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.2)


# ### To build the model, use the Random Forest classifier.

# In[593]:


rf=RandomForestClassifier().fit(x_train,y_train)
rf


# ### Grid search cross validation hyperparameter tuning will be used to improve our model's performance accuracy.

# In[594]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}


# In[595]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[596]:


grid_search.fit(x_train,y_train)


# ### This will yield the most effective parameters for our random forest classifier.

# In[597]:


grid_search.best_params_


# ### Explore the estimator

# In[598]:


grid_search.best_estimator_


# ### Investigate the optimal score for the random forest classifier's parameters.

# In[599]:


grid_search.best_score_


# ### Now, apply the above-mentioned parameters for the random forest classifier

# In[600]:


rf=RandomForestClassifier(max_depth=80, max_features=3, 
                          min_samples_leaf=3, min_samples_split=10,
                          n_estimators=100).fit(x_train,y_train)
rf


# ### Let us now apply the predict to our test set. 

# In[601]:


y_pred = rf.predict(x_test)


# ### We can use cross validation to further analyse the model's performance on its test set.

# In[602]:


print(classification_report(y_test, y_pred))

print('Accuracy of Decision Tree classifier on the training set: {:.2f}'.format(rf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on the test set: {:.2f}'.format(rf.score(x_test, y_test)))

#Decision Trees are very prone to overfitting as shown in the scores

score = cross_val_score(rf, x_train, y_train, cv=5) 
print('Cross-validation score: ',score)
print('Cross-validation mean score: ',score.mean())


# ### Let's build a function that will return the results of various metrics.

# In[603]:


def summarize_classification(y_test,y_pred,avg_method='weighted'):
    acc = accuracy_score(y_test, y_pred,normalize=True)
    num_acc = accuracy_score(y_test, y_pred,normalize=False)
    f1= f1_score(y_test, y_pred, average=avg_method)
    prec = precision_score(y_test, y_pred, average=avg_method)
    recall = recall_score(y_test, y_pred, average=avg_method)
    jaccard = jaccard_score(y_test, y_pred, average=avg_method)
    
    print("Length of testing data: ", len(y_test))
    print("accuracy_count : " , num_acc)
    print("accuracy_score : " , acc)
    print("f1_score : " , f1)
    print("precision_score : " , prec)
    print("recall_score : ", recall)
    print("jaccard_score : ", jaccard)


# ### Using the different classification metrics, summarise the model's results

# In[604]:


summarize_classification(y_test, y_pred)


# ### Let us now compare how well our model predicts on unseen data.

# In[618]:


#Accuracy
pred_results = pd.DataFrame({'y_test': pd.Series(y_test),
                             'y_pred': pd.Series(y_pred)})

pred_results.sample(10)


# # Prediction on a hold-out test set using (Support Vector Machine)
# We would use the SVM classifier to estimate on our hold-out test set since it has generated 92 percent of the model's accuracy.

# ### Load the travel insurance dataset

# In[605]:


datatest=pd.read_csv("TravelInsurance_Classification\TravelInsurance_TestSet.csv")
datatest.head(7)


# ### Pass all columns except 'class' to the variable X.

# In[606]:


X=datatest[['F1','F2','F3', 'F4', 'F5', 'F6', 'F7','F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']].values
X[0:5]


# ### Use the variable y to store the 'class' value, also known as the target variable that we will be predicting.

# In[607]:


y=datatest[["Class"]] 
y[0:5]


# ### Standardize the dataset

# In[608]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### Let us now apply the predict to our test set.

# In[609]:


y_pred = clf.predict(x_test)


# ### Prepare the model result

# In[612]:


s = pd.Series(y_pred)


# ### Save the model result to file

# In[613]:


s.to_csv("TravelInsurance_Classification\TravelInsurance_result.csv", index=False)


# ### Summary of the model perfomance 

# In[614]:


result=[{ 'Accuracy Score':0.78,
         'F1 Score': 0.77,
         'Precision Score': 0.77,
         'Recall Score': 0.78,
         'Jaccard Score': 0.64},
        {'Accuracy Score':0.92,
         'F1 Score': 0.92,
         'Precision Score': 0.92,
         'Recall Score': 0.92,
         'Jaccard Score': 0.86},
        {'Accuracy Score': 0.89,
         'F1 Score': 0.89,
         'Precision Score': 0.89,
         'Recall Score': 0.89,
         'Jaccard Score': 0.81},
        {'Accuracy Score': 0.84,
         'F1 Score': 0.83,
         'Precision Score': 0.85,
         'Recall Score': 0.84,
         'Jaccard Score': 0.72}]
df=pd.DataFrame(result, index=['Decision Tree', 'Support Vector Machine','Logistic Regression', 'Random Forest'])
df.head()


# In[ ]:




