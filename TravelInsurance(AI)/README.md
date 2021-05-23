

## Travel Insurance (AI models)
Travel Insurance models are built with both classification and regression technique in machine learning. The datasets used to create the model are available in the Dataset folder, which includes the hold-out test set and the travel insurance dataset. I have included the jupyter notebook and the python executable code. Many of the codes I used to develop the models for the insurance company are included in the notebook.

There are two jupyter notebook:

* **TravelInsuranceClassification.ipynb** is where I used the classification algorithms to predict which insured was likely to file a claim.
* **TravelInsuranceRegression.ipynb** - In this notebook, I've used regression algorithms to predict not only whether or not the insured will file a claim, but also the value of the claim.
Incase you run into this error message "Sorry, something went wrong. Reload?" while attempting to launch the jupyter notebook from GitHub:

* Please visit "https://nbviewer.jupyter.org/"
* Copy the Url link of my jupyter notebook like this "https://github.com/danladishitu/My_Portfolio/blob/main/TravelInsurance/TravelInsuranceClassification.ipynb" and paste it into the nbviewer search box. Please feel free to download the notebook and use it however you see necessary.

Problem 
A travel insurance company would like to offer a discounted premium (for the same cover requested) to customers that are less likely to make a future claim.
The travel insurance company has provided us with historical data about their clients, but they are now interested in using machine learning to determine which customers are likely to file a claim on their future travel. 

Solution
To solve these problems, I used four different classification algorithms: random forest, logistic regression, decision tree, and support vector machine. On the test set, these algorithms achieved an accuracy of more than 85%. This demonstrates that the model correctly predicted which customers are likely to file a claim in the future. I have evaluated the model performance using a hold-out test range, which are both clearly presented in the jupyter notebook.

The following packages must be imported from the sci-kit learn libraries:

* Pandas
* Numpy
* Matplotlib
* LogisticRegression
* RandomForest
* SVM
* Decision Tree
* LinearRegression
* PolynomialFeatures
* BayesianRidge

This link will provide you with additional information about the packages. https://scikit-learn.org/stable/supervised_learning.html
