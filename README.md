# My_Portfolio

### Travel Insurance (AI models)
Travel Insurance models are built with both classification and regression technique in machine learning.
The datasets used to create the model are available in the Dataset folder, which includes the hold-out test set and the travel insurance dataset.
I have included the jupyter notebook and the python executable code. Many of the codes I used to develop the models for the insurance company are included in the notebook.

There are two jupyter notebook:
* TravelInsuranceClassification.ipynb is where I used the classification algorithms to predict which insured was likely to file a claim.
* TravelInsuranceRegression.ipynb - In this notebook, I've used regression algorithms to predict not only whether or not the insured will file a claim, but also the value of the claim.

**Classification Problem** 

A travel insurance company would like to offer a discounted premium (for the same cover requested) to customers that are less likely to make a future claim.
The travel insurance company has provided us with historical data about their clients, but they are now interested in using machine learning to determine which customers are likely to file a claim on their future travel. 

**Solution**

To solve these problems, I used four different classification algorithms: random forest, logistic regression, decision tree, and support vector machine. Using hyperparameter tuning, these algorithms achieved an accuracy of more than 85% on the test set (grid search cross validation). This demonstrates that the model correctly predicted which customers are likely to file a claim in the future. I have evaluated the model performance using a hold-out test set, which are both clearly presented in the **TravelInsuranceClassification.ipynb**.

**Regression Problem**

Another company wants to employ us to develop a similar system for them, but unlike the first problem, they want us to predict not just whether or not the insured will file a claim, but also the amount of the claim. They have also provided us with historical data including features of each consumer as well as a numerical value representing the value of the claim (which may be zero).

**Solution**

To approach these problems, I have used three regression algorithms: multiple linear regression, polynomial regression and bayesian ridge regression. On the test set, some of these algorithms produced an accuracy of 70% for both multiple linear regression and bayesian ridge regression, while polynomial feature produced the highest accuracy of 85%. This demonstrates how accurately the model predicted the numerical value of each customer's claim. I have tested the model's performance with a hold-out test set, the results of which are clearly shown in the **TravelInsuranceRegression.ipynb**.

Incase you run into this error message **"Sorry, something went wrong. Reload?"**  while attempting to launch the jupyter notebook from GitHub:
* Please visit "https://nbviewer.jupyter.org/" 
* Copy the Url link of my jupyter notebook like this "https://github.com/danladishitu/My_Portfolio/blob/main/TravelInsurance/TravelInsuranceClassification.ipynb" and paste it into the nbviewer search box. Please feel free to download the notebook and use it however you see necessary.

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

This link will provide you with additional information about the packages.  https://scikit-learn.org/stable/supervised_learning.html

### Adressing Bias in Police Data
The structure of the model is described in two phases. The first step is to successfully develop an AI model that aids in predictive policing. The model should be able to identify the suspect and the type of crime that is likely to be committed based on the data points on which it was trained. This will then prepare the police and assist them in deciding on a plan for using force. The second step is to investigate for bias in the model and use the various metrics provided by IBM AIF360 toolkit to minimise any bias.

FairnessInML_2021 jupyter notebook is where I have all my codes implemented. I have also included the python executable code. 

The metropolitan police dataset is 74mb too large for GitHub, but I have included a link to download it inside the codes.
* https://data.london.gov.uk/dataset/use-of-force

### Problem and Motivation

The aim of this project is to provide an AI model for bias identification and mitigation in policing data. Machine learning fairness is a subject of machine learning that has recently gained a lot of interest. Furthermore, it is a field that aids in the detection of bias in data as well as ensuring that the model’s inaccuracies do not contribute to models that treat humans unfavorably based on characteristics such as race, gender, religion, tribes, disabilities, sexual orientation, political opinion, and so on.

In general, police have played an important role in protecting citizens’ life and property in certain parts of the world. They have also aided in crime reduction. However, there have been a greater number of incidents involving police use of force against black citizens than against the people of color. The recent high-profile reports of unarmed black men being fatally shot by police officers highlight the issue of racism in police use of force. According to reports, police violence and brutality against black people in the United States is on the rise. George Floyd’s death inspired a vast number of protesters and the ”Black Lives Matter” movement. In May 2020, an unarmed African American was killed by the police during an arrest.

In Nigeria, several youths were murdered by police after being accused of cyber theft. The excessive use of force by Nigerian police has claimed many lives simply because they mistook many promising young adults for fraudsters without sufficient proof. Every youth self-image and lifestyle has become overly scrutinized, and this is often the justification for police use of force.

### Solution 

The structure of the model is described in two phases. The first step is to successfully develop an AI model that aids in predictive policing. The model should be able to identify the suspect and the type of crime that is likely to be committed based on the data points on which it was trained. This will then prepare the police and assist them in deciding on a plan for using force. The second step is to investigate for bias in the model and use the various metrics provided by IBM AIF360 toolkit to minimise any bias.

Kindly click on the link provided above to access the dataset.

I have also added the python code.

Packages and the version that may be required to run the code: 
* aif360
* scikit-learn 0.24.1 
* matplotlib 3.3.4 
* numpy 1.19.2 
* pandas 1.2.3

### Topic Modelling in Text Analytics
As we all know that probabilistic topic models are a class of unsupervised machine learning models which help find the thematic patterns present in data. They have been widely applied to a range of domains from text to images. The basic topic model is called Latent Dirichlet Allocation (LDA) is being depicted by a graphical model in plate notation.

Our goal is to determine the number of topics (k) in a dataset. Similar to the algorithm that determines trends on Twitter. 

I have provided the link to download the dataset below:
* https://www.kaggle.com/danofer/starter-dbpedia-extraction

Packages that will be required to run the code:

* Pandas 
* Numpy
* Matplotlib
* sklearn.LatentDirichletAllocation
* Nltk
* String
* Re
* Gensim
* Simple_preprocess
* Spacy
* en_core_web_sm
* Corpora
* CoherenceModel
* LdaModel
* Tqdm

### Intelligent player scouting and talent acquisition for football managers using AI

This work will investigate the use of clustering techniques to identify possible groupings of players based on a blend of various attributes. This approach has resulted in numerous spectacular discoveries that would not have been possible using the manager's manual process.

### Problem statement

- Football managers are typically faced with the difficulty of choosing the right player selections in order to improve their team's performance. This is frequently reflected in their decisions about who to hire (for what position), who to sell, and when to sell.
- Individual players are also difficult to see as long-term investments due to a lack of skill transparency, which prevents management from knowing what has to be developed or optimised in order to maximise a player's worth.
- A lack of data to assess alternatives during player scouting, or a lack of competence to analyse the data itself, can lead to bias and favouritism in decision making.
- Due to the manager's incapacity to recognise overperforming and expensive players, the unsuitable players are commonly selected.
- Manager’s inadequate re-evaluation of players contributes to poor performance of the team; however, a player may perform better in a new position different from the one for which he is known.
- In recent years, football experts and club management have longed for a better means to estimate market value; although crowdsourcing sites such as Transfermarkt have proven useful in estimating market value but this may still be improved with the use of a data-driven approach.


### Proposed solution

- Model can potentially identify patterns that those certain players share in ways that would not normally have been considered by the team managers during their manual evaluation.
- Model would help managers better evaluate their teams to diagnose lack of skill diversity, identify under-valued and over-priced players, and potentially influence their transfer decisions.
- Model support team managers in the selection of players for their teams without human intervention in its decision making
- Models will be able to provide player’s re-evaluation (positions) and recommendations depending on the manager's preferences.
- Understanding of the model may assist the manager in building a suucessfull squad.
- Model can help the football business become more profitable and attract more investment through advertisements, sponsorships and brand ambassadorship.


The structure of the model is described:

The first phase is to effectively build a model capable of grouping players based on their similarity in traits. To do this, I have implemented K-means, K-means++ and DBSCAN algorithms to group players based on their individual abilities, as well as noise removal from the dataset. The model can potentially identify patterns those certain players share in ways that would not normally have been considered by the team managers during their manual evaluation.

The second phase entails building a classification model that will be capable of re-evaluating the player's position and transfer value based on the cluster labels provided by the clustering algorithm in the first phase. These classifiers will be able to predict what group a fresh set of players will belong to. Support Vector Machine and Random Forest are two ML algorithms that I used for this. This would also help managers diagnose lack of skill diversity, identify under-valued and over-priced players, and potentially influence their transfer decisions. The code to the second phase of this work can be found in **Re-evaluation_Classification.ipynb**

### Dataset
FIFA is a football simulation video game in the FIFA series released by Electronic Arts. The football dataset was obtained from Kaggle. The dataset contains 18,207 players, each with a distinct set of attributes. The link to the dataset is provided below. https://www.kaggle.com/karangadiya/fifa19

### Coding and Implementation
I have used Python and its diverse library to build the machine learning pipeline to tackle this classification problem. To improve readability, the code was written in a Jupyter notebook. Python executable code was also provided.
Packages that will be required to run the code:

- Pandas
- Numpy
- Seasborn
- Matplotlib
- Preprocessing/scale
- itertools/product
- OneHotEncoder

Clustering analysis:

- KMeans
- DBSCAN
- PCA
- NearestNeighbors
- Silhouette_score
- Metrics
- Counter
- Plotly/go

Classification analysis:

- RandomForestClassifier
- SVC
- imblearn/SMOTE
- train_test_split
- GridSearchCV
- cross_val_score
- accuracy_score
- precision_score
- recall_score
- f1_score
- jaccard_score
- confusion_matrix

This link will provide additional information about the packages. https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation


### Building a machine learning model to improve the effectiveness of an email advertising campaign.

### Problem statement

- The feasibility of using machine learning to improve the effectiveness of their email marketing campaigns.
- Over the year, the company conducts a series of email campaigns to promote their new items, and each email sent has a unique promotion code that allows the company to track if each email sent resulted in a purchase (this is called \conversion").
- They are currently employing the so-called batch-and-blast method (which entails sending all emails to their whole database), but they are concerned that this would result in high unsubscribe rates and reduce the overall efficacy of the campaigns.

### Proposed solution

- The model will help in the pursuit of targeted email marketing: for each campaign, emails will be delivered to just those clients who are most likely to purchase the advertised product.


I have used Python and its diverse library to build the machine learning pipeline to tackle this classification problem. To improve readability, the code was written in a Jupyter notebook.

The following are the four machine learning algorithms for classification problems:

- **Decision Tree**
- **Support Vector Machine** 
- **Naive Bayes** 
- **Random Forest**


