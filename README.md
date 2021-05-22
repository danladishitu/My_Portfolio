# My_Portfolio

### Travel Insurance (AI models)
Travel Insurance models are built with both classification and regression technique in machine learning.
The datasets used to create the model are available in the Dataset folder, which includes the hold-out test set and the travel insurance dataset.
I have included the jupyter notebook and the python executable code. Many of the codes I used to develop the models for the insurance company are included in the notebook.

There are two jupyter notebook:
* TravelInsuranceClassification.ipynb is where I used the classification algorithms to predict which insured was likely to file a claim.
* TravelInsuranceRegression.ipynb - In this notebook, I've used regression algorithms to predict not only whether or not the insured will file a claim, but also the value of the claim.

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

### Intelligent player scouting and talent acquisition for football managers

**Can clustering algorithm improve the player selection of a team?**

**Can we use machine learning to support team managers in the selection of players for their teams without any form of bias and favouritism from the decision making?**

**This work will investigate the use of clustering techniques to identify possible groupings of players based on a blend of various attributes.**

**Models can then potentially identify patterns that certain players share in ways that would not normally have been considered by the team managers during their manual evaluation. This would help managers diagnose lack of skill diversity, identify under-priced and over-priced players, and potentially influence their transfer decisions.**

**Models should be able to offer recommendation on players based on the manager's preferences.**



