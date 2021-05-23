
## Adressing Bias in Police Data

The structure of the model is described in two phases. The first step is to successfully develop an AI model that aids in predictive policing. The model should be able to identify the suspect and the type of crime that is likely to be committed based on the data points on which it was trained. This will then prepare the police and assist them in deciding on a plan for using force. The second step is to investigate for bias in the model and use the various metrics provided by IBM AIF360 toolkit to minimise any bias.

FairnessInML_2021 jupyter notebook is where I have all my codes implemented. I have also included the python executable code.

The metropolitan police dataset is 74mb too large for GitHub, but I have included a link to download it inside the codes.

* https://data.london.gov.uk/dataset/use-of-force
* Kindly click on the link provided above to access the dataset.

### Problem and Motivation

The aim of this project is to provide an AI model for bias identification and mitigation in policing data. Machine learning fairness is a subject of machine learning that has recently gained a lot of interest. Furthermore, it is a field that aids in the detection of bias in data as well as ensuring that the model’s inaccuracies do not contribute to models that treat humans unfavorably based on characteristics such as race, gender, religion, tribes, disabilities, sexual orientation, political opinion, and so on.

In general, police have played an important role in protecting citizens’ life and property in certain parts of the world. They have also aided in crime reduction. However, there have been a greater number of incidents involving police use of force against black citizens than against the people of color. The recent high-profile reports of unarmed black men being fatally shot by police officers highlight the issue of racism in police use of force. According to reports, police violence and brutality against black people in the United States is on the rise. George Floyd’s death inspired a vast number of protesters and the ”Black Lives Matter” movement. In May 2020, an unarmed African American was killed by the police during an arrest.

In Nigeria, several youths were murdered by police after being accused of cyber theft. The excessive use of force by Nigerian police has claimed many lives simply because they mistook many promising young adults for fraudsters without sufficient proof. Every youth self-image and lifestyle has become overly scrutinized, and this is often the justification for police use of force.

### Solution 

The structure of the model is described in two phases. The first step is to successfully develop an AI model that aids in predictive policing. The model should be able to identify the suspect and the type of crime that is likely to be committed based on the data points on which it was trained. This will then prepare the police and assist them in deciding on a plan for using force. The second step is to investigate for bias in the model and use the various metrics provided by IBM AIF360 toolkit to minimise any bias.

Incase you run into this error message "Sorry, something went wrong. Reload?" while attempting to launch the jupyter notebook from GitHub:

* Please visit "https://nbviewer.jupyter.org/"
* Copy the Url link of my jupyter notebook like this "https://github.com/danladishitu/My_Portfolio/blob/main/AddressingBiasInPoliceData/FairnessInML_2021.ipynb" and paste it into the nbviewer search box. Please feel free to download the notebook and use it however you see necessary.

Packages and the version that may be required to run the code:

* aif360
* scikit-learn 0.24.1
* matplotlib 3.3.4
* numpy 1.19.2
* pandas 1.2.3
