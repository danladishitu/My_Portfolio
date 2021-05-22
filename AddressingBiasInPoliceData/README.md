
## Adressing Bias in Police Data

The structure of the model is described in two phases. The first step is to successfully develop an AI model that aids in predictive policing. The model should be able to identify the suspect and the type of crime that is likely to be committed based on the data points on which it was trained. This will then prepare the police and assist them in deciding on a plan for using force. The second step is to investigate for bias in the model and use the various metrics provided by IBM AIF360 toolkit to minimise any bias.

FairnessInML_2021 jupyter notebook is where I have all my codes implemented. I have also included the python executable code.

The metropolitan police dataset is 74mb too large for GitHub, but I have included a link to download it inside the codes.

* https://data.london.gov.uk/dataset/use-of-force
* Kindly click on the link provided above to access the dataset.

Incase you run into this error message "Sorry, something went wrong. Reload?" while attempting to launch the jupyter notebook from GitHub:

* Please visit "https://nbviewer.jupyter.org/"
* Copy the Url link of my jupyter notebook like this "https://github.com/danladishitu/My_Portfolio/blob/main/AddressingBiasInPoliceData/FairnessInML_2021.ipynb" and paste it into the nbviewer search box. Please feel free to download the notebook and use it however you see necessary.

Packages and the version that may be required to run the code:

* aif360
* scikit-learn 0.24.1
* matplotlib 3.3.4
* numpy 1.19.2
* pandas 1.2.3
