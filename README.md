The Random Forest and Logistic Regression Classifier are implemented to classify from rock drill machine faults

# PHM-Classifiers

### Introduction

This repository consists of various implementations of modelling Classifiers algorithms to classify the PHM Rock-Drill dataset.

All the classifier modelling files are Jupyter Notebooks and require the 'feature_extracts/' dataset from the [PHM-Data-Preprocessing](https://github.com/Data-Pundits/PHM-Data-Preprocessing) repository.

### Software Installation

Please install the prerequisite Softwares and libraries in order to run the scripts in this repository successfully:

1. [Python 3](https://www.python.org/downloads/) or greater. (Required)
2. Python Libraries (Run the below *pip* commands in a **Terminal** or **Command-Prompt** window):

***numpy**: `pip install numpy==1.21.5`

***jupyter-notebook**: `pip install jupyter`

***scipy**: `pip install scipy==1.10.1`

***pandas**: `pip install pandas==1.5.3`

***scikit-learn**: `pip install scikit-learn==1.2.1`

***matplotlib**: `pip install matplotlib==3.7.0`

***seaborn**: `pip install seaborn==0.11.2`

***xgboost**: `pip install xgboost==1.7.4`

3. [VS Code](https://code.visualstudio.com/download) - A light-weight IDE (Optional)

### Datasource Configuration:

* The Jupyter Notebooks in this repository require the 'feature_extracts/' dataset to be available which can be generated using the bash script provided in the [PHM-Data-Preprocessing](https://github.com/Data-Pundits/PHM-Data-Preprocessing) repository.
* There is a ***datasource_config.py*** file which can be used to configure the location of the '*feature_extracts/*' dataset. The configured path will be dynamically used in all the Jupyter Notebooks of this repository for modelling the classifiers.

### How to Run the Jupyter Notebooks

> **Note:** Jupyter Notebook library must be installed on the system.

1. Follow the run instructions provided in the [PHM-Data-Preprocessing](https://github.com/Data-Pundits/PHM-Data-Preprocessing) repository in order to generate the '*feature_extracts/*' dataset folder.
2. Next, configure the source-path for the '*feature_extracts/*' folder in the '*datasource_config.py*' file.
3. Once installed, the Jupyter Explorer window can be opened by running the following command on a Terminal/Command Prompt window:

**`$ jupyter notebook`*

4. In the Jupyter Explorer window, navigate to the directory where this repository was cloned and open the desired Jupyter Notebook (.ipynb) file.
5. Use *Ctrl + Enter* (Windows) or *Command + Return* (MacOS) to run each cell of the notebook.

There are several other ways to open and run Jupyter Notebooks. A simple alternative woud be to use VS Code IDE which has a Jupyter Notebook extension that can be easily installed within VS Cod
