# IRDM Project 2017 - Group 30, at UCL  
## Option 3 - Home Depot Kaggle Challenge
#### Nitish Mutha, Russel Daries, Alaister Moull, Rafel Faruq  
---  

## Introduction  
The Home Depot Kaggle Challenge is a task of ranking products sold by Home Depot in order of relevance to the user search queries. 
The goal of is to develop a model which can accurately rank the products using Learning to Rank algorithms and provide critical
analysis with optimization of models by tuning the hyper-parameters and feature engineering.  

## Project tasks breakdown 
- Literature review.  
- Combining the input data from various files.  
- Pre-processing and cleaning of the data.  
- Feature engineering and selection.  
- Implement Learning to Rank Models from RankLib.  
- Implement Tree based methods as a comparison for RankLib methods.  
- Optimization of various implemented methods.  
- Ensemble Tree based methods.  

## Project Pipeline  
![Alt](/images/process.jpg "pipeline")  

## Codebase  
### 1. Preprocessing and Feature Engineering 

**Code:**  
`src/feature_extractor.py`: Main code to load dataset, combine, clean, preprocess and feature selection.  
`src/misc.py`: Helper funtions for feature_extrator.py.  

**Command:**  
Open terminal and execute   
`python feature_extractor.py`  

**Output:**  
`vectorised_features_final.csv`: Extracted features from the dataset. Can be used by RankLib as input   
`features_alldata.csv`: Dump of all processed data from the feature_extractor. To be used for Tree based methods.  
`feature_train_ranklib.txt`: A ranklib format train feature file. To be used to run training on RankLib methods.  
`feature_test_ranklib.txt`: A ranklib format test feature file. To be used to run tests on RankLib methods.  

__Note: Due to github file upload restrictions, we cannot upload any file above 100Mb. Hence before executing the code please extract the compressed `productDescription.zip` and `attributes.zip` under `data` folder which are required by the `feature_extractor.py` to run without any errors.__

### 2. RankLib Methods Implementation

## Software requirements  
* Python 3.5 and above
* Numpy
* Pandas
* sklearn
* request
* nltk
* matplotlib
* seaborn
