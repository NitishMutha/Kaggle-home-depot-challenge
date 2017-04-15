import numpy as np
import pandas as pd
import sys
import time
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# files
start_time = time.time()
preprocessed_features = 'vectorised_features.csv'
y_test_file = 'y_test.csv'
y_test_file = 'solution.csv'

# settings
preprocess = False
feature_selection = True
kaggle = False
hyperparam_opt = False

#---------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------
def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

def print_time():
	print("--- %s seconds ---" % (time.time() - start_time))

def get_y_test():
	y_test_data = pd.read_csv(y_test_file)
	y_test = y_test_data['relevance'].values
	usage = y_test_data['Usage'].values
	private_indices = np.where(usage=='Private')[0]
	return y_test, private_indices

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def rand_search(model_dict, print_report):

	model, model_type = (0,0)
	for key in model_dict:
		model_type = key
	
	model = model_dict[model_type]

	# specify parameters and distributions to sample from
	if model_type == 'rf':
		param_dist = {"max_depth": [5,10,20,50],
		              "max_features": [2,3,4],
		              "min_samples_split": [5,10,15],
		              "min_samples_leaf": [5,10,15],
		              "bootstrap": [True, False]}
	elif model_type == 'clf':
		param_dist = {"max_samples": [5,10,20,50,70,90,100],
		              "max_features": [2,3,4]}
		              # "bootstrap": [True, False]}
		              # "bootstrap_features": [True, False]}
	elif model_type == 'bf':
		param_dist = {"loss": ['ls', 'lad'], 
					  "learning_rate": [0.5,0.1,0.01],
					  "max_depth": [5,10,20,50],
		              "max_features": [2,3,4],
		              "min_samples_split": [5,10,15],
		              "min_samples_leaf": [5,10,15],
		              "bootstrap": [True, False]}

	# run randomized search
	n_iter_search = 20
	random_search = RandomizedSearchCV(model, param_distributions=param_dist,
	                                   n_iter=n_iter_search)

	start = time.time()
	print(model_type)
	random_search.fit(X_train, y_train)
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
	      " parameter settings." % ((time.time() - start), n_iter_search))
	
	if (print_report):
		report(random_search.cv_results_)

	return random_search

#---------------------------------------------------------------------------------
# Read data
#---------------------------------------------------------------------------------
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('product_descriptions.csv')
print("data read...")

#---------------------------------------------------------------------------------
# Preprocessing / feature selection
#---------------------------------------------------------------------------------
stemmer = SnowballStemmer('english')
n_train = df_train.shape[0]
X_train, X_valid, X_test, y_train, y_valid, y_test = (0,0,0,0,0,0)

if preprocess:
	df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
	df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
	print("data merged...")

	df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
	df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
	df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
	df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
	df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']
	df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
	df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
	print("main processing done...")

	df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

	df_train = df_all.iloc[:n_train]
	df_test = df_all.iloc[n_train:]
	id_test = df_test['id']

	y_train = df_train['relevance'].values
	X_train = df_train.drop(['id','relevance'],axis=1).values
	X_test = df_test.drop(['id','relevance'],axis=1).values
	print("dropping done...")

	# save data
	np.savetxt("y_train.csv", y_train, delimiter=",")
	np.savetxt("X_train.csv", X_train, delimiter=",")
	np.savetxt("X_test.csv", X_test, delimiter=",")

elif feature_selection:
	df_data = pd.read_csv(preprocessed_features)
	print("feature selected data read...")
	y = df_data['relevance'].values
	X = df_data.drop(['id','product_uid','relevance'],axis=1).values

	X_train_all = X[0:n_train, :]
	n_train_new = int(0.75*n_train)
	n_valid = int(0.25*n_train)

	X_train = X_train_all[0:n_train_new, :]
	X_valid = X_train_all[n_train_new+1:n_train_new+n_valid, :]
	X_test = X[n_train:, :]
	y_train = y[0:n_train_new]
	y_valid = y[n_train_new+1:]

	y_test, private_indices = get_y_test()

else:
	X_train = np.genfromtxt('X_train.csv', delimiter=',')
	X_test = np.genfromtxt("X_test.csv", delimiter=',')
	y_train = np.genfromtxt("y_train.csv", delimiter=',')
	print("processed data read...")
	# print(y_train.shape)	# (74067,)
	# print(X_train.shape)	# (74067, 4)
	# print(X_test.shape)		# (166693, 4)
	n_train = int(0.75*X_train.shape[0])
	n_valid = int(0.25*X_train.shape[0])

	X_train = X_train[0:n_train, :]
	X_valid = X_train[n_train+1:n_train+n_valid, :]
	y_train = y_train[0:n_train]
	y_valid = y_train[n_train+1:]

	y_test, private_indices = get_y_test()

#---------------------------------------------------------------------------------
# Model
#---------------------------------------------------------------------------------
if not hyperparam_opt:
	rf = RandomForestRegressor(n_estimators=20, bootstrap=True, min_samples_leaf=15, min_samples_split=15, max_features=3, max_depth=10)
	clf = BaggingRegressor(rf, n_estimators=5, max_samples=0.1, random_state=0)
	bf = GradientBoostingRegressor(n_estimators=5, max_depth=6, random_state=0)

	rf.fit(X_train, y_train)
	print("random forest fitted...")

	clf.fit(X_train, y_train)
	print("bagging fitted...")

	bf.fit(X_train, y_train)
	print("boosting fitted...")
else:
	rf = RandomForestRegressor(n_estimators=20)
	clf = BaggingRegressor(rf, n_estimators=20)
	bf = GradientBoostingRegressor(n_estimators=20)#min_samples_split=2,learning_rate=0.01, loss='ls')

	all_models = [{'rf':rf}, {'clf':clf}, {'bf':bf}]
	all_results = []
	for model in all_models:
		all_results.append(rand_search(model, True))
	rf,clf,bf = all_results

y_pred_rf = rf.predict(X_test)
y_pred_clf = clf.predict(X_test)
y_pred_bf = bf.predict(X_test)
y_pred_en = (y_pred_rf + y_pred_clf + y_pred_bf) / 3

y_pred_train_rf = rf.predict(X_train)
y_pred_train_clf = clf.predict(X_train)
y_pred_train_bf = bf.predict(X_train)
y_pred_train_en = (y_pred_train_rf + y_pred_train_clf + y_pred_train_bf) / 3

#---------------------------------------------------------------------------------
# Results
#---------------------------------------------------------------------------------
id_test = df_test['id']

if not kaggle:
	y_test_p = y_test[private_indices]
	y_pred_clf_p = y_pred_clf[private_indices]
	y_pred_rf_p = y_pred_rf[private_indices]
	y_pred_bf_p = y_pred_bf[private_indices]
	y_pred_en_p = y_pred_en[private_indices]

	RMSE_train_rf = mean_squared_error(y_train, y_pred_train_rf)**0.5	
	RMSE_train_clf = mean_squared_error(y_train, y_pred_train_clf)**0.5
	RMSE_train_bf = mean_squared_error(y_train, y_pred_train_bf)**0.5
	RMSE_train_en = mean_squared_error(y_train, y_pred_train_en)**0.5
	
	RMSE_rf = mean_squared_error(y_test_p, y_pred_rf_p)**0.5	
	RMSE_clf = mean_squared_error(y_test_p, y_pred_clf_p)**0.5
	RMSE_bf = mean_squared_error(y_test_p, y_pred_bf_p)**0.5
	RMSE_en = mean_squared_error(y_test_p, y_pred_en_p)**0.5
	
	print("\n")
	print("RMSE train for bagging: ", RMSE_train_clf)
	print("RMSE train for random forest: ", RMSE_train_rf)
	print("RMSE train for boosting: ", RMSE_train_bf)
	print("RMSE train for ensemble: ", RMSE_train_en)
	print("\n")
	print("RMSE test for bagging: :", RMSE_clf)
	print("RMSE test for random forest: ", RMSE_rf)
	print("RMSE test for boosting: ", RMSE_bf)
	print("RMSE test for ensemble: ", RMSE_en)

pd.DataFrame({"id": id_test, "relevance": y_pred_rf}).to_csv('submission_rf.csv',index=False)
pd.DataFrame({"id": id_test, "relevance": y_pred_clf}).to_csv('submission_clf.csv',index=False)
pd.DataFrame({"id": id_test, "relevance": y_pred_bf}).to_csv('submission_bf.csv',index=False)


