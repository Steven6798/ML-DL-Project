# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import BernoulliNB

class HyperparameterTuning:
	def __init__(self, classification_type, models_list):
		self.classification_type = classification_type
		self.models_list = models_list

	def data_preprocessing(self):
		# Importing the dataset
		if self.classification_type == "Natural Language Processing":
			dataset = pd.read_csv('Data.tsv', delimiter = '\t', quoting = 3)
		else:
			dataset = pd.read_csv('Data.csv')
		X = dataset.iloc[:, :-1].values
		y = dataset.iloc[:, -1].values

		# Splitting the dataset into the Training set and Test set
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

		# Feature Scaling
		from sklearn.preprocessing import StandardScaler
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)
		return [X_train, X_test, y_train, y_test]

	def training(self, X_train, y_train):
		# Applying Grid Search to find the best model and the best parameters
		from sklearn.model_selection import RandomizedSearchCV
		parameters_dict = [{'criterion':['gini', 'entropy'], 'max_depth':[range(2, 10), None], 'min_samples_split':range(2, 10), 'min_samples_leaf':range(1, 10), 'random_state':[0]},
						  {'random_state':[0], 'n_jobs':[-1]},
						  {'n_neighbors':range(1, 10), 'weights':['uniform', 'distance'], 'n_jobs':[-1]},
						  {'C':np.arange(0.25, 1.0, 0.25), 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'degree':range(3, 10), 'gamma':np.arange(0.1, 0.9, 0.1), 'random_state':[0], 'max_iter':[1000]},
						  {'penalty':['l1', 'l2', 'none'], 'class_weight':['balanced', None], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'max_iter':[1000], 'random_state':[0], 'n_jobs':[-1]},
						  {'var_smoothing':[1e-9]},
						  {'n_estimators':range(10, 100, 10), 'criterion':['gini', 'entropy'], 'max_depth':[range(2, 10), None], 'min_samples_split':range(2, 10), 'min_samples_leaf':range(1, 10), 'random_state':[0], 'n_jobs':[-1]},
						  {'class_weight':['balanced', None], 'random_state':[0]},
						  {'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty':['l1', 'l2'], 'alpha':[0.0001, 0.001, 0.01], 'class_weight':[None, 'balanced'], 'random_state':[0], 'n_jobs':[-1]},
						  {'criterion':['gini', 'entropy'], 'max_depth':[range(2, 10), None], 'min_samples_split':range(2, 10), 'min_samples_leaf':range(1, 10), 'random_state':[0], 'class_weight':['balanced', None]},
						  {'alpha':[1.0]}]
		best_accuracy = 0
		best_estimator = 0
		for i in range(len(self.models_list)):
			grid_search = RandomizedSearchCV(estimator = (self.models_list[i])[1],
									param_distributions = parameters_dict[i],
									scoring = 'accuracy',
									n_iter = 100,
									cv = 10,
									n_jobs = -1)
			grid_search.fit(X_train, y_train)
			estimator = grid_search.best_estimator_
			(self.models_list[i])[1] = estimator
			accuracy = grid_search.best_score_*100
			(self.models_list[i])[2] = accuracy
			#best_parameters = grid_search.best_params_
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_estimator = (self.models_list[i])[0]
			#print(models)
			#print("Best Accuracy: {:.2f} %".format(accuracy))
			#print("Best Parameters:", best_parameters)
			#print("Best Estimator:", estimator)
		print("The model with the best accuracy before training is %s with an accuracy of %.2f percent" %
			(best_estimator, best_accuracy))
		
		return self.models_list

		#with open("tunned_estimators.txt", "a+") as text_file:
		  #for estimators in models_dict.values():
			#print(f"{estimators}", file = text_file)

		# Applying k-Fold Cross Validation
		#from sklearn.model_selection import cross_val_score
		#for models in models_dict.keys():
			#accuracies = cross_val_score(estimator = models_dict[models], X = X_train, y = y_train, cv = 10)
			#print(models)
			#print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
			#print("Standard Deviation: {:.2f} % \n".format(accuracies.std()*100))

	def main(self):
		data_sets = self.data_preprocessing()
		trained_model = self.training(data_sets[0], data_sets[2])
		return trained_model