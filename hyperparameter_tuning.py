# Importing the libraries
import numpy as np
import pandas as pd

def hyperparameter_tuning(classification_type):
  # Importing the dataset
  if classification_type == "Natural Language Processing":
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

  # Initializing the models
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

  models_dict = {
    "Decision Tree Classifier": [DecisionTreeClassifier(), 0],
    "Gaussian Process Classifier": [GaussianProcessClassifier(), 0],
    "K Nearest Neighbors Classifier": [KNeighborsClassifier(), 0],
    "Support Vector Machine": [SVC(), 0],
    "Logistic Regression": [LogisticRegression(), 0],
    "Gaussian Naive Bayes": [GaussianNB(), 0],
    "Random Forest Classifier": [RandomForestClassifier(), 0],
    "Ridge Classifier": [RidgeClassifier(), 0],
    "Stochastic Gradient Descent Classifier": [SGDClassifier(), 0],
    "Extra Tree Classifier": [ExtraTreeClassifier(), 0],
    "Bernoulli Naive Bayes": [BernoulliNB(), 0]
  }
  
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
  i = 0
  best_accuracy = 0
  best_estimator = 0
  for models in models_dict.keys():
    grid_search = RandomizedSearchCV(estimator = (models_dict[models])[0],
                              param_distributions = parameters_dict[i],
                              scoring = 'accuracy',
                              n_iter = 100,
                              cv = 10,
                              n_jobs = -1)
    grid_search.fit(X_train, y_train)
    accuracy = grid_search.best_score_*100
    (models_dict[models])[1] = accuracy
    #best_parameters = grid_search.best_params_
    estimator = grid_search.best_estimator_
    (models_dict[models])[0] = estimator
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_estimator = models
    #print(models)
    #print("Best Accuracy: {:.2f} %".format(accuracy))
    #print("Best Parameters:", best_parameters)
    #print("Best Estimator:", estimator)
    i = i + 1
  print("The model with the best accuracy before training is %s with an accuracy of %.2f percent" %
       (best_estimator, best_accuracy))
  
  return models_dict

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