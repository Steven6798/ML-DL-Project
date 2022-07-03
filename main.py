classification_type = "Multi-Class Classification"
#classification_type = "Natural Language Processing"

# Tuning the model based on the dataset
from hyperparameter_tuning import hyperparameter_tuning
models_dict = hyperparameter_tuning(classification_type)

# Calling the models
from Models.decision_tree_classification import decision_tree_classification
from Models.k_nearest_neighbors import k_nearest_neighbors
from Models.logistic_regression import logistic_regression
from Models.gaussian_naive_bayes import gaussian_naive_bayes
from Models.random_forest_classification import random_forest_classification
from Models.support_vector_machine import support_vector_machine
from Models.gaussian_process_classification import gaussian_process_classification
from Models.ridge_classification import ridge_classification
from Models.stochastic_gradient_descent_classification import stochastic_gradient_descent_classification
from Models.extra_tree_classification import extra_tree_classification
from Models.bernoulli_naive_bayes import bernoulli_naive_bayes
metricValues = decision_tree_classification(models_dict["Decision Tree Classifier"], classification_type)
metricValues.extend(k_nearest_neighbors(models_dict["K Nearest Neighbors Classifier"], classification_type))
metricValues.extend(logistic_regression(models_dict["Logistic Regression"], classification_type))
metricValues.extend(gaussian_naive_bayes(models_dict["Gaussian Naive Bayes"], classification_type))
metricValues.extend(random_forest_classification(models_dict["Random Forest Classifier"], classification_type))
metricValues.extend(support_vector_machine(models_dict["Support Vector Machine"], classification_type))
metricValues.extend(gaussian_process_classification(models_dict["Gaussian Process Classifier"], classification_type))
metricValues.extend(ridge_classification(models_dict["Ridge Classifier"], classification_type))
metricValues.extend(stochastic_gradient_descent_classification(models_dict["Stochastic Gradient Descent Classifier"], classification_type))
metricValues.extend(extra_tree_classification(models_dict["Extra Tree Classifier"], classification_type))
metricValues.extend(bernoulli_naive_bayes(models_dict["Bernoulli Naive Bayes"], classification_type))

# Generating the graphs
from graph_generator import generate_graphs
generate_graphs(metricValues)