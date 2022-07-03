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

class Main():
	models_dict = {
		"Decision Tree Classifier": ["Decision Tree Classifier", DecisionTreeClassifier(), 0],
		"Gaussian Process Classifier": ["Gaussian Process Classifier", GaussianProcessClassifier(), 0],
		"K Nearest Neighbors Classifier": ["K Nearest Neighbors Classifier", KNeighborsClassifier(), 0],
		"Support Vector Machine": ["Support Vector Machine", SVC(), 0],
		"Logistic Regression": ["Logistic Regression", LogisticRegression(), 0],
		"Gaussian Naive Bayes": ["Gaussian Naive Bayes", GaussianNB(), 0],
		"Random Forest Classifier": ["Random Forest Classifier", RandomForestClassifier(), 0],
		"Ridge Classifier": ["Ridge Classifier", RidgeClassifier(), 0],
		"Stochastic Gradient Descent Classifier": ["Stochastic Gradient Descent Classifier", SGDClassifier(), 0],
		"Extra Tree Classifier": ["Extra Tree Classifier", ExtraTreeClassifier(), 0],
		"Bernoulli Naive Bayes": ["Bernoulli Naive Bayes", BernoulliNB(), 0]
	}

	models_list = [
		["Decision Tree Classifier", DecisionTreeClassifier(), 0, 0, 0, 0, 0, "DTC"],
		["Gaussian Process Classifier", GaussianProcessClassifier(), 0, 0, 0, 0, 0, "KNNC"],
		["K Nearest Neighbors Classifier", KNeighborsClassifier(), 0, 0, 0, 0, 0, "LR"],
		["Support Vector Machine", SVC(), 0, 0, 0, 0, 0, "GNB"],
		["Logistic Regression", LogisticRegression(), 0, 0, 0, 0, 0, "RFC"],
		["Gaussian Naive Bayes", GaussianNB(), 0, 0, 0, 0, 0, "SVM"],
		["Random Forest Classifier", RandomForestClassifier(), 0, 0, 0, 0, 0, "GPC"],
		["Ridge Classifier", RidgeClassifier(), 0, 0, 0, 0, 0, "RC"],
		["Stochastic Gradient Descent Classifier", SGDClassifier(), 0, 0, 0, 0, 0, "SGDC"],
		["Extra Tree Classifier", ExtraTreeClassifier(), 0, 0, 0, 0, 0, "ETC"],
		["Bernoulli Naive Bayes", BernoulliNB(), 0, 0, 0, 0, 0, "BNB"],
	]

	classification_type = "Multi-Class Classification"
	#classification_type = "Natural Language Processing"

	# Tuning the model based on the dataset
	from hyperparameter_tuning import HyperparameterTuning
	hyperparameter_tuning_obj = HyperparameterTuning(classification_type, models_list)
	models_list = hyperparameter_tuning_obj.main()

	from ModelTypes.classification import ClassificationModels
	machine_learning_models_obj = ClassificationModels(classification_type, models_list)
	models_list = machine_learning_models_obj.main()

	from graph_generator import generate_graphs
	generate_graphs(models_list)
	print("Done!")