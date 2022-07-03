# Importing the libraries
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

class ClassificationModels:
	def __init__(self, classification_type, model_list):
		self.classification_type = classification_type
		self.model_list = model_list

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

		#print("Number of original features = %d" % X_train.shape[1])

		# Applying PCA
		from sklearn.decomposition import PCA
		pca = PCA(0.95)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)

		#print("Number of new features after PCA = %d" % X_train.shape[1])

		#varianceRatios = pca.explained_variance_ratio_
		return [X_train, X_test, y_train, y_test]

	def training(self, X_train, X_test, y_train, model):
		# Training the model on the Training set
		classifier = model[1]
		classifier.fit(X_train, y_train)

		y_pred = classifier.predict(X_test)

		# Making a single prediction
		#age = 30
		#salary = 87000
		#print("A customer with age =", age, "and salary = ", salary,
		#"will buy the car if result is = 1. Result = ",
		#classifier.predict(sc.transform([[age, salary]])))
		return classifier, y_pred

	def metrics(self, y_pred, y_test, model):
		# Metrics. For Accuracy, Precision, Recall, and f1 scores, closer to 1 means
		# better. Making the Confusion Matrix
		from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #, confusion_matrix
		#accuracyScore = accuracy_score(y_test, y_pred)
		#precisionScore = precision_score(y_test, y_pred, average='macro')
		#recallScore = recall_score(y_test, y_pred, average='macro')
		#f1Score = f1_score(y_test, y_pred, average='macro')
		model[3] = accuracy_score(y_test, y_pred)
		model[4] = precision_score(y_test, y_pred, average='macro')
		model[5] = recall_score(y_test, y_pred, average='macro')
		model[6] = f1_score(y_test, y_pred, average='macro')
		#cm = confusion_matrix(y_test, y_pred)

		#with open("MetricValues.txt", "a+") as text_file:
		#    print(f"{accuracyScore:.4f}", f"\n{precisionScore:.4f}", 
		#        f"\n{recallScore:.4f}", f"\n{f1Score:.4f}", file=text_file)

		#print("Accuracy Score is %.4f" % accuracyScore)
		#print("Precision Score is %.4f" % precisionScore)
		#print("Recall Score is %.4f" % recallScore)
		#print("f1 Score is %.4f" % f1Score)
		#print(cm)
		
		return model

	def export(self, classifier, model_list):
		# Exporting the model for future use
		from joblib import dump
		name_formatted = model_list[0].replace(' ', '_').lower()
		file_name = 'TrainedModels/' + name_formatted + '.joblib'
		dump(classifier, file_name)

	def main(self):
		data_sets = self.data_preprocessing()
		for i in range(len(self.model_list)):
			trained_model = self.training(data_sets[0], data_sets[1], data_sets[2], self.model_list[i])
			self.model_list[i] = self.metrics(trained_model[1], data_sets[3], self.model_list[i])
			self.export(trained_model[0], self.model_list[i])
		return self.model_list
		