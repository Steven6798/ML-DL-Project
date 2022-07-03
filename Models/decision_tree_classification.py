# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def decision_tree_classification(estimator, classification_type):

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

    #print("Number of original features = %d" % X_train.shape[1])

    # Applying PCA
    from sklearn.decomposition import PCA
    pca = PCA(0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    #print("Number of new features after PCA = %d" % X_train.shape[1])

    #varianceRatios = pca.explained_variance_ratio_

    # Training the model on the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = estimator[0]
    classifier.fit(X_train, y_train)

    # Exporting the model for future use
    from joblib import dump
    dump(classifier, 'TrainedModels/decision_tree_classification.joblib') 

    y_pred = classifier.predict(X_test)

    # Visualizing the Test set results
    #from matplotlib.colors import ListedColormap
    #X_set, y_set = sc.inverse_transform(X_test), y_test
    #X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
    #                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
    #plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    #             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    #plt.xlim(X1.min(), X1.max())
    #plt.ylim(X2.min(), X2.max())
    #for i, j in enumerate(np.unique(y_set)):
    #    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    #plt.title('Random Forest Classification')
    #plt.xlabel('Age')
    #plt.ylabel('Estimated Salary')
    #plt.legend()
    #plt.show()

    # Making a single prediction
    #age = 30
    #salary = 87000
    #print("A customer with age =", age, "and salary = ", salary,
    #"will buy the car if result is = 1. Result = ",
    #classifier.predict(sc.transform([[age, salary]])))

    # Metrics. For Accuracy, Precision, Recall, and f1 scores, closer to 1 means
    # better. Making the Confusion Matrix
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #, confusion_matrix
    accuracyScore = accuracy_score(y_test, y_pred)
    precisionScore = precision_score(y_test, y_pred, average='macro')
    recallScore = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')
    #cm = confusion_matrix(y_test, y_pred)

    #with open("MetricValues.txt", "a+") as text_file:
        #print(f"{accuracyScore:.4f}", f"\n{precisionScore:.4f}", 
            #f"\n{recallScore:.4f}", f"\n{f1Score:.4f}", file=text_file)

    #print("Accuracy Score is %.4f" % accuracyScore)
    #print("Precision Score is %.4f" % precisionScore)
    #print("Recall Score is %.4f" % recallScore)
    #print("f1 Score is %.4f" % f1Score)
    #print(cm)
    return ["Decision Tree Classifier", "DTC", accuracyScore, precisionScore, recallScore, f1Score]