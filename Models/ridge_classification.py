# Ridge Classification

# Importing the libraries
import pandas as pd

def ridge_classification(estimator, classification_type):

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
    from sklearn.linear_model import RidgeClassifier
    classifier = estimator[0]
    classifier.fit(X_train, y_train)

    # Exporting the model for future use
    from joblib import dump
    dump(classifier, 'TrainedModels/ridge_classification.joblib') 

    y_pred = classifier.predict(X_test)

    # Metrics. For Accuracy, Precision, Recall, and f1 scores, closer to 1 means
    # better. Making the Confusion Matrix
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #, confusion_matrix
    accuracyScore = accuracy_score(y_test, y_pred)
    precisionScore = precision_score(y_test, y_pred, average='macro')
    recallScore = recall_score(y_test, y_pred, average='macro')
    f1Score = f1_score(y_test, y_pred, average='macro')
    #cm = confusion_matrix(y_test, y_pred)

    #with open("MetricValues.txt", "a+") as text_file:
    #    print(f"{accuracyScore:.4f}", f"\n{precisionScore:.4f}", 
    #        f"\n{recallScore:.4f}", f"\n{f1Score:.4f}", file=text_file)

    #print("Accuracy Score is %.4f" % accuracyScore)
    #print("Precision Score is %.4f" % precisionScore)
    #print("Recall Score is %.4f" % recallScore)
    #print("f1 Score is %.4f" % f1Score)
    #print(cm)
    return ["Ridge Classifier", "RC", accuracyScore, precisionScore, recallScore, f1Score]