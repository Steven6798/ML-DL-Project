# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def k_nearest_neighbors(estimator, classification_type):

    # Importing the dataset
    if classification_type == "Natural Language Processing":
        dataset = pd.read_csv('Data.tsv', delimiter = '\t', quoting = 3)
    else:
        dataset = pd.read_csv('Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

    if classification_type == "Natural Language Processing":
        # Cleaning the texts
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, 1000):
            review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            corpus.append(review)

        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    if classification_type == "Multi-Class Classification":
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
    from sklearn.neighbors import KNeighborsClassifier
    classifier = estimator[0]
    classifier.fit(X_train, y_train)

    # Exporting the model for future use
    from joblib import dump
    dump(classifier, 'TrainedModels/k_nearest_neighbors.joblib') 

    y_pred = classifier.predict(X_test)

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
    #    print(f"{accuracyScore:.4f}", f"\n{precisionScore:.4f}", 
    #        f"\n{recallScore:.4f}", f"\n{f1Score:.4f}", file=text_file)

    #print("Accuracy Score is %.4f" % accuracyScore)
    #print("Precision Score is %.4f" % precisionScore)
    #print("Recall Score is %.4f" % recallScore)
    #print("f1 Score is %.4f" % f1Score)
    #print(cm)
    return ["K Nearest Neighbors Classifier", "KNNC", accuracyScore, precisionScore, recallScore, f1Score]