import matplotlib.pyplot as plt

def generate_graphs(metricValues):
    metricTitles = ["Accuracy Score (AS)", "Precision Score (PS)", 
            "Recall Score (RS)", "F1 Score (F1)"]
    fileNames = ["AS", "PS", "RS", "F1"]

    models = [0] * (len(metricValues)//6)
    As = [0] * (len(metricValues)//6)
    Ps = [0] * (len(metricValues)//6)
    Rs = [0] * (len(metricValues)//6)
    f1 = [0] * (len(metricValues)//6)

    best_accuracy = 0
    best_estimator = 0

    for i in range(len(metricValues)//6):
        models[i] = metricValues[(i*(len(metricTitles) + 2)) + 1]
        As[i] = metricValues[(i*(len(metricTitles) + 2)) + 2]*100
        Ps[i] = metricValues[(i*(len(metricTitles) + 2)) + 3]*100
        Rs[i] = metricValues[(i*(len(metricTitles) + 2)) + 4]*100
        f1[i] = metricValues[(i*(len(metricTitles) + 2)) + 5]*100
        if As[i] > best_accuracy:
            best_accuracy = As[i]
            best_estimator = metricValues[i*(len(metricTitles) + 1)]

    metrics = [As, Ps, Rs, f1]

    print("The model with the best accuracy after training is %s with an accuracy of %.2f percent" %
         (best_estimator, best_accuracy))

    for i in range(len(metrics)):
        plt.bar(models, metrics[i], width = 0.5)
        plt.title(metricTitles[i])
        plt.xlabel("Classification Models")
        plt.ylabel("Score")
        plt.grid(visible=True, which='major', axis='y', linestyle='--')
        plt.savefig(f"Graphs/{fileNames[i]}.png")
        plt.clf()
