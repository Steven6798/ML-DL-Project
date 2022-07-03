import matplotlib.pyplot as plt

def generate_graphs(models_list):
	metric_titles = ["Accuracy Score (AS)", "Precision Score (PS)",
			"Recall Score (RS)", "F1 Score (F1)"]
	file_names = ["AS", "PS", "RS", "F1"]

	models = [0] * len(models_list)
	As = [0] * len(models_list)
	Ps = [0] * len(models_list)
	Rs = [0] * len(models_list)
	f1 = [0] * len(models_list)

	best_accuracy = 0
	best_estimator = 0

	for i in range(len(models_list)):
		As[i] = (models_list[i])[3]*100
		Ps[i] = (models_list[i])[4]*100
		Rs[i] = (models_list[i])[5]*100
		f1[i] = (models_list[i])[6]*100
		models[i] = (models_list[i])[7]
		if As[i] > best_accuracy:
			best_accuracy = As[i]
			best_estimator = (models_list[i])[0]

	metrics = [As, Ps, Rs, f1]

	print("The model with the best accuracy after training is %s with an accuracy of %.2f percent" %
		 (best_estimator, best_accuracy))

	for i in range(len(metrics)):
		plt.bar(models, metrics[i], width = 0.5)
		plt.title(metric_titles[i])
		plt.xlabel("Classification Models")
		plt.ylabel("Score")
		plt.grid(visible=True, which='major', axis='y', linestyle='--')
		plt.savefig(f"Graphs/{file_names[i]}.png")
		plt.clf()
