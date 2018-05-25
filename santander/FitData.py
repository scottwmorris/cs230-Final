import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
	training_data = np.load("Train_Set.npy")
	dev_data = np.load("Dev_Set.npy")

	# Separate data into features and labels
	num_feats = training_data.shape[1]
	training_feats = training_data[:, 0:num_feats-1]
	training_labels = training_data[:, num_feats-1]
	dev_feats = dev_data[:, 0:num_feats-1]
	dev_labels = dev_data[:, num_feats-1]

	run_log_reg(training_feats, training_labels, dev_feats, dev_labels)


def run_log_reg(training_feats, training_labels, dev_feats, dev_labels):
	log_reg = LogisticRegression().fit(training_feats, training_labels)
	print("Accuracy on the training set is", log_reg.score(training_feats, training_labels))
	print("Accuracy on the dev set is", log_reg.score(dev_feats, dev_labels))

	# The probabilities predict on each example in the train and dev set are stored in 
	# train_set_probabilities and dev_set_probabilities respectively
	train_set_probabilities = log_reg.predict_proba(training_feats)
	dev_set_probabilities = log_reg.predict_proba(dev_feats)


main()


# Action items: look at precision and recall and what the logistic regression is predicting (all 0's?)