import numpy as np

 
# Output shows that there are many more negative than positive examples
def check_labels(data):
	num_feats = data[0].shape[1]

	train_labels = data[0][:, num_feats - 1]
	dev_labels = data[1][:, num_feats - 1]
	test_labels = data[2][:, num_feats - 1]

	pos_train = np.where(train_labels > 0)[0]
	pos_dev = np.where(dev_labels > 0)[0]
	pos_test = np.where(test_labels > 0)[0]

	print("There are", pos_train.shape[0], "Positive examples in the training set and", train_labels.shape[0] - pos_train.shape[0], "negative examples.")
	print("There are", pos_dev.shape[0], "Positive examples in the dev set and", dev_labels.shape[0] - pos_dev.shape[0], "negative examples.")
	print("There are", pos_test.shape[0], "Positive examples in the test set and", test_labels.shape[0] - pos_test.shape[0], "negative examples.")


def main():
	data = [np.load("Train_Set.npy"), np.load("Dev_Set.npy"), np.load("Test_Set.npy")]
	print("There are", data[0].shape[0] + data[1].shape[0] + data[2].shape[0], "examples in total.")
	print("There are", data[0].shape[1],"features in total.")
	check_labels(data)


if __name__ == '__main__':
	main()