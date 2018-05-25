import numpy as np

# Perform 90/5/5 train/dev/test split
def perform_split(data):
	num_examples = data.shape[0]
	end_train = int(num_examples * 0.8)
	end_dev = int(num_examples * 0.9)
	print("There are a total of", num_examples, "datapoints.")

	train_set = np.copy(data[0:end_train])
	dev_set = np.copy(data[end_train:end_dev])
	test_set = np.copy(data[end_dev:])
	
	assert(train_set.shape[0] + dev_set.shape[0] + test_set.shape[0] == num_examples)

	# Save the arrays we've created so we don't have to do this again
	np.save("Train_Set", train_set)
	np.save("Dev_Set", dev_set)
	np.save("Test_Set", test_set)

def main():
	data = np.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1)
	np.random.shuffle(data)
	perform_split(data)

main()