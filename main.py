from margin_perceptron import MarginPerceptron
from gaussian_kernel import GaussianPerceptron


def get_file_data_to_array(file_path):
    data = {'n': 0, 'd': 0, 'inputs': [], 'targets': []}
    with open(file_path, 'r') as fin:
        data['n'], data['d'] = map(int, fin.readline().strip().split())
        for each in fin.readlines():
            each = list(map(int, each.strip().split()))
            data['inputs'].append(each[:-1])
            data['targets'].append(each[-1])
    return data


default_training_file = "train_set_0.txt"
default_testing_file = "test_set_0.txt"

print(
    "Default training file: %s\nInput 'yes' if using the deafult file to train, otherewise please input the path for training data"
    % (default_training_file))
training_file = input()
print(
    "Default testing file: %s\nInput 'yes' if using the deafult file to test, otherewise please input the path for testing data"
    % (default_testing_file))
testing_file = input()

if training_file == 'yes':
    training_file = default_training_file
if testing_file == 'yes':
    testing_file = default_testing_file

# get the file data to training set array and testing array
training_data = get_file_data_to_array(training_file)
testing_data = get_file_data_to_array(testing_file)

print(
    "Please choose method: \ninput '1' for Margin Perceptron\ninput '2' for Gaussian Kernel Perceptron\ninput '3' for Polynomial Kernel Perceptron ")
chosen_method = input()

print("training data size: ", len(training_data['inputs']))
print("testing data size: ", len(testing_data['inputs']))
if chosen_method == '1':
    model = MarginPerceptron(training_data)
    model.train()
    predicted_labels = model.predict(testing_data['inputs'])
    accuracy_score = model.accuracy_score(predicted_labels, testing_data['targets'])
    print("The accuracy of Margin Perceptron is : %s" %(accuracy_score))
elif chosen_method == '2':
    print("Please input the sigma value")
    training_data['sigma'] = float(input())
    print("Please input max iterations")
    training_data['max_iterations'] = float(input())
    model = GaussianPerceptron(training_data)
    model.train()
    predicted_labels = model.predict(testing_data['inputs'])
    accuracy_score = model.accuracy_score(predicted_labels, testing_data['targets'])
    print("The accuracy of Gaussian Kernel Perceptron is : %s" %(accuracy_score))
else:
    print("Please input 1 or 2 or 3")
