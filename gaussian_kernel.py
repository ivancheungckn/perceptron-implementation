import math


class GaussianPerceptron():
    def __init__(self, data):
        super(GaussianPerceptron, self).__init__()
        if data['n'] != len(data['inputs']):
            raise AssertionError("The length of inputs != n")
        if data['d'] != len(data['inputs'][0]):
            raise AssertionError("The length of attributes != d")

        self.w_list = []
        self.labels = []
        self.inputs = data['inputs']
        self.targets = self.transform_labels(data['targets'])
        self.sigma = data['sigma']
        self.n = data['n']
        self.d = data['d']
        self.max_iterations = data['max_iterations']

    def transform_labels(self, labels):
        return [[-1, 1][l] for l in labels]

    def gaussian_kernel(self, x, y):
        distance = self.get_distance_between_two_points(x, y)
        return math.exp(- distance ** 2 / (2 * (self.sigma ** 2)))

    def get_distance_between_two_points(self, x, y):
        distance = 0
        for d in range(len(x)):
            distance += (x[d] - y[d]) ** 2
        return math.sqrt(distance)

    def get_sum_dot_product(self, w_list, v, labels):
        sum_dot_product = 0
        for i in range(len(w_list)):
            dot_product_value = self.gaussian_kernel(w_list[i], v)
            sum_dot_product += dot_product_value * labels[i]
        return sum_dot_product

    def is_violation_point(self, sum_dot_product, label):
        return sum_dot_product * label <= 0

    def append_w_to_w_list(self, w_list, w):
        w_list.append(w)

    def append_label_to_label_list(self, labels, label):
        labels.append(label)

    def train_one_iteration(self):
        for index, each in enumerate(self.inputs):
            label = self.targets[index]

            sum_dot_product = self.get_sum_dot_product(self.w_list, each, self.labels)
            if self.is_violation_point(sum_dot_product, label):
                self.append_w_to_w_list(self.w_list, each)
                self.append_label_to_label_list(self.labels, label)
                return True

        return False

    def train(self):

        iteration_counter = 0
        while self.train_one_iteration() and iteration_counter <= self.max_iterations:
            print('iteration: ', iteration_counter)
            iteration_counter += 1
        print('Training is finish')

    def predict(self, inputs):
        #   input_data: test data
        # return accuracy of prediction
        predicted_labels = []
        sum_dot_product = 0
        for i in range(len(inputs)):
            sum_dot_product = self.get_sum_dot_product(self.w_list, inputs[i], self.labels)
            predicted_labels.append(int(sum_dot_product > 0))
        return predicted_labels

    def accuracy_score(self, predicted_labels, targets):
        if len(predicted_labels) != len(targets):
            raise AssertionError("Predicted labels' length is not the same as target label's")
        correct = 0
        for index, each in enumerate(targets):
            if predicted_labels[index] == targets[index]:
                correct += 1
        print("Correct predicted: %s out of %s" %(correct, len(targets)))
        return correct / len(targets)

