import math


class MarginPerceptron():
    def __init__(self, data):
        super(MarginPerceptron, self).__init__()
        if data['n'] != len(data['inputs']):
            raise AssertionError("The length of inputs != n")
        if data['d'] != len(data['inputs'][0]):
            raise AssertionError("The length of attributes != d")

        self.w = [0 for _ in range(data['d'])]
        # R is the largest radius in the inputs data set
        self.R = self.calculate_R(data['inputs'])
        self.inputs = data['inputs']
        self.targets = data['targets']
        # initialize the gamma guess to be R
        self.gamma_guess = self.R
        self.max_iterations = self.calculate_max_iterations(self.R, self.gamma_guess)

    def calculate_R(self, inputs):
        max_radius = 0
        for each in inputs:
            radius = math.sqrt(sum(list(map(lambda x: x * x, each))))
            if radius > max_radius:
                max_radius = radius
        return max_radius

    def calculate_max_iterations(self, R, gamma_guess):
        return int(12 * R * R / (gamma_guess * gamma_guess))

    def dot_product(self, x, y):
        if len(x) != len(y):
            raise AssertionError("cannot dot_product between two two vectors")
        result = 0
        for index, each in enumerate(x):
            result += each * y[index]
        return result

    def get_label(self, index):
        return [-1, 1][self.targets[index]]

    def is_label_match(self, dot_product_value, index):
        return dot_product_value * self.get_label(index) > 0

    def is_distance_too_close_to_plane(self, dot_product_value):
        return abs(dot_product_value / math.sqrt(sum(list(map(lambda x: x * x, self.w))))) < self.gamma_guess / 2.0

    def is_violation_point(self, dot_product_value, index):
        return (not self.is_label_match(dot_product_value, index)) or self.is_distance_too_close_to_plane(dot_product_value)

    def train_one_iteration(self):
        violation = -1
        for index, each in enumerate(self.inputs):
            dot_product_value = self.dot_product(self.w, each)
            if self.is_violation_point(dot_product_value, index):
                violation = index
                break
        return violation

    def train_one_round(self):
        for _ in range(self.max_iterations):
            violation_index = self.train_one_iteration()
            if violation_index > -1:
                # adjust the w if there is violation
                for i in range(len(self.inputs[0])):
                    self.w[i] += self.get_label(violation_index) * self.inputs[violation_index][i]
            else:
                return False
        return True

    def train(self):
        while True:
            print("training")
            if self.train_one_round() is False:
                # there is no more violation point
                print("Success")
                return
            # update gamma guess and then train again
            self.divide_gamma_guess_by_two()
            if self.validate_gamma_guess() is False:
                return
            self.max_iterations = self.calculate_max_iterations(self.R, self.gamma_guess)

    def divide_gamma_guess_by_two(self):
        self.gamma_guess = self.gamma_guess / 2

    def validate_gamma_guess(self):
        return self.is_gamma_guess_too_small()

    def is_gamma_guess_too_small(self):
        if self.gamma_guess <= 1e-8:
            print('Gamma_guess is small, data is not separable')
            return False
        return True

    def predict(self, inputs):
        # get the accuracy of trained w
        return [int(self.dot_product(each, self.w) > 0) for each in inputs]

    def accuracy_score(self, predicted_labels, targets):
        if len(predicted_labels) != len(targets):
            raise AssertionError("Predicted labels' length is not the same as target label's")
        correct = 0
        for index, each in enumerate(targets):
            if predicted_labels[index] == targets[index]:
                correct += 1
        print("Correct predicted: %s out of %s" %(correct, len(targets)))
        return correct / len(targets)
