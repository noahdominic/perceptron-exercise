# Simple Classification w/ a Single Layer Perceptron
# Noah Dominic M. Silvio

# Code based on "Nature of Code 10.01: Simple Perceptron" [YouTube Video].
# By Daniel Shiffman @ https://github.com/shiffman

import random


class Perceptron:
    """This is the Perceptron class"""

    def __init__(self, number_of_inputs, learning_rate_):
        """Perceptron constructor"""
        # vector of input of weights
        self.weights = []
        self.error_score = 0

        self.learning_rate = learning_rate_

        # generates random weights for inputs
        for i in range(number_of_inputs):
            self.weights.append(random.randint(1, 100))

    def __repr__(self):
        """Returns string of list of weights"""
        return str(self.weights)

    def feed_forward(self, inputs):
        # Sums up all values
        temp_sum = 0
        for i in range(len(inputs)):
            temp_sum += inputs[i] * self.weights[i]

        # Activation function.
        # Could be placed as a different function but won't be necessary for now
        if temp_sum > 0:
            return 0
        else:
            return 1

    def train(self, input_array, answer):
        guess = self.feed_forward(input_array)
        error_score = guess - answer
        self.error_score = abs(error_score)
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error_score * input_array[i]

    def get_error_score(self):
        return self.error_score


# Code below for testing purposes only
# p = Perceptron(2)
# print(p)
#
# for i in range(100):
#     x = random.randint(-100, 100)
#     y = random.randint(-100, 100)
#     string = str(x) + " " + str(y)
#     if x >= -y:
#         string = string + " " + str(1)
#     else:
#         string = string + " " + str(0)
#     print(string)
