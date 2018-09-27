from pylab import *
from Perceptron import Perceptron
from numpy import mean

training_data = open("training_data.txt", "r")
training_data_str = training_data.read().split()

plots_x = []
plots_y = []
plots_class = []

for i in range(50):
    plots_x.append(int(training_data_str[i*3])/25)
    plots_y.append(int(training_data_str[i*3 + 1])/25)
    if int(training_data_str[i * 3 + 2]) == 1:
        plots_class.append('r')
    else:
        plots_class.append('b');

# scatter(plots_x, plots_y, s=100, marker='.', c=plots_class)
#
# show()

# training
p = Perceptron(2, 0.001)

testing_data = open("testing_data.txt", "r")
testing_data_str = testing_data.read().split()

plots_x_testing = []
plots_y_testing = []
plots_class_testing = []

for i in range(50):
    plots_x_testing.append(int(testing_data_str[i*3]))
    plots_y_testing.append(int(testing_data_str[i*3 + 1]))
    if int(testing_data_str[i * 3 + 2]) == p.feed_forward([int(testing_data_str[i*3]), int(testing_data_str[i*3 + 1])]):
        plots_class_testing.append('g')
    else:
        plots_class_testing.append('k')

scatter(plots_x_testing, plots_y_testing, s=100, marker='.', c=plots_class_testing)

show()

loss_x = []
loss_y = []
training_loss = []

for i in range(5000):
    for j in range(50):
        p.train([int(training_data_str[j*3])/25, int(training_data_str[j*3 + 1])/25], int(training_data_str[j*3 + 2]))
        training_loss.append(p.get_error_score())
    loss_x.append(i)
    loss_y.append(np.mean(training_loss))


# show loss function
scatter(loss_x, loss_y, s=100, marker='.', c='k')
show()

# show results

# reset vectors
plots_x_testing = []
plots_y_testing = []
plots_class_testing = []
plots_class_correctness = []

for i in range(50):
    plots_x_testing.append(int(testing_data_str[i*3]))
    plots_y_testing.append(int(testing_data_str[i*3 + 1]))
    if int(testing_data_str[i * 3 + 2]) == p.feed_forward([int(testing_data_str[i*3]), int(testing_data_str[i*3 + 1])]):
        plots_class_correctness.append('g')
    else:
        plots_class_correctness.append('k')

    if int(testing_data_str[i * 3 + 2]) == 1:
        plots_class_testing.append('r')
    else:
        plots_class_testing.append('b')

scatter(plots_x_testing, plots_y_testing, s=100, marker='.', c=plots_class_testing)
show()
scatter(plots_x_testing, plots_y_testing, s=100, marker='.', c=plots_class_correctness)
show()

print(int(p.feed_forward([75, -43])))
