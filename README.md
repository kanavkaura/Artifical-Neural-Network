# Artificial-Neural-Network
2-layer ANN model made using Python

The perceptron learning algorithm is implemented in the following code. A straightforward linear binary classifier used in machine learning is the perceptron. By locating a linear decision boundary, it can divide data points into two categories.
The code is explained as follows:

We first import the numpy library, which offers assistance for using arrays and carrying out mathematical operations quickly. It helps give an explanation of the perceptron() function, which accepts inputs and outputs and returns the sign of the dot product between them. The dot product gauges the similarity of the two vectors. If the dot product is positive, the sign function returns 1, if it is negative, -1, and if it is zero, it returns 0.

The function train_perceptron(), accepts the initial weight, inputs, desired outputs, learning rate, and maximum number of iterations. This function iteratively adjusts the weight to reduce the discrepancy between the desired and actual results. The final weight and the error history are returned.
We then set the training parameters, training data, and beginning weight. Ten data points make up the input data, each of which has two features (X1 and X2) and a bias term (X3). (an array of ones). For each data point, the class labels represent the desired outputs. Next, we utilize the train_perceptron() function with the given arguments to train the perceptron. We then print the final weight and the number of convergence iterations. Ten data points are classified into two classes (1 and -1) by the algorithm using their features. (X1 and X2). The best weight that minimizes the error between the projected output and the desired output is found using the perceptron learning method.

The amount of iterations needed to get closer to zero is 6. The errors between the desired output and the present output are used by the perceptron training process to change the weights on each iteration. The amount by which the weights should be adjusted after each iteration depends on the learning rate, which in this case is 0.2. The initial weights were reasonably close to the ideal weights, and/or the learning rate was adjusted to an adequate value if the perceptron converges quickly.
In our example, the perceptron was able to identify a decision boundary after 6 iterations that accurately categorizes the input data because the total error for that iteration was zero.
