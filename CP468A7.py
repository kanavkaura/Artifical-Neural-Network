import numpy as np

def perceptron(weights, inputs):
   return np.sign(np.dot(weights, inputs))

def train_perceptron(weights, inputs, desired_outputs, learning_rate, max_iterations):
   errors = []
   for _ in range(max_iterations):
       error_total = 0
       for input_vector, desired_output in zip(inputs, desired_outputs):
           curr_output = perceptron(weights, input_vector)
           error = desired_output - curr_output
           error_total += abs(error)
           weights += learning_rate * error * input_vector
       errors.append(error_total)
       if error_total == 0:
           break
   return weights, errors

# Initial weights
weights = np.array([0.75, 0.5, -0.6])

# Training data
X1 = np.array([1.0, 9.4, 2.5, 8.0, 0.5, 7.9, 7.0, 2.8, 1.2, 7.8])
X2 = np.array([1.0, 6.4, 2.1, 7.7, 2.2, 8.4, 7.0, 0.8, 3.0, 6.1])
X3 = np.ones(10)
inputs = np.column_stack((X1, X2, X3))
desired_outputs = np.array([1, -1, 1, -1, 1, -1, -1, 1, 1, -1])

# Training parameters
learning_rate = 0.2
max_iterations = 1000

# Train the perceptron
final_weights, errors = train_perceptron(weights, inputs, desired_outputs, learning_rate, max_iterations)

print("The final weights:", final_weights)
print("The number of iterations:", len(errors))