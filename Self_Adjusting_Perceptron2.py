def activation(x):
    return 1 if x >= 1 else 0

def prediction(inputs, weights, bias):
    z = sum(x * w for x, w in zip(inputs, weights)) + bias
    return activation(z)

# Each item is ([features], target)
training_data = [
    ([0, 1, 0, 0], 0),
    ([.9, 1.1, 0, 0],1),
    ([.73, 1.4, 0, 0],0),
    ([2, 2, 2, 6], 1),
    ([1, 1, 1, 3], 0),
]

# Initialize weights and bias
num_features = len(training_data[0][0])
weights = [0.0] * num_features
bias = 0.0
learning_rate = 0.1
epochs = 6000

# Training loop
for epoch in range(epochs):
    total_error = 0
    print(f'\nEpoch {epoch + 1}')

    for inputs, target in training_data:
        output = prediction(inputs, weights, bias)
        error = target - output
        total_error += abs(error)

        # Update weights and bias
        weights = [w + learning_rate * error * x for w, x in zip(weights, inputs)]
        bias += learning_rate * error

        print(f"Input: {inputs} | Target: {target} | Output: {output} | Error: {error}")
        print(f"Updated weights: {weights} | Bias: {bias: .2f}")

    if total_error == 0:
        print("\n Converged - all predictions correct.")
        break
else:
    print("\n Reached max epochs without convergence :(")