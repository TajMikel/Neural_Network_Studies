def activation(x):
    return 1 if x >= 0.8 else 0

def predict(inputs, weights, bias):
    z = sum(x * w for x, w in zip(inputs, weights)) + bias
    return activation(z)

# Each item: ([features], target_output)
training_data = [
    ([1.0, 2.0, 1.0, 0.0], 1),   # Strong features → recommend
    ([0.0, 0.0, 0.0, 1.0], 0),   # Weak features → don't recommend
    ([0.5, 1.0, 0.0, 0.0], 1),   # Moderate strength
    ([0.0, 0.2, 0.1, 0.0], 0),   # Likely too weak
]

# Initialize weights and bias
num_features = len(training_data[0][0])
weights = [0.0] * num_features
bias = 0.0
learning_rate = 0.1
epochs = 30

# Training loop
for epoch in range(epochs):
    total_error = 0
    print(f"\nEpoch {epoch + 1}")

    for inputs, target in training_data:
        output = predict(inputs, weights, bias)
        error = target - output
        total_error += abs(error)

        # Update weights and bias
        weights = [w + learning_rate * error * x for w, x in zip(weights, inputs)]
        bias += learning_rate * error

        print(f"Input: {inputs} | Target: {target} | Output: {output} | Error: {error}")
        print(f"Updated weights: {weights} | Bias: {bias:.2f}")

    if total_error == 0:
        print("\n✅ Converged — all predictions correct.")
        break
else:
    print("\n⚠️ Reached max epochs without full convergence.")