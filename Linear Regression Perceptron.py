def predict(inputs, weights, bias):
    return sum(x * w for x, w in zip(inputs, weights)) + bias


# Each item: ([features], target_output)
training_data = [
    ([1, 1], 7),
    ([2, 0], 4),
    ([0, 3], 15),
    ([1, 2], 12),
]

# Initialize weights and bias
num_features = len(training_data[0][0])
weights = [0.0] * num_features
bias = 0.0
learning_rate = 0.05
epochs = 10000

# Training loop
for epoch in range(epochs):
    total_loss = 0
    print(f"\nEpoch {epoch + 1}")

    for inputs, target in training_data:
        output = predict(inputs, weights, bias)
        error = target - output
        total_loss += error ** 2

        # Update weights and bias
        weights = [w + learning_rate * error * x for w, x in zip(weights, inputs)]
        bias += learning_rate * error

        print(f"Input: {inputs} | Target: {target} | Output: {output} | Error: {error} | Total Loss: {total_loss}")
        print(f"Updated weights: {weights} | Bias: {bias:.2f}")

    if total_loss < 1e-5:
        print("\n✅ Converged — all predictions correct.")
        break
else:
    print("\n⚠️ Reached max epochs without full convergence.")