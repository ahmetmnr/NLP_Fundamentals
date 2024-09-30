import numpy as np

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    Her bir skor için softmax değerini hesaplar.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numeric stability için max çıkar
    return e_x / e_x.sum(axis=-1, keepdims=True)

class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        """
        Initializes the Simple RNN with random weights for input and hidden layers.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01  # Input to hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden

        # Initialize biases
        self.b_h = np.zeros((1, hidden_size))

    def forward(self, x):
        """
        Forward pass through the RNN.
        """
        batch_size, timesteps, _ = x.shape
        h = np.zeros((batch_size, self.hidden_size))
        hidden_states = np.zeros((batch_size, timesteps, self.hidden_size))

        for t in range(timesteps):
            x_t = x[:, t, :]
            h = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            hidden_states[:, t, :] = h

        self.hidden_states = hidden_states  # Store hidden states for potential backward pass
        return hidden_states  # Shape: (batch_size, timesteps, hidden_size)

    # The backward function is not implemented in this simple example

# Example usage
if __name__ == "__main__":
    # Sample input and output for XOR problem
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])  # Input sequences
    y = np.array([[1, 0],
                  [0, 1],
                  [0, 1],
                  [1, 0]])  # True labels for XOR

    # Reshape X to 3D (batch_size, timesteps, input_size)
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Add timestep dimension

    # Initialize the RNN
    input_size = 2  # Input size (e.g., for XOR problem)
    hidden_size = 5  # Number of neurons in the hidden layer
    output_size = 2  # Output size (e.g., two classes)

    rnn = SimpleRNN(input_size, hidden_size)

    # Initialize output layer weights and biases
    W_out = np.random.randn(hidden_size, output_size) * 0.01
    b_out = np.zeros((1, output_size))

    # Training parameters
    epochs = 1000
    learning_rate = 0.01

    for epoch in range(epochs):
        # Forward pass
        h = rnn.forward(X)  # h: (batch_size, timesteps, hidden_size)
        last_hidden_state = h[:, -1, :]  # Take the last hidden state

        # Compute outputs
        y_pred = np.dot(last_hidden_state, W_out) + b_out  # Shape: (batch_size, output_size)
        y_pred = softmax(y_pred)  # Apply softmax to get probabilities

        # Compute loss
        loss = -np.sum(y * np.log(y_pred + 1e-8)) / y.shape[0]

        # Compute gradients (for output layer only)
        grad_output = y_pred - y  # Shape: (batch_size, output_size)
        dW_out = np.dot(last_hidden_state.T, grad_output) / y.shape[0]
        db_out = np.sum(grad_output, axis=0, keepdims=True) / y.shape[0]

        # Update output layer weights and biases
        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out

        # Note: Backpropagation through time (BPTT) for the RNN weights is not implemented

        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # Test the trained RNN
    for input_data in X:
        h = rnn.forward(input_data.reshape(1, 1, -1))
        last_hidden_state = h[:, -1, :]
        output = np.dot(last_hidden_state, W_out) + b_out
        output = softmax(output)
        print(f"Input: {input_data.flatten()}, Predicted Output: {output}")
