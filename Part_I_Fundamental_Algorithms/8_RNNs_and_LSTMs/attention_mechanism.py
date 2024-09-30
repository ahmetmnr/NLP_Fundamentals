import numpy as np
def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    Her bir skor için softmax değerini hesaplar.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtract max for numeric stability
    return e_x / e_x.sum(axis=-1, keepdims=True)


class Attention:
    def __init__(self, hidden_size):
        """
        Initializes the attention mechanism with randomly initialized weights.
        """
        self.hidden_size = hidden_size
        self.W1 = np.random.randn(hidden_size, hidden_size) * 0.01  # Weight for encoder hidden states
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01  # Weight for decoder hidden states
        self.V = np.random.randn(hidden_size, 1) * 0.01  # Weight for calculating scores

    def _calculate_score(self, hidden_encoder, hidden_decoder):
        """
        Calculates the alignment score between encoder and decoder hidden states.
        """
        score = np.dot(np.tanh(np.dot(hidden_encoder, self.W1) + np.dot(hidden_decoder, self.W2)), self.V)
        return score

    def forward(self, encoder_outputs, hidden_decoder):
        """
        Forward pass through the attention mechanism.
        """
        # Calculate alignment scores between decoder hidden state and each encoder hidden state
        scores = np.array([self._calculate_score(h_enc, hidden_decoder) for h_enc in encoder_outputs])

        # Apply softmax to the scores to obtain attention weights
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=0)

        # Compute the context vector as the weighted sum of encoder outputs
        context_vector = np.sum(attention_weights * encoder_outputs, axis=0)

        return context_vector, attention_weights

# Example usage within an encoder-decoder model
class Encoder:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01  # Input to hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.b_h = np.zeros((1, hidden_size))  # Bias for hidden state

    def forward(self, x):
        batch_size, timesteps, _ = x.shape
        h = np.zeros((batch_size, self.hidden_size))  # Initialize hidden state
        encoder_outputs = []

        for t in range(timesteps):
            x_t = x[:, t, :]
            h = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            encoder_outputs.append(h)

        encoder_outputs = np.stack(encoder_outputs, axis=1)  # Shape: (batch_size, timesteps, hidden_size)
        return encoder_outputs, h  # Return all hidden states and the last hidden state

class Decoder:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_hy = np.random.randn(hidden_size + input_size, output_size) * 0.01  # Adjusted Hidden to output weight size
        self.b_y = np.zeros((1, output_size))  # Bias for output
        self.attention = Attention(hidden_size)

    def forward(self, decoder_input, hidden_encoder, encoder_outputs):
        """
        Forward pass through the decoder with attention.
        """
        # Reshape decoder_input for compatibility
        decoder_input = decoder_input.reshape(-1, decoder_input.shape[-1])

        # Calculate the context vector using the attention mechanism
        context_vector, attention_weights = self.attention.forward(encoder_outputs, hidden_encoder)

        # Combine the context vector with the current input and pass through the decoder
        context_vector = context_vector.reshape(1, -1)  # Reshape context vector to match input shape
        decoder_combined_input = np.hstack((decoder_input, context_vector))

        # Now decoder_combined_input has shape (batch_size, hidden_size + input_size)
        decoder_hidden = np.tanh(np.dot(decoder_combined_input, self.W_hy) + self.b_y)
        output = softmax(np.dot(decoder_hidden, self.W_hy) + self.b_y)

        return output, decoder_hidden, attention_weights

# Example usage of attention mechanism in encoder-decoder RNN
if __name__ == "__main__":
    input_size = 5   # Input size (e.g., embedding size)
    hidden_size = 10  # Hidden state size
    output_size = 5  # Output size (e.g., vocabulary size)

    # Define encoder and decoder
    encoder = Encoder(input_size, hidden_size)
    decoder = Decoder(input_size, hidden_size, output_size)

    # Example input data (batch_size=1, timesteps=4, input_size=5)
    encoder_input = np.random.randn(1, 4, input_size)
    decoder_input = np.random.randn(1, 1, input_size)  # Single timestep for decoder input

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder.forward(encoder_input)

    # Forward pass through decoder with attention
    output, decoder_hidden, attention_weights = decoder.forward(decoder_input, encoder_hidden, encoder_outputs)

    # Print results
    print("Decoder Output:", output)
    print("Attention Weights:", attention_weights)
