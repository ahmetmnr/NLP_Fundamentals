import numpy as np
from rnn import SimpleRNN  # Import the previously defined SimpleRNN class

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    Her bir skor için softmax değerini hesaplar.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtract max for numeric stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

class EncoderRNN:
    def __init__(self, input_size, hidden_size):
        """
        Initializes the Encoder RNN.
        Encoder RNN başlatma.
        
        :param input_size: Input size (Girdi boyutu)
        :param hidden_size: Hidden layer size (Gizli katman boyutu)
        """
        self.rnn = SimpleRNN(input_size, hidden_size)

    def forward(self, input_sequence):
        """
        Forward pass through the encoder.
        Encoder üzerinden ileri besleme işlemi.
        
        :param input_sequence: Input sequence (batch_size, timesteps, input_size)
        :return: Final hidden state (Son gizli durum)
        """
        hidden_states = self.rnn.forward(input_sequence)
        return hidden_states[:, -1, :]  # Return the last hidden state


class DecoderRNN:
    def __init__(self, output_size, hidden_size):
        """
        Initializes the Decoder RNN.
        Decoder RNN başlatma.
        
        :param output_size: Output size (Çıktı boyutu)
        :param hidden_size: Hidden layer size (Gizli katman boyutu)
        """
        self.rnn = SimpleRNN(output_size, hidden_size)
        self.W_out = np.random.randn(hidden_size, output_size) * 0.01  # Random weights
        self.b_out = np.zeros((1, output_size))  # Bias initialized to zero

    def forward(self, decoder_input, hidden_state):
        """
        Forward pass through the decoder.
        Decoder üzerinden ileri besleme işlemi.
        
        :param decoder_input: Decoder input sequence (batch_size, timesteps, output_size)
        :param hidden_state: Final hidden state from the encoder (Encoder'dan gelen son gizli durum)
        :return: Predicted outputs (Tahmin edilen çıktılar)
        """
        hidden_states = self.rnn.forward(decoder_input)
        outputs = []
        for t in range(hidden_states.shape[1]):
            h_t = hidden_states[:, t, :]  # Hidden state at time t
            output_t = np.dot(h_t, self.W_out) + self.b_out  # Calculate output
            output_t = softmax(output_t)  # Normalize with softmax
            outputs.append(output_t)

        return np.stack(outputs, axis=1)  # (batch_size, timesteps, output_size)


class EncoderDecoderRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the encoder-decoder model using SimpleRNN.
        SimpleRNN kullanarak encoder-decoder modelini başlatır.
        
        :param input_size: Encoder input size (Encoder giriş boyutu)
        :param hidden_size: Hidden layer size (Gizli katman boyutu)
        :param output_size: Decoder output size (Decoder çıktı boyutu)
        """
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(output_size, hidden_size)

    def forward(self, encoder_input, decoder_input):
        """
        Forward pass through the encoder-decoder model.
        Encoder-decoder modeli üzerinden ileri besleme işlemi.
        
        :param encoder_input: Encoder input sequence (batch_size, timesteps, input_size)
        :param decoder_input: Decoder input sequence (target sequence shifted right)
        :return: Decoder's predicted outputs (Decoder'ın ürettiği çıktı)
        """
        # Encode the input sequence
        encoder_hidden_state = self.encoder.forward(encoder_input)  # Get the encoder's final hidden state

        # Use the encoder's last hidden state as the decoder's initial hidden state
        decoder_output = self.decoder.forward(decoder_input, encoder_hidden_state)

        return decoder_output

    def compute_loss(self, y_true, y_pred):
        """
        Computes the loss between the true and predicted sequences.
        Gerçek ve tahmin edilen diziler arasındaki kaybı hesaplar.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted labels (Tahmin edilen etiketler)
        :return: Loss value (Kayıp değeri)
        """
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
        return loss

    def train(self, encoder_input, decoder_input, y_true, epochs=1000, learning_rate=0.01):
        """
        Trains the encoder-decoder model using simple gradient descent.
        Encoder-decoder modelini gradient descent kullanarak eğitir.
        
        :param encoder_input: Encoder input sequence (Input to encoder)
        :param decoder_input: Decoder input sequence (Input to decoder)
        :param y_true: True output sequences (Gerçek çıktılar)
        :param epochs: Number of training iterations (Eğitim iterasyon sayısı)
        :param learning_rate: Learning rate (Öğrenme oranı)
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(encoder_input, decoder_input)

            # Compute loss
            loss = self.compute_loss(y_true, y_pred)

            # Backward pass (only updating decoder)
            grad_output = y_pred - y_true
            dW_out = np.dot(self.decoder.rnn.hidden_states.reshape(-1, self.decoder.rnn.hidden_size).T,
                            grad_output.reshape(-1, grad_output.shape[2]))
            db_out = np.sum(grad_output, axis=(0, 1))

            # Update decoder output layer
            self.decoder.W_out -= learning_rate * dW_out
            self.decoder.b_out -= learning_rate * db_out

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, encoder_input, decoder_input):
        """
        Generates predictions for a given input sequence.
        Verilen giriş dizisi için tahminler üretir.
        
        :param encoder_input: Encoder input sequence (Input to encoder)
        :param decoder_input: Decoder input sequence (Input to decoder)
        :return: Predicted sequences (Tahmin edilen diziler)
        """
        y_pred = self.forward(encoder_input, decoder_input)
        return np.argmax(y_pred, axis=-1)


# Example usage
if __name__ == "__main__":
    # Updated input sequences with more diversity
    encoder_input = np.array([[[0.1], [0.5], [0.9]],  # Sequence 1
                              [[0.3], [0.6], [0.8]]])  # Sequence 2

    decoder_input = np.array([[[0.2], [0.4], [0.7]],  # Sequence 1
                              [[0.5], [0.7], [0.9]]])  # Sequence 2

    # True output sequences (Target output)
    y_true = np.array([[[0.2], [0.4], [0.7]],  # Sequence 1
                       [[0.5], [0.7], [0.9]]])  # Sequence 2

    # Model parameters
    input_size = 1  # Input size
    hidden_size = 20  # Increased hidden layer size
    output_size = 1  # Output size

    # Initialize the encoder-decoder RNN model
    encoder_decoder = EncoderDecoderRNN(input_size, hidden_size, output_size)

    # Train the model
    encoder_decoder.train(encoder_input, decoder_input, y_true, epochs=1000, learning_rate=0.01)

    # Test the model with new input sequences
    test_encoder_input = np.array([[[0.1], [0.5], [0.9]]])  # New encoder input
    test_decoder_input = np.array([[[0.2], [0.4], [0.7]]])  # New decoder input

    predicted_output = encoder_decoder.predict(test_encoder_input, test_decoder_input)
    print(f"Predicted output: {predicted_output}")
