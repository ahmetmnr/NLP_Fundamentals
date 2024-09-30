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
        Simple RNN'i giriş ve gizli katmanlar için rastgele ağırlıklarla başlatır.
        
        :param input_size: Giriş boyutu (örneğin zaman adımı başına özellik sayısı)
        :param hidden_size: Gizli katmandaki nöron sayısı
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
        RNN üzerinden ileri besleme işlemi.
        
        :param x: Girdi dizisi (batch_size, timesteps, input_size)
        :return: Gizli durumlar (hidden_states) (batch_size, timesteps, hidden_size)
        """
        batch_size, timesteps, _ = x.shape
        h = np.zeros((batch_size, self.hidden_size))
        hidden_states = np.zeros((batch_size, timesteps, self.hidden_size))

        for t in range(timesteps):
            x_t = x[:, t, :]
            h = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            hidden_states[:, t, :] = h

        return hidden_states  # Shape: (batch_size, timesteps, hidden_size)


class StackedRNN:
    def __init__(self, input_size, hidden_sizes):
        """
        Initializes the stacked RNN with multiple layers.
        Katmanlı RNN'i birden fazla katmanla başlatır.

        :param input_size: İlk katmanın giriş boyutu
        :param hidden_sizes: Her katmandaki gizli katman boyutlarının listesi
        """
        self.layers = []
        current_input_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(SimpleRNN(current_input_size, hidden_size))
            current_input_size = hidden_size

    def forward(self, x):
        """
        Forward pass through the stacked RNN layers.
        Katmanlı RNN katmanları üzerinden ileri besleme işlemi.

        :param x: Girdi dizisi (batch_size, timesteps, input_size)
        :return: Son katmanın gizli durumları (hidden_states) (batch_size, timesteps, hidden_size)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x  # Son katmanın gizli durumları (hidden_states)


# Example usage
if __name__ == "__main__":
    # XOR problemi için örnek giriş ve çıkış verisi
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])  # Girdi dizileri
    y = np.array([[1, 0],
                  [0, 1],
                  [0, 1],
                  [1, 0]])  # XOR için gerçek etiketler

    # Giriş dizisini 3D'ye (batch_size, timesteps, input_size) çevir
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Zaman adımı boyutunu ekle

    # RNN'i başlat
    input_size = 2  # Girdi boyutu (örneğin XOR problemi için)
    hidden_sizes = [5, 5]  # Gizli katmanlardaki nöron sayısı (iki katman)
    output_size = 2  # Çıktı boyutu (örneğin iki sınıf)

    stacked_rnn = StackedRNN(input_size, hidden_sizes)

    # Çıkış katmanı ağırlıklarını ve bias'larını başlat
    W_out = np.random.randn(hidden_sizes[-1], output_size) * 0.01
    b_out = np.zeros((1, output_size))

    # Eğitim parametreleri
    epochs = 1000
    learning_rate = 0.01

    for epoch in range(epochs):
        # İleri besleme
        h = stacked_rnn.forward(X)  # h: (batch_size, timesteps, hidden_size)
        last_hidden_state = h[:, -1, :]  # Son gizli durumu al

        # Çıktıları hesapla
        y_pred = np.dot(last_hidden_state, W_out) + b_out  # (batch_size, output_size)
        y_pred = softmax(y_pred)  # Olasılık elde etmek için softmax uygula

        # Kayıp hesapla
        loss = -np.sum(y * np.log(y_pred + 1e-8)) / y.shape[0]

        # Gradyanları hesapla (sadece çıkış katmanı için)
        grad_output = y_pred - y  # (batch_size, output_size)
        dW_out = np.dot(last_hidden_state.T, grad_output) / y.shape[0]
        db_out = np.sum(grad_output, axis=0, keepdims=True) / y.shape[0]

        # Çıkış katmanı ağırlıklarını ve bias'larını güncelle
        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out

        # Not: RNN ağırlıkları için geri yayılım (BPTT) uygulanmamıştır

        # Her 100 epoch'ta kaybı yazdır
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # Eğitilen RNN'i test et
    for input_data in X:
        h = stacked_rnn.forward(input_data.reshape(1, 1, -1))
        last_hidden_state = h[:, -1, :]
        output = np.dot(last_hidden_state, W_out) + b_out
        output = softmax(output)
        print(f"Input: {input_data.flatten()}, Predicted Output: {output}")
 