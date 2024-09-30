import numpy as np
from rnn import SimpleRNN  # rnn.py dosyasından SimpleRNN sınıfını içe aktarın

class BidirectionalRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # İleri ve geri RNN'leri başlat
        self.rnn_forward = SimpleRNN(input_size, hidden_size)
        self.rnn_backward = SimpleRNN(input_size, hidden_size)

        # Çıkış ağırlıkları ve bias
        self.W_out = np.random.randn(2 * hidden_size, output_size) * 0.01

        self.b_out = np.zeros((1, output_size))

    def forward(self, x):
        # İleri RNN
        output_forward = self.rnn_forward.forward(x)  # Şekil: (batch_size, timesteps, output_size)

        # Geri RNN
        reversed_x = np.flip(x, axis=1)
        output_backward = self.rnn_backward.forward(reversed_x)
        output_backward = np.flip(output_backward, axis=1)

        # Çıktıları birleştir
        output_concat = np.concatenate((output_forward, output_backward), axis=-1)  # Şekil: (batch_size, timesteps, 2 * output_size)

        # Son çıkışı hesapla
        batch_size, timesteps, _ = output_concat.shape
        output_concat_reshaped = output_concat.reshape(batch_size * timesteps, -1)
        output = np.dot(output_concat_reshaped, self.W_out) + self.b_out  # Şekil: (batch_size * timesteps, output_size)
        output = output.reshape(batch_size, timesteps, -1)

        return output

    def compute_loss(self, y_true, y_pred):
        y_pred = self.softmax(y_pred)
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
        return loss

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def backward(self, x, y_true, y_pred, learning_rate):
        # Kayıp fonksiyonunun çıktısına göre gradyan
        grad_output = y_pred - y_true  # Şekil: (batch_size, timesteps, output_size)
        batch_size, timesteps, output_size = grad_output.shape
        grad_output_reshaped = grad_output.reshape(batch_size * timesteps, output_size)

        # Gizli durumları alın
        h_forward = self.rnn_forward.hidden_states[:, :timesteps, :]  # Şekil: (batch_size, timesteps, hidden_size)
        h_backward = self.rnn_backward.hidden_states[:, :timesteps, :]  # Şekil: (batch_size, timesteps, hidden_size)
        h_backward = np.flip(h_backward, axis=1)

        # Gizli durumları birleştir
        h_concat = np.concatenate((h_forward, h_backward), axis=-1)  # Şekil: (batch_size, timesteps, 2 * hidden_size)
        h_concat_reshaped = h_concat.reshape(batch_size * timesteps, -1)

        # Gradyanları hesapla
        dW_out = np.dot(h_concat_reshaped.T, grad_output_reshaped)  # Şekil: (2 * output_size, output_size)
        db_out = np.sum(grad_output_reshaped, axis=0, keepdims=True)  # Şekil: (1, output_size)

        # Ağırlıkları güncelle
        self.W_out -= learning_rate * dW_out
        self.b_out -= learning_rate * db_out

        # Not: RNN'ler için tam BPTT burada uygulanmamıştır
        pass

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred, learning_rate)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        y_pred = self.softmax(y_pred)
        return np.argmax(y_pred, axis=-1)

# Örnek kullanım
if __name__ == "__main__":
    sentences = [
        ["the", "dog", "barks"],
        ["the", "cat", "meows"],
        ["the", "bird", "chirps"]
    ]

    labels = [
        [0, 1, 2],  # DET NOUN VERB
        [0, 1, 2],
        [0, 1, 2]
    ]

    word_to_index = {"the": 0, "dog": 1, "cat": 2, "bird": 3, "barks": 4, "meows": 5, "chirps": 6}
    index_to_label = {0: "DET", 1: "NOUN", 2: "VERB"}

    # Cümleleri ve etiketleri indekslere dönüştürün
    X = np.array([[word_to_index[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)

    # Etiketleri one-hot kodlamasına dönüştürün
    num_classes = len(index_to_label)
    y_one_hot = np.eye(num_classes)[y]  # Şekil: (batch_size, timesteps, num_classes)

    # X'i 3D şekline getirin
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Girdileri normalize edin
    X = X / float(len(word_to_index))

    # Hiperparametreleri tanımlayın
    input_size = 1
    hidden_size = 10
    output_size = num_classes

    # Çift yönlü RNN'i başlatın
    bidirectional_rnn = BidirectionalRNN(input_size, hidden_size, output_size)

    # Modeli eğitin
    bidirectional_rnn.train(X, y_one_hot, epochs=1000, learning_rate=0.01)

    # Yeni bir cümle için etiket tahmini yapın
    test_sentence = np.array([[word_to_index["the"], word_to_index["cat"], word_to_index["meows"]]])
    test_sentence = test_sentence.reshape((test_sentence.shape[0], test_sentence.shape[1], 1))
    test_sentence = test_sentence / float(len(word_to_index))
    predicted_labels = bidirectional_rnn.predict(test_sentence)

    # Tahmin edilen etiketleri insan tarafından okunabilir forma dönüştürün
    predicted_label_names = [index_to_label[label] for label in predicted_labels[0]]
    print(f"Predicted labels for 'the cat meows': {predicted_label_names}")
