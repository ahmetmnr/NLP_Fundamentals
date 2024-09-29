import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from feedforward_network import FeedforwardNeuralNetwork
from neural_network_units import NeuralNetworkUnit

# Örnek veri seti (Basit bir sentiment analysis veri seti: pozitif/negatif yorumlar)
data = [
    ("I love this movie, it was fantastic!", 1),  # Pozitif yorum
    ("This film was a waste of time, so boring.", 0),  # Negatif yorum
    ("Great performance by the actors!", 1),  # Pozitif yorum
    ("I did not like the movie, it was too slow.", 0),  # Negatif yorum
    ("The plot was amazing, what a great story.", 1),  # Pozitif yorum
    ("Terrible movie, I won't recommend it.", 0)  # Negatif yorum
]

# Veri ve etiketleri ayır
sentences, labels = zip(*data)

# Adım 1: Text verisini vektörize etme
vectorizer = CountVectorizer(binary=True)  # Basit bag-of-words modeli ile vektörizasyon
X = vectorizer.fit_transform(sentences).toarray()
y = np.array(labels).reshape(-1, 1)

# Adım 2: Eğitim ve test veri setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network parametreleri
input_size = X_train.shape[1]  # Girdi boyutu, vektörizedeki kelime sayısı
hidden_size = 5  # Gizli katman nöron sayısı
output_size = 1  # Tek bir çıktı (pozitif/negatif)

# Feedforward Neural Network sınıfını başlat
neural_net = FeedforwardNeuralNetwork(input_size, hidden_size, output_size)

# Adım 3: Neural Network eğitim parametreleri
epochs = 1000
learning_rate = 0.1

# Sinir ağını eğit
print("Training the feedforward neural network for sentiment classification...")
neural_net.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

# Adım 4: Test seti üzerinde sinir ağını test et
def evaluate_model(X_test, y_test, model):
    correct = 0
    total = len(X_test)
    for i in range(total):
        prediction = model.forward(X_test[i])
        predicted_class = 1 if prediction >= 0.5 else 0  # Sigmoid çıktısına göre 0 veya 1 sınıfı
        if predicted_class == y_test[i]:
            correct += 1
    accuracy = correct / total
    return accuracy

accuracy = evaluate_model(X_test, y_test, neural_net)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Modelin cümleleri nasıl sınıflandırdığını görelim
sample_sentence = ["This is a great movie, I loved it!"]
sample_vectorized = vectorizer.transform(sample_sentence).toarray()
prediction = neural_net.forward(sample_vectorized[0])
predicted_class = 1 if prediction >= 0.5 else 0
print(f"Sample sentence prediction: {'Positive' if predicted_class == 1 else 'Negative'}")
