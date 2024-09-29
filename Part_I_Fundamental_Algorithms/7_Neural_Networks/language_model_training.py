import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from feedforward_network import FeedforwardNeuralNetwork
from feedforward_language_model import create_training_data, predict_next_word

# Örnek corpus, dil modeli eğitimi için basit bir metin kümesi
corpus = [
    "the cat is on the mat",
    "the dog is in the house",
    "the cat and the dog are friends",
    "the mat is near the door",
    "the dog likes to play with the ball",
    "the cat sleeps on the mat"
]

# Adım 1: Bag-of-Words ile vektörizasyon
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
vocab_size = len(vectorizer.get_feature_names_out())

# Adım 2: Girdi ve çıktı verilerini oluşturma
X_train, y_train = create_training_data(corpus, vectorizer)

# Adım 3: Eğitim ve test setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Neural Network parametreleri
input_size = X_train.shape[1]  # Girdi boyutu: kelime sayısı
hidden_size = 10  # Gizli katman nöron sayısı
output_size = vocab_size  # Çıktı boyutu: kelime dağarcığı boyutu

# Feedforward Neural Network modeli başlat
neural_net = FeedforwardNeuralNetwork(input_size, hidden_size, output_size)

# Adım 4: Model eğitimi parametreleri
epochs = 1000
learning_rate = 0.01

# Adım 5: Modeli eğitme
print("Training the neural language model...")
neural_net.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

# Adım 6: Modeli test etme
def evaluate_language_model(model, X_test, y_test):
    """
    Test seti üzerinde modeli değerlendirir ve doğruluk oranını hesaplar.
    """
    correct = 0
    total = len(X_test)
    
    for i in range(total):
        prediction = model.forward(X_test[i])
        predicted_word_index = np.argmax(prediction)
        actual_word_index = np.argmax(y_test[i])
        
        if predicted_word_index == actual_word_index:
            correct += 1
    
    accuracy = correct / total
    return accuracy

accuracy = evaluate_language_model(neural_net, X_test, y_test)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

# Adım 7: Modelin tahmin ettiği bir sonraki kelimeyi kontrol etme
test_sequence = "the cat is"
predicted_word = predict_next_word(neural_net, test_sequence, vectorizer)
print(f"Input sequence: '{test_sequence}' -> Predicted next word: '{predicted_word}'")
