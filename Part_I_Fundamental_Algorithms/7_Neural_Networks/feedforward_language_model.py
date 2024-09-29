import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from feedforward_network import FeedforwardNeuralNetwork

# Örnek cümleler ile dil modelini oluşturmak için basit bir corpus
corpus = [
    "the cat is on the mat",
    "the dog is in the house",
    "the cat and the dog are friends",
    "the mat is near the door"
]

# Adım 1: Kelimeleri vektörize etmek için Bag-of-Words (BoW) yaklaşımı kullanıyoruz
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()  # Bag-of-words vektörleri
vocab_size = len(vectorizer.get_feature_names_out())  # Kelime dağarcığı boyutu

# Adım 2: Girdi ve çıktı verilerini hazırlama
# Cümledeki kelimeleri girdi (input) ve bir sonraki kelimeyi çıktı (output) olarak kullanacağız
def create_training_data(corpus, vectorizer):
    """
    Bu fonksiyon, cümlelerden input-output çiftlerini çıkarır. 
    Girdi olarak kelime dizisi ve çıktı olarak bir sonraki kelimenin olasılığı.
    """
    X_data, y_data = [], []
    
    for sentence in corpus:
        words = sentence.split()
        for i in range(1, len(words)):  # Kelime dizisinin her bir kısmı için
            input_sequence = " ".join(words[:i])  # Mevcut kelimeler dizisi (girdi)
            next_word = words[i]  # Sonraki kelime (çıktı)
            
            # Girdiyi vektörize et ve çıktı kelimeyi vektörize et
            X_vectorized = vectorizer.transform([input_sequence]).toarray().flatten()
            y_vectorized = vectorizer.transform([next_word]).toarray().flatten()
            
            X_data.append(X_vectorized)
            y_data.append(y_vectorized)
    
    return np.array(X_data), np.array(y_data)

X_train, y_train = create_training_data(corpus, vectorizer)

# Adım 3: Neural network parametrelerini belirle
input_size = X_train.shape[1]  # Girdi boyutu: bag-of-words vektör uzunluğu
hidden_size = 10  # Gizli katmandaki nöron sayısı
output_size = vocab_size  # Çıktı boyutu: kelime dağarcığındaki kelime sayısı

# Feedforward Neural Network sınıfını başlat
neural_net = FeedforwardNeuralNetwork(input_size, hidden_size, output_size)

# Neural network eğitim parametreleri
epochs = 500
learning_rate = 0.01

# Adım 4: Sinir ağını eğit
print("Training the feedforward language model...")
neural_net.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

# Adım 5: Modeli test etme
def predict_next_word(model, input_sequence, vectorizer):
    """
    Modeli kullanarak bir sonraki kelimeyi tahmin eder.
    """
    input_vector = vectorizer.transform([input_sequence]).toarray().flatten()
    output_vector = model.forward(input_vector)
    
    # En olası kelimeyi seç (softmax uygulamak yerine basitçe max kullanıyoruz)
    predicted_word_index = np.argmax(output_vector)
    predicted_word = vectorizer.get_feature_names_out()[predicted_word_index]
    return predicted_word

# Örnek cümlelerin başına göre bir sonraki kelimeyi tahmin etme
test_sequence = "the cat is"
predicted_word = predict_next_word(neural_net, test_sequence, vectorizer)
print(f"Input sequence: '{test_sequence}' -> Predicted next word: '{predicted_word}'")
