import numpy as np
import random

class Word2VecSkipGram:
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.01, window_size=2):
        """
        Initializes the Word2Vec Skip-Gram model.
        Word2Vec Skip-Gram modelini başlatır.
        
        :param vocab_size: Size of the vocabulary (Kelime dağarcığının boyutu)
        :param embedding_dim: Dimension of word embeddings (Kelime gömme boyutu)
        :param learning_rate: Learning rate for gradient descent (Gradient descent için öğrenme oranı)
        :param window_size: Size of the context window (Bağlam penceresinin boyutu)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.window_size = window_size
        
        # Initialize weights (Ağırlıkları başlat)
        self.center_embeddings = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
        self.context_embeddings = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
    
    def generate_training_data(self, corpus, word_to_index):
        """
        Generates training data in the form of (center_word, context_word) pairs from the corpus.
        Corpus'tan (center_word, context_word) çiftleri şeklinde eğitim verisi üretir.
        
        :param corpus: List of tokenized sentences (Tokenize edilmiş cümlelerden oluşan liste)
        :param word_to_index: Dictionary mapping words to their indices (Kelime indeks eşlemesi yapan sözlük)
        :return: List of training pairs (Eğitim çiftlerinden oluşan liste)
        """
        training_data = []
        
        for sentence in corpus:
            sentence_indices = [word_to_index[word] for word in sentence if word in word_to_index]
            
            for i, center_word in enumerate(sentence_indices):
                # Define the context window (Bağlam penceresini belirle)
                window_start = max(0, i - self.window_size)
                window_end = min(len(sentence_indices), i + self.window_size + 1)
                
                # Create (center, context) pairs (Merkez ve bağlam çiftleri oluştur)
                for j in range(window_start, window_end):
                    if i != j:
                        context_word = sentence_indices[j]
                        training_data.append((center_word, context_word))
        
        return training_data
    
    def softmax(self, x):
        """
        Computes the softmax function for a given input vector.
        Verilen giriş vektörü için softmax fonksiyonunu hesaplar.
        
        :param x: Input vector (Girdi vektörü)
        :return: Softmax probabilities (Softmax olasılıkları)
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def train(self, training_data, epochs=1000):
        """
        Trains the Word2Vec Skip-Gram model using stochastic gradient descent.
        Word2Vec Skip-Gram modelini stochastic gradient descent kullanarak eğitir.
        
        :param training_data: List of (center_word, context_word) pairs (Merkez ve bağlam kelime çiftlerinden oluşan liste)
        :param epochs: Number of training epochs (Eğitim dönemlerinin sayısı)
        """
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)
            
            for center_word, context_word in training_data:
                # Forward pass (İleri geçiş)
                center_vector = self.center_embeddings[center_word]
                context_vector = self.context_embeddings[context_word]
                score = np.dot(self.context_embeddings, center_vector)
                probs = self.softmax(score)
                
                # Compute loss (Kayıp hesapla)
                loss = -np.log(probs[context_word])
                total_loss += loss
                
                # Backward pass (Geri yayılım)
                probs[context_word] -= 1  # Subtract 1 from the true context word probability
                grad_center = np.dot(probs, self.context_embeddings)  # Gradient with respect to center embedding
                grad_context = np.outer(probs, center_vector)  # Gradient with respect to context embeddings
                
                # Update embeddings (Gömüleri güncelle)
                self.center_embeddings[center_word] -= self.learning_rate * grad_center
                self.context_embeddings -= self.learning_rate * grad_context

            # Optional: Print loss every 100 epochs (İsteğe bağlı: Her 100 epoch'ta kayıp değeri yazdır)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    
    def get_word_embedding(self, word_index):
        """
        Retrieves the word embedding for a specific word index.
        Belirli bir kelime indeksi için kelime gömmesini getirir.
        
        :param word_index: Index of the word (Kelimenin indeksi)
        :return: Word embedding vector (Kelime gömme vektörü)
        """
        return self.center_embeddings[word_index]

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Example corpus (Örnek corpus)
    corpus = [
        ["i", "love", "machine", "learning"],
        ["machine", "learning", "is", "great"],
        ["natural", "language", "processing", "is", "part", "of", "machine", "learning"],
        ["i", "enjoy", "learning", "about", "natural", "language", "processing"]
    ]
    
    # Build vocabulary (Kelime dağarcığını oluştur)
    vocab = set(word for sentence in corpus for word in sentence)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    
    # Initialize the Word2Vec Skip-Gram model (Word2Vec Skip-Gram modelini başlat)
    vocab_size = len(vocab)
    embedding_dim = 100  # Kelime gömme boyutu
    model = Word2VecSkipGram(vocab_size, embedding_dim, learning_rate=0.01, window_size=2)
    
    # Generate training data (Eğitim verisini oluştur)
    training_data = model.generate_training_data(corpus, word_to_index)
    
    # Train the model (Modeli eğit)
    model.train(training_data, epochs=1000)
    
    # Get the embedding for a specific word (Belirli bir kelime için gömme vektörünü al)
    word = "machine"
    word_index = word_to_index[word]
    embedding = model.get_word_embedding(word_index)
    print(f"Word embedding for '{word}':")
    print(embedding)
