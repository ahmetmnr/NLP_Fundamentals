import math
import pickle  # To save the trained model (Eğitilen modeli kaydetmek için)
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
import nltk

# Download necessary data for tokenization
# Tokenizasyon için gerekli verileri indir
nltk.download('punkt')

# Binary Naive Bayes Classifier class (Binary Naive Bayes Sınıflandırıcısı sınıfı)
class BinaryNaiveBayesClassifier:
    def __init__(self):
        # Dictionary to hold word counts for each class (Her sınıf için kelime sayımlarını tutmak için sözlük)
        self.word_counts = {'positive': Counter(), 'negative': Counter()}
        # Total number of documents per class (Her sınıf için toplam belge sayısı)
        self.class_doc_counts = {'positive': 0, 'negative': 0}
        # Vocabulary to track unique words (Benzersiz kelimeleri izlemek için kelime dağarcığı)
        self.vocab = set()
        # Total number of words per class (Her sınıf için toplam kelime sayısı)
        self.total_word_counts = {'positive': 0, 'negative': 0}
        # Priors for each class (Her sınıf için öncel olasılıklar)
        self.class_priors = {'positive': 0.0, 'negative': 0.0}

    # Function to train the model (Modeli eğitmek için fonksiyon)
    def train(self, documents, labels):
        """
        Train the Binary Naive Bayes Classifier with the provided documents and labels.
        Sağlanan belgeler ve etiketler ile Binary Naive Bayes Sınıflandırıcısını eğitin.
        
        :param documents: List of documents (strings) for training
                          Eğitim için belgelerin listesi (string)
        :param labels: List of labels (positive/negative) corresponding to the documents
                       Belgelere karşılık gelen etiketlerin listesi (pozitif/negatif)
        """
        total_docs = len(documents)

        for i, doc in enumerate(documents):
            label = labels[i]
            self.class_doc_counts[label] += 1  # Count documents per class (Her sınıf için belge sayısını artır)
            words = set(word_tokenize(doc.lower()))  # Tokenize, convert to lowercase, and use binary (unique) words (Kelimeyi tokenize et ve benzersiz kelimeleri kullan)
            self.vocab.update(words)  # Update the vocabulary with new words (Yeni kelimelerle kelime dağarcığını güncelle)
            
            for word in words:  # Only count presence, not frequency (Sadece varlığı say, sıklığı değil)
                self.word_counts[label][word] += 1
                self.total_word_counts[label] += 1

        # Calculate class priors (Sınıf önceliklerini hesapla)
        for label in self.class_doc_counts:
            self.class_priors[label] = self.class_doc_counts[label] / total_docs

    # Function to calculate the probability of a word given a class
    # Bir sınıf için bir kelimenin olasılığını hesaplayan fonksiyon
    def word_likelihood(self, word, label):
        """
        Calculate the likelihood P(word|label) with Laplace smoothing.
        Laplace düzeltmesi ile P(kelime|etiket) olasılığını hesaplayın.
        
        :param word: The word to calculate the likelihood for (Olasılığı hesaplamak için kelime)
        :param label: The class label (positive/negative) (Sınıf etiketi: pozitif/negatif)
        :return: The likelihood P(word|label) (Olasılık P(kelime|etiket))
        """
        word_count = self.word_counts[label][word]
        total_words = self.total_word_counts[label]
        vocab_size = len(self.vocab)
        # Apply Laplace smoothing (Laplace düzeltmesi uygula)
        return (word_count + 1) / (total_words + vocab_size)

    # Function to predict the class of a new document
    # Yeni bir belgenin sınıfını tahmin eden fonksiyon
    def predict(self, document):
        """
        Predict the class of a document using the trained Binary Naive Bayes model.
        Eğitilmiş Binary Naive Bayes modeli kullanarak bir belgenin sınıfını tahmin edin.
        
        :param document: The input document (string) to classify (Sınıflandırılacak giriş belgesi)
        :return: The predicted class label (positive/negative) (Tahmin edilen sınıf etiketi: pozitif/negatif)
        """
        words = set(word_tokenize(document.lower()))  # Use binary presence of words (Kelimenin varlığını kullan)
        class_scores = {}

        # Calculate log probability for each class (Her sınıf için log olasılığı hesaplayın)
        for label in self.class_priors:
            log_prob = math.log(self.class_priors[label])  # Start with the prior P(label) (Öncelik P(sınıf) ile başla)
            for word in words:
                # Multiply likelihoods (in log space, add logs) (Log alanında, logları topla)
                log_prob += math.log(self.word_likelihood(word, label))
            class_scores[label] = log_prob

        # Return the class with the highest score (En yüksek skora sahip sınıfı döndür)
        return max(class_scores, key=class_scores.get)

    # Function to save the trained model to a file (Eğitilen modeli bir dosyaya kaydetmek için fonksiyon)
    def save_model(self, filename):
        """
        Save the trained Binary Naive Bayes model to a file for future use.
        Eğitilen Binary Naive Bayes modelini gelecekteki kullanım için bir dosyaya kaydedin.
        
        :param filename: The file name where the model will be saved (Modelin kaydedileceği dosya adı)
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

# Function to load a saved model (Kaydedilen bir modeli yüklemek için fonksiyon)
def load_model(filename):
    """
    Load a previously saved Binary Naive Bayes model from a file.
    Daha önce kaydedilen bir Binary Naive Bayes modelini bir dosyadan yükleyin.
    
    :param filename: The file name where the model is saved (Modelin kaydedildiği dosya adı)
    :return: The loaded Binary Naive Bayes model (Yüklenen Binary Naive Bayes modeli)
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample training data (Belgeler ve karşılık gelen etiketler)
    train_documents = [
        "I love this movie. It is fantastic!",
        "This film is terrible. I hate it.",
        "What a wonderful performance!",
        "I really dislike the plot of this movie.",
        "The film is great and very enjoyable.",
        "The movie was a disaster. Very bad acting."
    ]
    
    train_labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

    # Create and train the classifier (Sınıflandırıcıyı oluştur ve eğit)
    classifier = BinaryNaiveBayesClassifier()
    classifier.train(train_documents, train_labels)

    # Save the trained model (Eğitilen modeli kaydet)
    classifier.save_model('binary_naive_bayes_model.pkl')

    # Load the model (Modeli yükle)
    loaded_classifier = load_model('binary_naive_bayes_model.pkl')

    # Test the loaded classifier with new documents (Yeni belgelerle eğitilen sınıflandırıcıyı test edin)
    test_document = "I love the performance but dislike the plot."
    predicted_class = loaded_classifier.predict(test_document)
    print(f"Predicted class for test document: {predicted_class}")
