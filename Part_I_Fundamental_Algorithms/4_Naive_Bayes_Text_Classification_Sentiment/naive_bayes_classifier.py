import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk

# Download necessary data for tokenization
# Tokenizasyon için gerekli verileri indir
nltk.download('punkt')

# Naive Bayes Classifier class (Naive Bayes Sınıflandırıcısı sınıfı)
class NaiveBayesClassifier:
    def __init__(self):
        # Dictionary to hold word counts for each class
        # Her sınıf için kelime sayımlarını tutmak için sözlük
        self.word_counts = defaultdict(lambda: defaultdict(int))
        # Total number of documents per class
        # Her sınıf için toplam belge sayısı
        self.class_doc_counts = defaultdict(int)
        # Vocabulary to track unique words
        # Benzersiz kelimeleri izlemek için kelime dağarcığı
        self.vocab = set()
        # Total number of words per class
        # Her sınıf için toplam kelime sayısı
        self.total_word_counts = defaultdict(int)
        # Priors for each class
        # Her sınıf için öncel olasılıklar (class priors)
        self.class_priors = defaultdict(float)

    # Function to train the model (Modeli eğitmek için fonksiyon)
    def train(self, documents, labels):
        """
        Train the Naive Bayes Classifier with the provided documents and labels.
        Sağlanan belgeler ve etiketler ile Naive Bayes Sınıflandırıcısını eğitin.
        
        :param documents: List of documents (strings) for training
                          Eğitim için belgelerin listesi (string)
        :param labels: List of labels (positive/negative) corresponding to the documents
                       Belgelere karşılık gelen etiketlerin listesi (pozitif/negatif)
        """
        total_docs = len(documents)

        for i, doc in enumerate(documents):
            label = labels[i]
            self.class_doc_counts[label] += 1  # Count documents per class (Her sınıf için belge sayısını artır)
            words = word_tokenize(doc.lower())  # Tokenize and convert to lowercase (Kelimeleri tokenize et ve küçük harfe dönüştür)
            self.vocab.update(words)  # Update the vocabulary with new words (Yeni kelimelerle kelime dağarcığını güncelle)
            
            for word in words:
                self.word_counts[label][word] += 1  # Count word frequency in the class (Sınıftaki kelime sıklığını artır)
                self.total_word_counts[label] += 1  # Count total words in the class (Sınıftaki toplam kelime sayısını artır)

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
    # Yeni bir belgenin sınıfını tahmin etmek için fonksiyon
    def predict(self, document):
        """
        Predict the class of a document using the trained Naive Bayes model.
        Eğitilmiş Naive Bayes modeli kullanarak bir belgenin sınıfını tahmin edin.
        
        :param document: The input document (string) to classify (Sınıflandırılacak giriş belgesi)
        :return: The predicted class label (positive/negative) (Tahmin edilen sınıf etiketi: pozitif/negatif)
        """
        words = word_tokenize(document.lower())
        class_scores = {}

        # Calculate log probability for each class (Her sınıf için log olasılığı hesaplayın)
        for label in self.class_priors:
            log_prob = math.log(self.class_priors[label])  # Start with the prior P(label) (Öncelik P(sınıf) ile başla)
            for word in words:
                # Multiply likelihoods (in log space, add logs)
                # Olasılıkları çarp (log alanında, logları topla)
                log_prob += math.log(self.word_likelihood(word, label))
            class_scores[label] = log_prob

        # Return the class with the highest score (En yüksek skora sahip sınıfı döndür)
        return max(class_scores, key=class_scores.get)

# Example usage of NaiveBayesClassifier (NaiveBayesClassifier kullanım örneği)
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
    classifier = NaiveBayesClassifier()
    classifier.train(train_documents, train_labels)

    # Test the classifier with new documents (Yeni belgelerle sınıflandırıcıyı test edin)
    test_document = "I love the performance but dislike the plot."
    predicted_class = classifier.predict(test_document)
    print(f"Predicted class for test document: {predicted_class}")
