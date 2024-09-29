import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from logistic_regression_classifier import LogisticRegressionClassifier
import nltk

# Download necessary NLTK data
# Gerekli NLTK verilerini indir
nltk.download('punkt')

# Preprocessing and feature extraction for sentiment classification
# Duygu sınıflandırması için ön işleme ve özellik çıkarımı
class SentimentClassification:
    def __init__(self, positive_words, negative_words):
        """
        Initializes the SentimentClassification class with positive and negative word lists.
        Olumlu ve olumsuz kelime listeleriyle SentimentClassification sınıfını başlatır.
        
        :param positive_words: List of positive sentiment words (Olumlu duygu kelimeleri listesi)
        :param negative_words: List of negative sentiment words (Olumsuz duygu kelimeleri listesi)
        """
        self.positive_words = positive_words  # Olumlu kelimeler
        self.negative_words = negative_words  # Olumsuz kelimeler

    def preprocess(self, text):
        """
        Tokenizes and processes the text to extract features (positive and negative word counts).
        Metni tokenize eder ve özellikleri çıkarmak için işler (olumlu ve olumsuz kelime sayısı).
        
        :param text: Input text (Giriş metni)
        :return: Feature vector with positive and negative word counts (Olumlu ve olumsuz kelime sayısı ile özellik vektörü)
        """
        tokens = word_tokenize(text.lower())  # Lowercase and tokenize the text (Metni küçük harfe çevir ve tokenize et)
        pos_count = sum([1 for word in tokens if word in self.positive_words])  # Olumlu kelime sayısı
        neg_count = sum([1 for word in tokens if word in self.negative_words])  # Olumsuz kelime sayısı
        return np.array([pos_count, neg_count])

    def prepare_features(self, documents):
        """
        Prepares feature vectors for all documents by counting positive and negative words.
        Tüm belgeler için olumlu ve olumsuz kelime sayılarını kullanarak özellik vektörlerini hazırlar.
        
        :param documents: List of documents (Belgelerin listesi)
        :return: Feature matrix (Özellik matrisi)
        """
        return np.array([self.preprocess(doc) for doc in documents])

# Example sentiment classification using logistic regression
# Logistic regression kullanarak örnek duygu sınıflandırması
if __name__ == "__main__":
    # Example lists of positive and negative words (Örnek olumlu ve olumsuz kelime listeleri)
    positive_words = ["good", "great", "fantastic", "amazing", "positive", "love", "enjoy", "wonderful"]
    negative_words = ["bad", "terrible", "horrible", "negative", "hate", "dislike", "awful", "boring"]

    # Sample dataset of reviews and their corresponding sentiment labels
    # Yorumların ve onlara karşılık gelen duygu etiketlerinin örnek veri seti
    documents = [
        "I love this movie. It was fantastic!",
        "This film was terrible and boring.",
        "What an amazing performance!",
        "I really dislike the plot of this movie.",
        "The film is great and very enjoyable.",
        "The acting was horrible and I hated it."
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative (1 = Olumlu, 0 = Olumsuz)

    # Initialize sentiment classification (Duygu sınıflandırmasını başlat)
    sentiment_classifier = SentimentClassification(positive_words, negative_words)

    # Prepare feature matrix (Özellik matrisini hazırla)
    X = sentiment_classifier.prepare_features(documents)
    y = np.array(labels)

    # Split data into training and testing sets (Verileri eğitim ve test kümelerine ayır)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression classifier (Logistic regression sınıflandırıcısını başlat ve eğit)
    classifier = LogisticRegressionClassifier(learning_rate=0.01, epochs=1000)
    classifier.fit(X_train, y_train)

    # Predict on the test set (Test kümesi üzerinde tahmin yap)
    y_pred = classifier.predict(X_test)

    # Evaluate model performance (Model performansını değerlendir)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics (Değerlendirme metriklerini yazdır)
    print(f"Accuracy (Doğruluk): {accuracy:.2f}")
    print(f"Precision (Kesinlik): {precision:.2f}")
    print(f"Recall (Duyarlılık): {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
