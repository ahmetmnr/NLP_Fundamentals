from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
import nltk
from sentiment_analysis_with_negation import NaiveBayesWithNegation, load_model  # Sınıfı ve yükleme fonksiyonunu import edin

# Download necessary data for tokenization
# Tokenizasyon için gerekli verileri indir
nltk.download('punkt')

# Function to evaluate a trained Naive Bayes model (Eğitilmiş Naive Bayes modelini değerlendirmek için fonksiyon)
def evaluate_model(classifier, test_documents, test_labels):
    """
    Evaluate the performance of a Naive Bayes classifier using precision, recall, F1-score, and accuracy.
    Naive Bayes sınıflandırıcısının performansını precision, recall, F1-score ve accuracy kullanarak değerlendirin.
    
    :param classifier: Trained Naive Bayes classifier (Eğitilmiş Naive Bayes sınıflandırıcısı)
    :param test_documents: List of test documents (Test belgelerinin listesi)
    :param test_labels: List of actual labels corresponding to test documents (Test belgelerine karşılık gelen gerçek etiketler listesi)
    :return: Performance metrics (precision, recall, F1-score, accuracy) and confusion matrix
             Performans metrikleri (kesinlik, duyarlılık, F1-score, doğruluk) ve karışıklık matrisi
    """
    predicted_labels = []

    # Predict the label for each document (Her belge için etiketi tahmin et)
    for doc in test_documents:
        predicted_label = classifier.predict(doc)
        predicted_labels.append(predicted_label)

    # Calculate evaluation metrics (Değerlendirme metriklerini hesapla)
    precision = precision_score(test_labels, predicted_labels, pos_label="positive")
    recall = recall_score(test_labels, predicted_labels, pos_label="positive")
    f1 = f1_score(test_labels, predicted_labels, pos_label="positive")
    accuracy = accuracy_score(test_labels, predicted_labels)
    conf_matrix = confusion_matrix(test_labels, predicted_labels)

    # Print metrics (Metrikleri yazdır)
    print(f"Precision (Kesinlik): {precision:.2f}")
    print(f"Recall (Duyarlılık): {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"Accuracy (Doğruluk): {accuracy:.2f}")
    print("Confusion Matrix (Karışıklık Matrisi):")
    print(conf_matrix)

    return precision, recall, f1, accuracy, conf_matrix

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample test data (Test verisi örneği)
    test_documents = [
        "I love this movie. It was amazing!",
        "I don't like this film. It was boring.",
        "The acting was wonderful!",
        "The plot was terrible. I didn't enjoy it at all.",
        "This movie is fantastic.",
        "I wouldn't recommend this film."
    ]
    
    test_labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

    # Load the trained classifier (Eğitilmiş sınıflandırıcıyı yükle)
    classifier = load_model('naive_bayes_with_negation_model.pkl')  # Doğru sınıfı ve yükleme fonksiyonunu kullanarak modeli yükle

    # Evaluate the model (Modeli değerlendir)
    evaluate_model(classifier, test_documents, test_labels)
