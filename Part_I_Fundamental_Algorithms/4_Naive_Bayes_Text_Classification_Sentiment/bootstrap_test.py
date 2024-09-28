import numpy as np
from sklearn.metrics import accuracy_score
from sentiment_analysis_with_negation import NaiveBayesWithNegation
from binary_naive_bayes import BinaryNaiveBayesClassifier  # İki farklı modelin karşılaştırılması için ikisini import ediyoruz
import random
import nltk

# Download necessary data for tokenization
# Tokenizasyon için gerekli verileri indir
nltk.download('punkt')

# Bootstrap test function to compare two models (İki modeli karşılaştırmak için bootstrap testi fonksiyonu)
def bootstrap_test(model1, model2, documents, labels, num_iterations=1000):
    """
    Perform Bootstrap test to compare two Naive Bayes models' performance.
    İki Naive Bayes modelinin performansını karşılaştırmak için Bootstrap testi gerçekleştirir.
    
    :param model1: First trained model (Birinci eğitimli model)
    :param model2: Second trained model (İkinci eğitimli model)
    :param documents: List of documents (strings) to test the models (Modelleri test etmek için belge listesi)
    :param labels: List of actual labels corresponding to the documents (Belgelere karşılık gelen gerçek etiketler)
    :param num_iterations: Number of bootstrap iterations (Bootstrap tekrar sayısı)
    :return: Bootstrap results indicating whether the difference is statistically significant
             Farkın istatistiksel olarak anlamlı olup olmadığını gösteren bootstrap sonuçları
    """
    n = len(documents)
    original_diff = calculate_model_diff(model1, model2, documents, labels)
    print(f"Original Difference in Accuracy: {original_diff:.4f}")

    bootstrap_diffs = []

    # Perform bootstrap sampling (Bootstrap örneklemesi yapın)
    for i in range(num_iterations):
        # Sample with replacement (Yerine koyarak örnekleme yapın)
        indices = [random.randint(0, n-1) for _ in range(n)]
        bootstrap_docs = [documents[i] for i in indices]
        bootstrap_labels = [labels[i] for i in indices]

        # Calculate performance difference on the bootstrap sample (Bootstrap örneği üzerinde performans farkını hesaplayın)
        bootstrap_diff = calculate_model_diff(model1, model2, bootstrap_docs, bootstrap_labels)
        bootstrap_diffs.append(bootstrap_diff)

    # Calculate p-value: proportion of bootstrap samples where model2 outperforms model1
    # P-değerini hesaplayın: model2'nin model1'den daha iyi performans gösterdiği bootstrap örneklerinin oranı
    p_value = np.mean([1 if diff >= original_diff else 0 for diff in bootstrap_diffs])
    print(f"p-value: {p_value:.4f}")

    # Check significance (Anlamlılık kontrolü yapın)
    if p_value < 0.05:
        print("The performance difference is statistically significant.")
    else:
        print("The performance difference is NOT statistically significant.")
    
    return p_value

# Function to calculate the difference in accuracy between two models (İki model arasındaki doğruluk farkını hesaplayan fonksiyon)
def calculate_model_diff(model1, model2, documents, labels):
    """
    Calculate the accuracy difference between two models on the same dataset.
    Aynı veri kümesinde iki model arasındaki doğruluk farkını hesaplar.
    
    :param model1: First trained model (Birinci eğitimli model)
    :param model2: Second trained model (İkinci eğitimli model)
    :param documents: List of documents to test the models (Modelleri test etmek için belge listesi)
    :param labels: List of actual labels (Gerçek etiketler)
    :return: The difference in accuracy between the two models (İki model arasındaki doğruluk farkı)
    """
    # Predict with model 1 (Model 1 ile tahmin yap)
    model1_predictions = [model1.predict(doc) for doc in documents]
    model1_accuracy = accuracy_score(labels, model1_predictions)

    # Predict with model 2 (Model 2 ile tahmin yap)
    model2_predictions = [model2.predict(doc) for doc in documents]
    model2_accuracy = accuracy_score(labels, model2_predictions)

    # Return the difference in accuracy (Doğruluk farkını döndür)
    return model1_accuracy - model2_accuracy

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample documents and labels (Örnek belgeler ve etiketler)
    documents = [
        "I love this movie. It is fantastic!",
        "I don't like this film. It was boring.",
        "The acting was wonderful!",
        "The plot was terrible. I didn't enjoy it.",
        "This movie is great.",
        "I wouldn't recommend this movie.",
        "I like the direction but not the acting.",
        "The film was absolutely amazing and enjoyable.",
        "It was a disaster. I hated it.",
        "What a fantastic story!"
    ]

    labels = ["positive", "negative", "positive", "negative", "positive", "negative", "negative", "positive", "negative", "positive"]

    # Train model 1 (NaiveBayesWithNegation) (Model 1'i eğitin)
    model1 = NaiveBayesWithNegation()
    model1.train(documents, labels)

    # Train model 2 (BinaryNaiveBayesClassifier) (Model 2'yi eğitin)
    model2 = BinaryNaiveBayesClassifier()
    model2.train(documents, labels)

    # Perform bootstrap test (Bootstrap testini gerçekleştirin)
    bootstrap_test(model1, model2, documents, labels, num_iterations=1000)
