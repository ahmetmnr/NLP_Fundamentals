import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFWeighting:
    def __init__(self, corpus):
        """
        Initializes the TF-IDF weighting class with a given corpus.
        Verilen bir corpus ile TF-IDF ağırlıklandırma sınıfını başlatır.
        
        :param corpus: A list of documents (Dokümanlardan oluşan bir liste)
        """
        self.corpus = corpus  # Dokümanlar
        self.tfidf_matrix = None  # TF-IDF matrisi
        self.feature_names = None  # Özellik isimleri (kelimeler)
    
    def apply_tfidf_weighting(self):
        """
        Applies TF-IDF weighting to the corpus and creates a TF-IDF matrix.
        Corpus'a TF-IDF ağırlıklandırma uygular ve bir TF-IDF matrisi oluşturur.
        
        :return: TF-IDF matrix (TF-IDF matrisi)
        """
        vectorizer = TfidfVectorizer()
        self.tfidf_matrix = vectorizer.fit_transform(self.corpus).toarray()
        self.feature_names = vectorizer.get_feature_names_out()
        return self.tfidf_matrix
    
    def get_feature_names(self):
        """
        Retrieves the feature names (words) from the TF-IDF vectorizer.
        TF-IDF vektörleştiricisinden özellik isimlerini (kelimeleri) getirir.
        
        :return: A list of feature names (Özellik isimleri - kelimeler)
        """
        if self.feature_names is None:
            raise ValueError("TF-IDF matrix has not been computed yet. Call apply_tfidf_weighting first.")
        return self.feature_names
    
    def get_tfidf_for_word(self, word):
        """
        Retrieves the TF-IDF scores for a specific word across all documents.
        Belirli bir kelimenin tüm dokümanlar üzerindeki TF-IDF skorlarını getirir.
        
        :param word: The word to retrieve TF-IDF scores for (TF-IDF skorlarını almak istediğiniz kelime)
        :return: TF-IDF scores for the word (Kelime için TF-IDF skorları)
        """
        if self.feature_names is None:
            raise ValueError("TF-IDF matrix has not been computed yet. Call apply_tfidf_weighting first.")
        
        if word not in self.feature_names:
            raise ValueError(f"The word '{word}' is not in the corpus vocabulary.")
        
        word_index = np.where(self.feature_names == word)[0][0]
        return self.tfidf_matrix[:, word_index]
    
    def get_top_words_for_document(self, doc_index, top_n=5):
        """
        Retrieves the top N words with the highest TF-IDF scores for a specific document.
        Belirli bir doküman için en yüksek TF-IDF skoruna sahip en iyi N kelimeyi getirir.
        
        :param doc_index: The index of the document (Dokümanın indeks numarası)
        :param top_n: The number of top words to retrieve (En yüksek skorlu N kelime)
        :return: A list of top N words and their TF-IDF scores (En iyi N kelime ve TF-IDF skorları)
        """
        if self.tfidf_matrix is None:
            self.apply_tfidf_weighting()  # Eğer TF-IDF matrisi yoksa oluştur
        
        tfidf_scores = self.tfidf_matrix[doc_index]
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
        top_words = [(self.feature_names[i], tfidf_scores[i]) for i in top_indices]
        return top_words

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Example corpus of documents (Örnek doküman corpus'u)
    corpus = [
        "I love machine learning",
        "Machine learning is great",
        "Natural language processing is a part of machine learning",
        "I enjoy learning about natural language processing",
        "Machine learning can solve many problems"
    ]
    
    # Initialize the TF-IDF weighting (TF-IDF ağırlıklandırmayı başlat)
    tfidf_weighting = TFIDFWeighting(corpus)
    
    # Apply TF-IDF weighting to the corpus (Corpus'a TF-IDF ağırlıklandırmayı uygula)
    tfidf_matrix = tfidf_weighting.apply_tfidf_weighting()
    print("TF-IDF Matrix:")
    print(tfidf_matrix)
    
    # Get feature names (Kelime listesi)
    feature_names = tfidf_weighting.get_feature_names()
    print("\nFeature Names (Words):")
    print(feature_names)
    
    # Get TF-IDF scores for a specific word (Belirli bir kelime için TF-IDF skorlarını al)
    word = "learning"
    tfidf_scores_for_word = tfidf_weighting.get_tfidf_for_word(word)
    print(f"\nTF-IDF scores for the word '{word}':")
    print(tfidf_scores_for_word)
    
    # Get top 3 words for the first document (İlk doküman için en yüksek TF-IDF skoru olan 3 kelimeyi getir)
    top_words_for_doc = tfidf_weighting.get_top_words_for_document(0, top_n=3)
    print("\nTop 3 words for document 0:")
    print(top_words_for_doc)
