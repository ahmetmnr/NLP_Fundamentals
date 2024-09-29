import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

class PPMIWeighting:
    def __init__(self, corpus, window_size=2):
        """
        Initializes the PPMI weighting class with a given corpus and context window size.
        Verilen bir corpus ve pencere boyutu ile PPMI ağırlıklandırma sınıfını başlatır.
        
        :param corpus: A list of documents (Dokümanlardan oluşan bir liste)
        :param window_size: Size of the context window for co-occurrence (Birlikte görünüm için pencere boyutu)
        """
        self.corpus = corpus  # Dokümanlar
        self.window_size = window_size  # Pencere boyutu
        self.co_occurrence_matrix = None  # Birlikte görünüm matrisi
        self.ppmi_matrix = None  # PPMI matrisi
        self.vocab = None  # Kelime dağarcığı
    
    def build_co_occurrence_matrix(self):
        """
        Builds the co-occurrence matrix from the corpus using the given window size.
        Verilen pencere boyutunu kullanarak corpus'tan birlikte görünüm matrisini oluşturur.
        
        :return: Co-occurrence matrix (Birlikte görünüm matrisi)
        """
        vectorizer = CountVectorizer()
        tokenized_corpus = [doc.split() for doc in self.corpus]
        vocab = vectorizer.fit(self.corpus).get_feature_names_out()
        self.vocab = vocab
        vocab_size = len(vocab)
        
        # Kelime sayıları (kelimeleri numaralandırmak için)
        word_index = {word: idx for idx, word in enumerate(vocab)}
        
        # Boş birlikte görünüm matrisi
        co_occurrence = np.zeros((vocab_size, vocab_size))
        
        # Her dokümanda kelimeler arası ilişkileri pencere boyutuna göre incele
        for doc in tokenized_corpus:
            for i, word in enumerate(doc):
                word_id = word_index.get(word)
                if word_id is None:
                    continue  # Kelime kelime dağarcığında yoksa atla
                start = max(0, i - self.window_size)
                end = min(len(doc), i + self.window_size + 1)
                
                # Pencere içindeki kelimeler ile ilişkiyi kaydet
                for j in range(start, end):
                    if i != j:  # Kendi kendine birlikte görünüm sayılmaz
                        context_word_id = word_index.get(doc[j])
                        if context_word_id is not None:
                            co_occurrence[word_id, context_word_id] += 1
        
        self.co_occurrence_matrix = co_occurrence
        return co_occurrence
    
    def compute_ppmi_matrix(self):
        """
        Computes the Positive Pointwise Mutual Information (PPMI) matrix from the co-occurrence matrix.
        Birlikte görünüm matrisinden Positive Pointwise Mutual Information (PPMI) matrisini hesaplar.
        
        :return: PPMI matrix (PPMI matrisi)
        """
        if self.co_occurrence_matrix is None:
            self.build_co_occurrence_matrix()  # Eğer birlikte görünüm matrisi yoksa oluştur
        
        total_count = np.sum(self.co_occurrence_matrix)
        word_count = np.sum(self.co_occurrence_matrix, axis=1)
        ppmi_matrix = np.zeros_like(self.co_occurrence_matrix)
        
        for i in range(self.co_occurrence_matrix.shape[0]):
            for j in range(self.co_occurrence_matrix.shape[1]):
                if self.co_occurrence_matrix[i, j] > 0:
                    p_ij = self.co_occurrence_matrix[i, j] / total_count
                    p_i = word_count[i] / total_count
                    p_j = word_count[j] / total_count
                    pmi = np.log2(p_ij / (p_i * p_j))
                    ppmi_matrix[i, j] = max(pmi, 0)  # PPMI: PMI'nin negatif olmaması için sıfırla sınırla
        
        self.ppmi_matrix = ppmi_matrix
        return ppmi_matrix
    
    def get_top_associated_words(self, word, top_n=5):
        """
        Retrieves the top N most associated words to the given word based on PPMI.
        Verilen bir kelimeye en fazla ilişkilendirilen N kelimeyi PPMI'ya göre getirir.
        
        :param word: The word to find associated words for (İlişkilendirilecek kelime)
        :param top_n: The number of top associated words to retrieve (En fazla ilişkilendirilen N kelime)
        :return: A list of top N associated words and their PPMI scores (İlişkilendirilen N kelime ve PPMI skorları)
        """
        if self.ppmi_matrix is None:
            self.compute_ppmi_matrix()  # Eğer PPMI matrisi yoksa oluştur
        
        if word not in self.vocab:
            raise ValueError(f"The word '{word}' is not in the corpus vocabulary.")
        
        word_idx = np.where(self.vocab == word)[0][0]
        ppmi_scores = self.ppmi_matrix[word_idx]
        top_indices = np.argsort(ppmi_scores)[::-1][:top_n]
        top_words = [(self.vocab[i], ppmi_scores[i]) for i in top_indices]
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
    
    # Initialize the PPMI weighting (PPMI ağırlıklandırmayı başlat)
    ppmi_weighting = PPMIWeighting(corpus, window_size=2)
    
    # Build the co-occurrence matrix (Birlikte görünüm matrisini oluştur)
    co_occurrence_matrix = ppmi_weighting.build_co_occurrence_matrix()
    print("Co-occurrence Matrix:")
    print(co_occurrence_matrix)
    
    # Compute the PPMI matrix (PPMI matrisini hesapla)
    ppmi_matrix = ppmi_weighting.compute_ppmi_matrix()
    print("\nPPMI Matrix:")
    print(ppmi_matrix)
    
    # Get the top 3 associated words for the word 'machine' (En çok ilişkili 3 kelimeyi 'machine' için getir)
    top_words = ppmi_weighting.get_top_associated_words('machine', top_n=3)
    print("\nTop 3 associated words for 'machine':")
    print(top_words)
