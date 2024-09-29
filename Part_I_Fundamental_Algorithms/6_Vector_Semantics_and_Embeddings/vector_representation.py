import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jaccard

class VectorRepresentation:
    def __init__(self, corpus):
        """
        Initializes the vector representation class with a given corpus.
        Verilen bir corpus ile vektör temsili sınıfını başlatır.
        
        :param corpus: A list of documents (Dokümanlardan oluşan bir liste)
        """
        self.corpus = corpus  # Dokümanlar
        self.term_doc_matrix = None  # Terim-Doküman Matrisi
        self.term_term_matrix = None  # Terim-Terim Matrisi
    
    def create_term_document_matrix(self):
        """
        Creates a term-document matrix from the corpus using bag-of-words.
        Bag-of-words kullanarak corpus'tan terim-doküman matrisi oluşturur.
        
        :return: Term-document matrix (Terim-doküman matrisi)
        """
        vectorizer = CountVectorizer()
        self.term_doc_matrix = vectorizer.fit_transform(self.corpus).toarray()
        return self.term_doc_matrix
    
    def create_term_term_matrix(self):
        """
        Creates a term-term co-occurrence matrix.
        Terim-terim birlikte görünüm matrisini oluşturur.
        
        :return: Term-term matrix (Terim-terim matrisi)
        """
        if self.term_doc_matrix is None:
            self.create_term_document_matrix()  # Eğer terim-doküman matrisi yoksa oluştur
        
        # Terim-terim matrisi, terim-doküman matrisinin çarpımı ile elde edilir
        self.term_term_matrix = np.dot(self.term_doc_matrix.T, self.term_doc_matrix)
        return self.term_term_matrix
    
    def cosine_similarity(self, matrix):
        """
        Computes cosine similarity between rows of the given matrix.
        Verilen matrisin satırları arasında cosine similarity hesaplar.
        
        :param matrix: The input matrix (Giriş matrisi)
        :return: Cosine similarity matrix (Cosine similarity matrisi)
        """
        return cosine_similarity(matrix)
    
    def euclidean_distance(self, matrix):
        """
        Computes Euclidean distance between rows of the given matrix.
        Verilen matrisin satırları arasında Euclidean mesafe hesaplar.
        
        :param matrix: The input matrix (Giriş matrisi)
        :return: Euclidean distance matrix (Euclidean mesafe matrisi)
        """
        return euclidean_distances(matrix)
    
    def manhattan_distance(self, matrix):
        """
        Computes Manhattan distance between rows of the given matrix.
        Verilen matrisin satırları arasında Manhattan mesafesini hesaplar.
        
        :param matrix: The input matrix (Giriş matrisi)
        :return: Manhattan distance matrix (Manhattan mesafe matrisi)
        """
        return manhattan_distances(matrix)
    
    def jaccard_similarity(self, matrix):
        """
        Computes Jaccard similarity between rows of the given matrix.
        Verilen matrisin satırları arasında Jaccard similarity hesaplar.
        
        :param matrix: The input matrix (Giriş matrisi)
        :return: Jaccard similarity matrix (Jaccard similarity matrisi)
        """
        n_rows = matrix.shape[0]
        jaccard_sim_matrix = np.zeros((n_rows, n_rows))
        for i in range(n_rows):
            for j in range(n_rows):
                jaccard_sim_matrix[i, j] = 1 - jaccard(matrix[i], matrix[j])  # Jaccard benzerliği ters çevirilir
        return jaccard_sim_matrix
    
    def get_top_similar_documents(self, doc_index, metric="cosine", top_n=5):
        """
        Retrieves the top N most similar documents to the given document using the specified similarity metric.
        Verilen bir dokümana belirtilen benzerlik metriğini kullanarak en benzer N dokümanı getirir.
        
        :param doc_index: The index of the document in the corpus (Dokümanın indeks numarası)
        :param metric: Similarity metric to use (Benzerlik metriği: 'cosine', 'euclidean', 'manhattan', 'jaccard')
        :param top_n: The number of top similar documents to retrieve (En benzer N doküman sayısı)
        :return: A list of top N similar document indices (En benzer N dokümanın indeks numaraları)
        """
        if self.term_doc_matrix is None:
            self.create_term_document_matrix()  # Eğer terim-doküman matrisi yoksa oluştur
        
        if metric == "cosine":
            sim_matrix = self.cosine_similarity(self.term_doc_matrix)
        elif metric == "euclidean":
            sim_matrix = self.euclidean_distance(self.term_doc_matrix)
        elif metric == "manhattan":
            sim_matrix = self.manhattan_distance(self.term_doc_matrix)
        elif metric == "jaccard":
            sim_matrix = self.jaccard_similarity(self.term_doc_matrix)
        else:
            raise ValueError("Invalid similarity metric. Choose from 'cosine', 'euclidean', 'manhattan', or 'jaccard'.")
        
        # Sort by similarity, ignore the document itself (En benzer dokümanları sırala, kendi dokümanını hariç tut)
        similar_docs = np.argsort(-sim_matrix[doc_index] if metric == "cosine" else sim_matrix[doc_index])[1:top_n+1]
        return similar_docs

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
    
    # Initialize the vector representation (Vektör temsili başlat)
    vector_rep = VectorRepresentation(corpus)
    
    # Create term-document matrix (Terim-doküman matrisi oluştur)
    term_doc_matrix = vector_rep.create_term_document_matrix()
    print("Term-Document Matrix:")
    print(term_doc_matrix)
    
    # Cosine similarity between documents (Dokümanlar arası cosine similarity)
    cosine_sim_matrix = vector_rep.cosine_similarity(term_doc_matrix)
    print("\nCosine Similarity Between Documents:")
    print(cosine_sim_matrix)
    
    # Euclidean distance between documents (Dokümanlar arası Euclidean mesafe)
    euclidean_dist_matrix = vector_rep.euclidean_distance(term_doc_matrix)
    print("\nEuclidean Distance Between Documents:")
    print(euclidean_dist_matrix)
    
    # Manhattan distance between documents (Dokümanlar arası Manhattan mesafe)
    manhattan_dist_matrix = vector_rep.manhattan_distance(term_doc_matrix)
    print("\nManhattan Distance Between Documents:")
    print(manhattan_dist_matrix)
    
    # Jaccard similarity between documents (Dokümanlar arası Jaccard similarity)
    jaccard_sim_matrix = vector_rep.jaccard_similarity(term_doc_matrix)
    print("\nJaccard Similarity Between Documents:")
    print(jaccard_sim_matrix)
    
    # Get top 3 similar documents to the first document using cosine similarity (Cosine similarity ile en benzer 3 doküman)
    top_similar_docs_cosine = vector_rep.get_top_similar_documents(0, metric="cosine", top_n=3)
    print("\nTop 3 similar documents to document 0 (Cosine Similarity):")
    print(top_similar_docs_cosine)

    # Get top 3 similar documents using Euclidean distance (Euclidean mesafe ile en benzer 3 doküman)
    top_similar_docs_euclidean = vector_rep.get_top_similar_documents(0, metric="euclidean", top_n=3)
    print("\nTop 3 similar documents to document 0 (Euclidean Distance):")
    print(top_similar_docs_euclidean)
