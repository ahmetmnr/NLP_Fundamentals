import numpy as np
from scipy.spatial.distance import cosine

class StaticEmbeddingsEvaluation:
    def __init__(self, embeddings, word_to_index):
        """
        Initializes the evaluation class with embeddings and word-to-index mapping.
        Gömme vektörleri ve kelime-indeks eşlemeleri ile değerlendirme sınıfını başlatır.
        
        :param embeddings: Word embeddings matrix (Kelime gömme matrisi)
        :param word_to_index: Dictionary mapping words to their indices (Kelime-indeks eşlemesi yapan sözlük)
        """
        self.embeddings = embeddings
        self.word_to_index = word_to_index
    
    def evaluate_similarity(self, word_pairs, human_ratings):
        """
        Evaluates word similarity by comparing cosine similarity of word embeddings with human similarity ratings.
        İnsan değerlendirmeleri ile kelime gömme vektörlerinin cosine similarity skorlarını karşılaştırarak kelime benzerliğini değerlendirir.
        
        :param word_pairs: List of tuples containing word pairs (Kelime çiftlerinden oluşan liste)
        :param human_ratings: List of human similarity ratings for the word pairs (Kelime çiftleri için insan benzerlik dereceleri)
        :return: Correlation between human ratings and embedding similarity scores (İnsan dereceleri ile gömme benzerlik skorları arasındaki korelasyon)
        """
        similarities = []
        
        for word1, word2 in word_pairs:
            if word1 in self.word_to_index and word2 in self.word_to_index:
                vec1 = self.embeddings[self.word_to_index[word1]]
                vec2 = self.embeddings[self.word_to_index[word2]]
                
                # Cosine similarity hesapla
                similarity = 1 - cosine(vec1, vec2)
                similarities.append(similarity)
            else:
                similarities.append(0.0)  # Eğer kelime gömme vektörü yoksa, sıfır ekle
                
        # İnsan dereceleri ile modelin cosine similarity skorları arasındaki korelasyonu hesapla
        correlation = np.corrcoef(human_ratings, similarities)[0, 1]
        return correlation
    
    def evaluate_analogy(self, analogy_questions):
        """
        Evaluates word embeddings on analogy tasks like "man:king :: woman:queen".
        Kelime gömme vektörlerini analoji görevlerinde değerlendirir (ör. "adam:kral :: kadın:kraliçe").
        
        :param analogy_questions: List of analogy questions (Analogy sorularından oluşan liste)
        :return: Accuracy of the embeddings on analogy tasks (Analogy görevlerindeki doğruluk oranı)
        """
        correct = 0
        total = len(analogy_questions)
        
        for a, b, c, expected in analogy_questions:
            if a in self.word_to_index and b in self.word_to_index and c in self.word_to_index:
                vec_a = self.embeddings[self.word_to_index[a]]
                vec_b = self.embeddings[self.word_to_index[b]]
                vec_c = self.embeddings[self.word_to_index[c]]
                
                # "vec_d = vec_b - vec_a + vec_c" formülü ile d kelimesi tahmin edilir
                vec_d = vec_b - vec_a + vec_c
                
                # En yakın kelimeyi bul (cosine similarity ile)
                max_similarity = -1
                best_word = None
                for word, idx in self.word_to_index.items():
                    if word not in [a, b, c]:
                        vec_word = self.embeddings[idx]
                        similarity = 1 - cosine(vec_d, vec_word)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_word = word
                
                if best_word == expected:
                    correct += 1
        
        return correct / total if total > 0 else 0

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Örnek kelime gömmeleri ve kelime-indeks eşlemeleri
    vocab = ['man', 'woman', 'king', 'queen', 'apple', 'orange', 'fruit']
    vocab_size = len(vocab)
    embedding_dim = 100
    embeddings = np.random.uniform(-1, 1, (vocab_size, embedding_dim))  # Rastgele gömme vektörleri (örnek için)
    
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    # Kelime Benzerliği Görevi
    word_pairs = [('man', 'woman'), ('king', 'queen'), ('apple', 'orange')]
    human_ratings = [9.0, 9.5, 8.0]  # İnsan benzerlik dereceleri (örnek veri)
    
    # Analogy Görevi (adam:kral :: kadın:kraliçe)
    analogy_questions = [
        ('man', 'king', 'woman', 'queen'),
        ('apple', 'fruit', 'orange', 'fruit')
    ]
    
    # Değerlendirme sınıfını başlat
    evaluator = StaticEmbeddingsEvaluation(embeddings, word_to_index)
    
    # Kelime benzerliği görevini değerlendir
    similarity_correlation = evaluator.evaluate_similarity(word_pairs, human_ratings)
    print(f"Word Similarity Correlation: {similarity_correlation:.4f}")
    
    # Analogy görevini değerlendir
    analogy_accuracy = evaluator.evaluate_analogy(analogy_questions)
    print(f"Analogy Task Accuracy: {analogy_accuracy:.4f}")
