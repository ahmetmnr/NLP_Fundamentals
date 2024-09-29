import numpy as np
from scipy.spatial.distance import cosine

class BiasDetectionEmbeddings:
    def __init__(self, embeddings, word_to_index):
        """
        Initializes the bias detection class with embeddings and word-to-index mapping.
        Gömme vektörleri ve kelime-indeks eşlemeleri ile önyargı tespit sınıfını başlatır.
        
        :param embeddings: Word embeddings matrix (Kelime gömme matrisi)
        :param word_to_index: Dictionary mapping words to their indices (Kelime-indeks eşlemesi yapan sözlük)
        """
        self.embeddings = embeddings
        self.word_to_index = word_to_index

    def get_embedding(self, word):
        """
        Retrieves the embedding vector for a given word.
        Verilen bir kelimenin gömme vektörünü getirir.
        
        :param word: The word to retrieve the embedding for (Gömme vektörünü almak istediğiniz kelime)
        :return: The embedding vector (Gömme vektörü)
        """
        return self.embeddings[self.word_to_index[word]]

    def compute_bias_subspace(self, word_pairs):
        """
        Computes the gender or racial bias subspace using word pairs.
        Kelime çiftleri kullanarak cinsiyet veya ırk önyargısı alt uzayını hesaplar.
        
        :param word_pairs: List of word pairs to define the bias subspace (Önyargı alt uzayını tanımlayan kelime çiftleri)
        :return: Bias subspace vector (Önyargı alt uzayı vektörü)
        """
        vectors = []
        
        for word1, word2 in word_pairs:
            if word1 in self.word_to_index and word2 in self.word_to_index:
                vec1 = self.get_embedding(word1)
                vec2 = self.get_embedding(word2)
                vectors.append(vec1 - vec2)
        
        # Ortalama vektörü alınarak alt uzayı hesapla
        bias_subspace = np.mean(vectors, axis=0)
        return bias_subspace
    
    def measure_bias(self, word, bias_subspace):
        """
        Measures the bias of a word by projecting its embedding onto the bias subspace.
        Bir kelimenin önyargısını, gömme vektörünü önyargı alt uzayına projekte ederek ölçer.
        
        :param word: The word to measure bias for (Önyargısı ölçülecek kelime)
        :param bias_subspace: The bias subspace vector (Önyargı alt uzayı vektörü)
        :return: Bias score (Önyargı skoru)
        """
        word_vec = self.get_embedding(word)
        bias_score = np.dot(word_vec, bias_subspace) / (np.linalg.norm(word_vec) * np.linalg.norm(bias_subspace))
        return bias_score
    
    def weat_test(self, target_words_1, target_words_2, attribute_words_1, attribute_words_2):
        """
        Implements the Word Embedding Association Test (WEAT) to measure bias in word embeddings.
        Kelime gömmelerindeki önyargıyı ölçmek için Kelime Gömme İlişkilendirme Testi'ni (WEAT) uygular.
        
        :param target_words_1: First set of target words (İlk hedef kelime kümesi)
        :param target_words_2: Second set of target words (İkinci hedef kelime kümesi)
        :param attribute_words_1: First set of attribute words (İlk özellik kelime kümesi)
        :param attribute_words_2: Second set of attribute words (İkinci özellik kelime kümesi)
        :return: WEAT score (WEAT skoru)
        """
        def association_score(word, attribute_set):
            """
            Calculates the association score between a word and an attribute set.
            Bir kelime ile bir özellik kümesi arasındaki ilişki skorunu hesaplar.
            """
            word_vec = self.get_embedding(word)
            scores = [1 - cosine(word_vec, self.get_embedding(attr_word)) for attr_word in attribute_set]
            return np.mean(scores)
        
        score_1 = sum([association_score(w, attribute_words_1) - association_score(w, attribute_words_2) for w in target_words_1])
        score_2 = sum([association_score(w, attribute_words_1) - association_score(w, attribute_words_2) for w in target_words_2])
        
        return score_1 - score_2

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Örnek kelime gömmeleri ve kelime-indeks eşlemeleri
    vocab = ['man', 'woman', 'king', 'queen', 'doctor', 'nurse', 'he', 'she', 'engineer', 'teacher']
    vocab_size = len(vocab)
    embedding_dim = 100
    embeddings = np.random.uniform(-1, 1, (vocab_size, embedding_dim))  # Rastgele gömme vektörleri (örnek için)
    
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    # Cinsiyet önyargısı alt uzayı tanımlamak için kelime çiftleri (ör. "he - she", "man - woman")
    gender_word_pairs = [('he', 'she'), ('man', 'woman')]
    
    # Değerlendirme sınıfını başlat
    evaluator = BiasDetectionEmbeddings(embeddings, word_to_index)
    
    # Cinsiyet önyargısı alt uzayı hesapla
    gender_bias_subspace = evaluator.compute_bias_subspace(gender_word_pairs)
    
    # "doctor" ve "nurse" kelimeleri için önyargı ölç
    doctor_bias = evaluator.measure_bias('doctor', gender_bias_subspace)
    nurse_bias = evaluator.measure_bias('nurse', gender_bias_subspace)
    print(f"Doctor Bias Score: {doctor_bias:.4f}")
    print(f"Nurse Bias Score: {nurse_bias:.4f}")
    
    # WEAT testi (Hedef kelimeler: "engineer", "teacher"; Özellik kelimeler: "he", "she")
    target_words_1 = ['engineer']
    target_words_2 = ['teacher']
    attribute_words_1 = ['he']
    attribute_words_2 = ['she']
    
    weat_score = evaluator.weat_test(target_words_1, target_words_2, attribute_words_1, attribute_words_2)
    print(f"WEAT Score: {weat_score:.4f}")
