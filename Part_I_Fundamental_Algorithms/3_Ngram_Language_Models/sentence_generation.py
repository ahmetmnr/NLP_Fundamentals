import random
from collections import defaultdict
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Sample text corpus for training (Eğitim için örnek metin corpus'u)
corpus = "I love natural language processing. Natural language processing is fun. I love machine learning."

# Tokenize the text into words (Metni kelimelere böl)
nltk.download('punkt')
tokens = word_tokenize(corpus.lower())

# Function to calculate n-grams (N-gram'ları hesaplayan fonksiyon)
def calculate_ngrams(tokens, n):
    """Calculate n-grams from the tokenized text. (Tokenize edilmiş metinden n-gram'ları hesapla.)"""
    return list(ngrams(tokens, n))

# Unigrams, Bigrams, Trigrams (Unigram'lar, Bigram'lar, Trigram'lar)
bigrams = calculate_ngrams(tokens, 2)
trigrams = calculate_ngrams(tokens, 3)

# Function to calculate N-gram frequencies (N-gram frekanslarını hesaplayan fonksiyon)
def calculate_ngram_frequencies(n_grams):
    """Calculate the frequency of each N-gram. (Her N-gram'ın frekansını hesapla.)"""
    ngram_freqs = defaultdict(int)
    for ngram in n_grams:
        ngram_freqs[ngram] += 1
    return ngram_freqs

# Calculate frequencies for bigrams and trigrams (Bigram ve trigram frekanslarını hesaplayın)
bigram_freqs = calculate_ngram_frequencies(bigrams)
trigram_freqs = calculate_ngram_frequencies(trigrams)

# Function to generate a sentence using a bigram or trigram model
# Bigram veya trigram modeli kullanarak bir cümle oluşturma fonksiyonu
def generate_sentence(ngram_probs, n, max_words=15, start_words=None):
    """ 
    Generate a sentence using the N-gram model. 
    (N-gram modelini kullanarak bir cümle oluştur.)
    """
    
    # Start with given words or pick random starting N-gram (Başlangıç kelimeleri veya rastgele bir N-gram seçilir)
    if start_words:
        current_words = tuple(start_words)
    else:
        current_words = random.choice(list(ngram_probs.keys()))  # Randomly choose starting n-gram (Başlangıç için rastgele bir N-gram seçilir)
    
    sentence = list(current_words)

    # Generate up to max_words (max_words sayısına kadar cümle oluştur)
    for _ in range(max_words - n + 1):
        possible_ngrams = [ngram for ngram in ngram_probs.keys() if ngram[:-1] == current_words]
        
        if not possible_ngrams:
            break  # No valid continuation, stop the generation (Geçerli bir devam yoksa dur)
        
        # Choose next word based on probabilities (Olasılıklara göre bir sonraki kelimeyi seç)
        next_ngram = random.choices(possible_ngrams, weights=[ngram_probs[ngram] for ngram in possible_ngrams], k=1)[0]
        sentence.append(next_ngram[-1])
        current_words = next_ngram[1:]  # Update current words for next step (Bir sonraki adım için kelimeleri güncelle)
    
    return ' '.join(sentence)

# Example bigram model (Örnek bigram modeli)
bigram_probs = {bigram: freq / sum(bigram_freqs.values()) for bigram, freq in bigram_freqs.items()}

# Example trigram model (Örnek trigram modeli)
trigram_probs = {trigram: freq / sum(trigram_freqs.values()) for trigram, freq in trigram_freqs.items()}

# Generate sentences using bigram and trigram models (Bigram ve trigram modelleri kullanarak cümle üret)
bigram_sentence = generate_sentence(bigram_probs, 2)
trigram_sentence = generate_sentence(trigram_probs, 3)

# Print generated sentences (Oluşturulan cümleleri yazdır)
print("Generated Bigram Sentence: ", bigram_sentence)
print("Generated Trigram Sentence: ", trigram_sentence)
