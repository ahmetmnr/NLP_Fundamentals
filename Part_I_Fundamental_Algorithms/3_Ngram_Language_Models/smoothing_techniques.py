from collections import defaultdict
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Sample text corpus (Örnek metin corpus'u)
corpus = "I love natural language processing. Natural language processing is fun. I love machine learning."

# Tokenize the text into words (Metni kelimelere böl)
nltk.download('punkt')
tokens = word_tokenize(corpus.lower())

# Function to calculate n-grams (N-gram'ları hesaplayan fonksiyon)
def calculate_ngrams(tokens, n):
    """Calculate n-grams from the tokenized text. (Tokenize edilmiş metinden n-gram'ları hesapla.)"""
    return list(ngrams(tokens, n))

# Unigrams, Bigrams, Trigrams (Unigram'lar, Bigram'lar, Trigram'lar)
unigrams = calculate_ngrams(tokens, 1)
bigrams = calculate_ngrams(tokens, 2)
trigrams = calculate_ngrams(tokens, 3)

# Function to calculate N-gram frequencies (N-gram frekanslarını hesaplayan fonksiyon)
def calculate_ngram_frequencies(n_grams):
    """Calculate the frequency of each N-gram. (Her N-gram'ın frekansını hesapla.)"""
    ngram_freqs = defaultdict(int)
    for ngram in n_grams:
        ngram_freqs[ngram] += 1
    return ngram_freqs

# Calculate frequencies for unigrams, bigrams, and trigrams (Unigram, bigram ve trigram frekanslarını hesaplayın)
unigram_freqs = calculate_ngram_frequencies(unigrams)
bigram_freqs = calculate_ngram_frequencies(bigrams)
trigram_freqs = calculate_ngram_frequencies(trigrams)

# Vocabulary size (Kelime dağarcığının büyüklüğü)
vocab_size = len(set(tokens))

# Total count of unigrams (Unigram'ların toplam sayısı)
total_unigrams = sum(unigram_freqs.values())

# 1. Laplace Smoothing (Add-1 Smoothing)
# Laplace smoothing, adds 1 to each N-gram count to prevent zero probabilities.
# Laplace smoothing, sıfır olasılıkları engellemek için her N-gram sayısına 1 ekler.
def laplace_smoothing(ngram_freqs, lower_order_freqs, vocab_size):
    smoothed_probs = {}
    for ngram, count in ngram_freqs.items():
        # P(w_n | w_{n-1}) = (Count(w_{n-1}, w_n) + 1) / (Count(w_{n-1}) + V)
        smoothed_probs[ngram] = (count + 1) / (lower_order_freqs[ngram[:-1]] + vocab_size)
    return smoothed_probs

# Laplace smoothed bigram probabilities (Laplace düzeltmeli bigram olasılıkları)
laplace_bigram_probs = laplace_smoothing(bigram_freqs, unigram_freqs, vocab_size)

# 2. Add-k Smoothing
# Similar to Laplace, but we add k (which can be any value) instead of 1.
# Laplace smoothing'e benzer, ancak 1 yerine k eklenir.
def add_k_smoothing(ngram_freqs, lower_order_freqs, vocab_size, k=0.5):
    smoothed_probs = {}
    for ngram, count in ngram_freqs.items():
        # P(w_n | w_{n-1}) = (Count(w_{n-1}, w_n) + k) / (Count(w_{n-1}) + k * V)
        smoothed_probs[ngram] = (count + k) / (lower_order_freqs[ngram[:-1]] + k * vocab_size)
    return smoothed_probs

# Add-k smoothed bigram probabilities (Add-k düzeltmeli bigram olasılıkları)
add_k_bigram_probs = add_k_smoothing(bigram_freqs, unigram_freqs, vocab_size, k=0.5)

# 3. Interpolation (Interpolasyon)
# Combines unigrams, bigrams, and trigrams with different weights.
# Unigram, bigram ve trigram'ları farklı ağırlıklarla birleştirir.
def interpolation(uni_freqs, bi_freqs, tri_freqs, unigram_probs, bigram_probs, trigram_probs, lambda1=0.3, lambda2=0.4, lambda3=0.3):
    interpolated_probs = {}
    for trigram in tri_freqs.keys():
        unigram_prob = unigram_probs[(trigram[2],)] if (trigram[2],) in unigram_probs else 1 / vocab_size
        bigram_prob = bigram_probs[(trigram[1], trigram[2])] if (trigram[1], trigram[2]) in bigram_probs else 1 / vocab_size
        trigram_prob = trigram_probs[trigram] if trigram in trigram_probs else 1 / vocab_size
        interpolated_probs[trigram] = lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob
    return interpolated_probs

# For demonstration purposes, we assume probabilities for unigrams, bigrams, and trigrams are already calculated.
# Basit bir örnek için, unigram, bigram ve trigram olasılıklarını hesapladığımızı varsayıyoruz.
unigram_probs = {unigram: freq / total_unigrams for unigram, freq in unigram_freqs.items()}
bigram_probs = laplace_bigram_probs
trigram_probs = calculate_ngram_frequencies(trigrams)  # Assuming MLE probabilities here

# Interpolated trigram probabilities (Interpolasyon ile trigram olasılıkları)
interpolated_trigram_probs = interpolation(unigram_freqs, bigram_freqs, trigram_freqs, unigram_probs, bigram_probs, trigram_probs)

# Print smoothed probabilities (Düzeltmeli olasılıkları yazdırma)
print("Laplace Smoothed Bigram Probabilities:", laplace_bigram_probs)
print("\nAdd-k Smoothed Bigram Probabilities:", add_k_bigram_probs)
print("\nInterpolated Trigram Probabilities:", interpolated_trigram_probs)
