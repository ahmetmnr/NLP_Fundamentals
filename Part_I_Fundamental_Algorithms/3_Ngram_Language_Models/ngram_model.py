from collections import defaultdict
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import math

# Sample text corpus (Örnek metin corpus'u)
corpus = "I love natural language processing. Natural language processing is fun. I love machine learning."

# Tokenize the text into words (Metni kelimelere böl)
nltk.download('punkt')
tokens = word_tokenize(corpus.lower())

# Function to calculate n-grams (N-gram'ları hesaplayan fonksiyon)
def calculate_ngrams(tokens, n):
    """ Calculate n-grams from the tokenized text. (Tokenize edilmiş metinden n-gram'ları hesapla.) """
    n_grams = list(ngrams(tokens, n))
    return n_grams

# Unigrams (Unigram'lar)
unigrams = calculate_ngrams(tokens, 1)
print(f"Unigrams: {unigrams}")

# Bigrams (Bigram'lar)
bigrams = calculate_ngrams(tokens, 2)
print(f"Bigrams: {bigrams}")

# Trigrams (Trigram'lar)
trigrams = calculate_ngrams(tokens, 3)
print(f"Trigrams: {trigrams}")

# Function to calculate N-gram frequency (N-gram frekanslarını hesaplama)
def calculate_ngram_frequencies(n_grams):
    """ Calculate the frequency of each N-gram. (Her N-gram'ın frekansını hesapla.) """
    ngram_freqs = defaultdict(int)
    for ngram in n_grams:
        ngram_freqs[ngram] += 1
    return ngram_freqs

# Calculate frequencies for unigrams, bigrams, and trigrams (Unigram, bigram ve trigram frekanslarını hesaplayın)
unigram_freqs = calculate_ngram_frequencies(unigrams)
bigram_freqs = calculate_ngram_frequencies(bigrams)
trigram_freqs = calculate_ngram_frequencies(trigrams)

# Print frequencies (Frekansları yazdırma)
print("\nUnigram Frequencies:", dict(unigram_freqs))
print("Bigram Frequencies:", dict(bigram_freqs))
print("Trigram Frequencies:", dict(trigram_freqs))

# Function to calculate N-gram probabilities (N-gram olasılıklarını hesaplayan fonksiyon)
def calculate_ngram_probabilities(n_gram_freqs, total_count):
    """ Calculate the probability of each N-gram. (Her N-gram'ın olasılığını hesapla.) """
    ngram_probs = {}
    for ngram, count in n_gram_freqs.items():
        ngram_probs[ngram] = count / total_count
    return ngram_probs

# Calculate unigram, bigram, and trigram probabilities (Unigram, bigram ve trigram olasılıklarını hesaplayın)
total_unigrams = len(unigrams)
unigram_probs = calculate_ngram_probabilities(unigram_freqs, total_unigrams)

total_bigrams = len(bigrams)
bigram_probs = calculate_ngram_probabilities(bigram_freqs, total_bigrams)

total_trigrams = len(trigrams)
trigram_probs = calculate_ngram_probabilities(trigram_freqs, total_trigrams)

# Print probabilities (Olasılıkları yazdırma)
print("\nUnigram Probabilities:", unigram_probs)
print("Bigram Probabilities:", bigram_probs)
print("Trigram Probabilities:", trigram_probs)

# Function to calculate the probability of a sequence using n-grams
# N-gram'lar kullanarak bir dizinin olasılığını hesaplama fonksiyonu
def calculate_sequence_probability(sequence, ngram_probs, n):
    """ Calculate the probability of a given sequence using the N-gram model. (Verilen bir dizinin olasılığını N-gram modeliyle hesapla.) """
    tokens = word_tokenize(sequence.lower())
    sequence_ngrams = list(ngrams(tokens, n))
    prob = 1.0
    for ngram in sequence_ngrams:
        prob *= ngram_probs.get(ngram, 1e-5)  # Use a small probability for unseen N-grams (Görülmeyen N-gram'lar için küçük bir olasılık kullanın)
    return prob

# Example sequence probability (Örnek dizinin olasılığı)
sequence = "I love natural language"
sequence_prob = calculate_sequence_probability(sequence, bigram_probs, 2)
print(f"\nProbability of the sequence '{sequence}': {sequence_prob}")

