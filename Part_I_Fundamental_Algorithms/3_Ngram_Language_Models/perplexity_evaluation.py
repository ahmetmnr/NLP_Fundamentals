import math
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import defaultdict

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

# Function to calculate Maximum Likelihood Estimation (MLE) probabilities for bigrams
# Bigram'lar için Maximum Likelihood Estimation (MLE) olasılıklarını hesaplayan fonksiyon
def calculate_mle_ngram_probabilities(n_gram_freqs, lower_order_gram_freqs):
    """Calculate MLE for N-grams. (N-gram'lar için MLE hesapla.)"""
    ngram_probs = {}
    for ngram, count in n_gram_freqs.items():
        ngram_probs[ngram] = count / lower_order_gram_freqs[ngram[:-1]]
    return ngram_probs

# MLE for bigrams and trigrams (Bigram ve trigram'lar için MLE)
bigram_mle_probs = calculate_mle_ngram_probabilities(bigram_freqs, unigram_freqs)
trigram_mle_probs = calculate_mle_ngram_probabilities(trigram_freqs, bigram_freqs)

# Function to calculate perplexity of an N-gram model
# N-gram modelinin perplexity'sini hesaplayan fonksiyon
def calculate_perplexity(test_sequence, ngram_probs, n):
    """Calculate the perplexity of a test sequence using the N-gram model.
       (N-gram modeli kullanarak bir test dizisinin perplexity'sini hesapla.)"""
    
    tokens = word_tokenize(test_sequence.lower())
    sequence_ngrams = list(ngrams(tokens, n))
    
    log_probability_sum = 0
    for ngram in sequence_ngrams:
        prob = ngram_probs.get(ngram, 1e-5)  # Use a small probability for unseen N-grams
        log_probability_sum += math.log(prob, 2)
    
    # Calculate perplexity
    N = len(sequence_ngrams)
    perplexity = math.pow(2, -log_probability_sum / N)
    
    return perplexity

# Example test sentence (Örnek test cümlesi)
test_sentence = "I love natural language processing"

# Calculate perplexity using bigram model (Bigram modeli ile perplexity hesaplama)
bigram_perplexity = calculate_perplexity(test_sentence, bigram_mle_probs, 2)
print(f"Bigram Model Perplexity for the sentence '{test_sentence}': {bigram_perplexity}")

# Calculate perplexity using trigram model (Trigram modeli ile perplexity hesaplama)
trigram_perplexity = calculate_perplexity(test_sentence, trigram_mle_probs, 3)
print(f"Trigram Model Perplexity for the sentence '{test_sentence}': {trigram_perplexity}")
