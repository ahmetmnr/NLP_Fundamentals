
# N-gram Language Models

This document describes various N-gram modeling techniques using Python code examples. These models are essential for understanding how Natural Language Processing (NLP) handles language patterns based on the probability of word sequences. We'll cover topics such as N-gram models, Maximum Likelihood Estimation (MLE), perplexity, smoothing techniques, sentence generation, and backoff models.

## Files:

1. `ngram_model.py`: Basic N-gram model implementation.
2. `mle_ngram.py`: Maximum Likelihood Estimation (MLE) for N-grams.
3. `perplexity_evaluation.py`: Perplexity evaluation for N-gram models.
4. `smoothing_techniques.py`: Smoothing techniques like Laplace and Add-k.
5. `sentence_generation.py`: Sentence generation using N-gram models.
6. `backoff_model.py`: Stupid Backoff model implementation.

---

## `ngram_model.py`

This script implements a basic N-gram model, allowing for the calculation of unigrams, bigrams, and trigrams, as well as their frequencies and probabilities.

```python
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

# Function to calculate N-gram frequency
def calculate_ngram_frequencies(n_grams):
    ngram_freqs = defaultdict(int)
    for ngram in n_grams:
        ngram_freqs[ngram] += 1
    return ngram_freqs

# Calculate frequencies for unigrams, bigrams, and trigrams
unigram_freqs = calculate_ngram_frequencies(unigrams)
bigram_freqs = calculate_ngram_frequencies(bigrams)
trigram_freqs = calculate_ngram_frequencies(trigrams)

# Print frequencies
print("\nUnigram Frequencies:", dict(unigram_freqs))
print("Bigram Frequencies:", dict(bigram_freqs))
print("Trigram Frequencies:", dict(trigram_freqs))

# Function to calculate N-gram probabilities
def calculate_ngram_probabilities(n_gram_freqs, total_count):
    ngram_probs = {}
    for ngram, count in n_gram_freqs.items():
        ngram_probs[ngram] = count / total_count
    return ngram_probs

# Calculate unigram, bigram, and trigram probabilities
total_unigrams = len(unigrams)
unigram_probs = calculate_ngram_probabilities(unigram_freqs, total_unigrams)

total_bigrams = len(bigrams)
bigram_probs = calculate_ngram_probabilities(bigram_freqs, total_bigrams)

total_trigrams = len(trigrams)
trigram_probs = calculate_ngram_probabilities(trigram_freqs, total_trigrams)

# Print probabilities
print("\nUnigram Probabilities:", unigram_probs)
print("Bigram Probabilities:", bigram_probs)
print("Trigram Probabilities:", trigram_probs)

# Function to calculate the probability of a sequence using n-grams
def calculate_sequence_probability(sequence, ngram_probs, n):
    tokens = word_tokenize(sequence.lower())
    sequence_ngrams = list(ngrams(tokens, n))
    prob = 1.0
    for ngram in sequence_ngrams:
        prob *= ngram_probs.get(ngram, 1e-5)
    return prob

# Example sequence probability
sequence = "I love natural language"
sequence_prob = calculate_sequence_probability(sequence, bigram_probs, 2)
print(f"\nProbability of the sequence '{sequence}': {sequence_prob}")
```

---

## `mle_ngram.py`

This script calculates Maximum Likelihood Estimation (MLE) probabilities for bigrams and trigrams.

```python
from collections import defaultdict
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Sample text corpus
corpus = "I love natural language processing. Natural language processing is fun. I love machine learning."

# Tokenize the text into words
nltk.download('punkt')
tokens = word_tokenize(corpus.lower())

# Function to calculate n-grams
def calculate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Function to calculate N-gram frequencies
def calculate_ngram_frequencies(n_grams):
    ngram_freqs = defaultdict(int)
    for ngram in n_grams:
        ngram_freqs[ngram] += 1
    return ngram_freqs

# MLE for bigrams and trigrams
def calculate_mle_ngram_probabilities(n_gram_freqs, lower_order_gram_freqs):
    ngram_probs = {}
    for ngram, count in n_gram_freqs.items():
        ngram_probs[ngram] = count / lower_order_gram_freqs[ngram[:-1]]
    return ngram_probs
```

This document continues with the rest of the Python scripts (`perplexity_evaluation.py`, `smoothing_techniques.py`, `sentence_generation.py`, `backoff_model.py`), explaining and showing examples for each.

---
