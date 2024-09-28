
# Regular Expressions, Tokenization, and Edit Distance

## Overview
This section covers fundamental Natural Language Processing (NLP) techniques including **Regular Expressions**, **Tokenization**, and **Edit Distance**. Each of these topics is crucial for text processing tasks in NLP, and here we provide Python implementations with explanations.

Bu bölüm, Doğal Dil İşleme (NLP) tekniklerinden **Düzenli İfadeler (Regex)**, **Tokenizasyon** ve **Edit Mesafesi** gibi temel kavramları kapsar. Her biri, metin işleme görevleri için çok önemlidir ve burada Python uygulamaları ve açıklamalar yer almaktadır.

## File Structure

- **regex_example.py**
- **tokenization_example.py**
- **edit_distance_example.py**

### regex_example.py

In this script, we demonstrate the usage of **Regular Expressions** (regex), a powerful tool for pattern matching in text. Regular expressions allow you to search for specific patterns, extract information, and manipulate strings efficiently.

Bu betikte, metinlerde desen eşleştirme için güçlü bir araç olan **Düzenli İfadeler**in (regex) kullanımını gösteriyoruz. Düzenli ifadeler, belirli kalıpları aramanıza, bilgileri çıkarmanıza ve dizeleri verimli bir şekilde değiştirmenize olanak tanır.

#### Topics Covered:
1. **Basic Pattern Matching:** Simple matching of a word in a text.
2. **Character Sets:** Matching specific character groups, such as vowels.
3. **Quantifiers:** Using operators like `*`, `+`, and `?` for flexible pattern matching.
4. **Anchors:** Matching patterns at the start or end of a line using `^` and `$`.
5. **Substitution:** Replacing matching patterns in a text with a different string.
6. **Advanced Features:** Lookahead/lookbehind assertions, greedy/non-greedy matching, and grouping.

#### Example Usage:

```python
import re
pattern = r'\d+'
text = "I have 100 apples and 200 oranges."
numbers = re.findall(pattern, text)
print(numbers)  # Output: ['100', '200']
```

### tokenization_example.py

This script covers **Tokenization**, which is the process of breaking down text into smaller units such as words or sentences. Tokenization is a fundamental step in NLP for processing and analyzing text.

Bu betik, metinleri kelimelere veya cümlelere bölme işlemi olan **Tokenizasyon**u ele alır. Tokenizasyon, NLP'de metinleri işlemek ve analiz etmek için temel bir adımdır.

#### Topics Covered:
1. **Word Tokenization:** Breaking text into individual words.
2. **Sentence Tokenization:** Breaking text into sentences.
3. **Regex-based Tokenization:** Tokenizing based on custom patterns using regular expressions.
4. **Tweet Tokenization:** Handling informal text, such as tweets.
5. **Subword Tokenization:** Breaking words into subword units for use in advanced models like BERT.

#### Example Usage:

```python
from nltk.tokenize import word_tokenize
text = "Natural Language Processing is amazing!"
tokens = word_tokenize(text)
print(tokens)  # Output: ['Natural', 'Language', 'Processing', 'is', 'amazing', '!']
```

### edit_distance_example.py

In this script, we explore **Edit Distance** (Levenshtein Distance), which measures the number of operations (insertions, deletions, substitutions) required to transform one string into another. This is useful in spell checking, string matching, and sequence alignment tasks.

Bu betikte, bir dizeyi başka bir dizeye dönüştürmek için gereken işlem sayısını ölçen **Edit Mesafesi**ni (Levenshtein Mesafesi) inceliyoruz. Bu teknik yazım denetimi, dize eşleştirme ve dizilim hizalaması gibi görevlerde kullanışlıdır.

#### Topics Covered:
1. **Basic Edit Distance:** Calculating the minimum number of operations to convert one word into another.
2. **Custom Edit Distance:** Defining custom costs for insertions, deletions, and substitutions.
3. **Optimal String Alignment:** Considering transpositions when calculating edit distance.

#### Example Usage:

```python
from nltk.metrics import edit_distance
word1 = "kitten"
word2 = "sitting"
distance = edit_distance(word1, word2)
print(f"Edit distance: {distance}")  # Output: 3
```

## Requirements

This section uses the following libraries:

- **nltk:** Natural Language Toolkit for text processing tasks. You can install it using:

```bash
pip install nltk
```

Additionally, make sure to download the necessary NLTK data:

```python
import nltk
nltk.download('punkt')
```

## Conclusion

This directory provides Python implementations for Regular Expressions, Tokenization, and Edit Distance. These techniques are widely used in Natural Language Processing for text manipulation, analysis, and comparison. Each script is well-commented and can be adapted for various NLP projects.

Bu dizin, Düzenli İfadeler, Tokenizasyon ve Edit Mesafesi için Python uygulamaları sunar. Bu teknikler, metin işleme, analiz ve karşılaştırma için Doğal Dil İşleme'de yaygın olarak kullanılır. Her betik iyi yorumlanmıştır ve çeşitli NLP projeleri için uyarlanabilir.
