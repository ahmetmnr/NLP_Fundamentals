import re

# Regular expressions (regex) are patterns used to match character combinations in strings.
# Düzenli ifadeler (regex), metinlerde karakter kombinasyonlarını eşleştirmek için kullanılan kalıplardır.

# Example 1: Basic pattern matching (Basit kalıp eşleştirme)
# We will search for the word 'cat' in a given text.
# Verilen bir metinde 'cat' kelimesini arayacağız.
text = "The cat is sitting on the mat."
pattern = r'cat'

# Search for the pattern (Kalıbı arama)
match = re.search(pattern, text)
if match:
    print(f"Found '{match.group()}' in the text!")  # Found 'cat' in the text
else:
    print("No match found.")

# Example 2: Using character sets (Karakter setlerini kullanma)
# Search for any vowel (Sesli harf arama)
pattern = r'[aeiou]'
vowels = re.findall(pattern, text)
print(f"Vowels in the text: {vowels}")  # ['e', 'a', 'i', 'i', 'o', 'e', 'a']

# Example 3: Matching digits (Sayısal karakterleri eşleştirme)
# Let's find all the numbers in a text.
# Metindeki tüm sayıları bulalım.
text_with_numbers = "My phone number is 555-1234 and my zip code is 90210."
pattern = r'\d+'
numbers = re.findall(pattern, text_with_numbers)
print(f"Numbers in the text: {numbers}")  # ['555', '1234', '90210']

# Example 4: Matching specific patterns (Belirli kalıpları eşleştirme)
# We will search for a price format such as $199 or $24.99.
# Fiyat formatlarını (örneğin $199 veya $24.99) arayacağız.
price_text = "The item costs $25.99, but there is a discount: $5 off!"
pattern = r'\$\d+(\.\d{2})?'
prices = re.findall(pattern, price_text)
print(f"Prices found: {prices}")  # ['$25.99', '$5']

# Example 5: Using quantifiers (Kümeleme operatörleri kullanımı)
# '+' means one or more repetitions, '*' means zero or more repetitions
# '+' bir veya daha fazla tekrar, '*' sıfır veya daha fazla tekrar anlamına gelir.
# Let's find words starting with 'b' followed by one or more vowels.
# Bir veya daha fazla sesli harf ile başlayan 'b' harfini bulalım.
pattern = r'b[aeiou]+'
text = "bat, bit, bot, but, beet"
words = re.findall(pattern, text)
print(f"Words starting with 'b' and vowels: {words}")  # ['bat', 'bit', 'bot', 'but', 'beet']

# Example 6: Optional characters (İsteğe bağlı karakterler)
# The '?' means the preceding character is optional.
# '?' önceki karakterin isteğe bağlı olduğunu gösterir.
# Example: color and colour
pattern = r'colou?r'
text = "The color of the sky is blue. In British English, it's spelled as colour."
matches = re.findall(pattern, text)
print(f"Matches for 'color' or 'colour': {matches}")  # ['color', 'colour']

# Example 7: Anchors (^ for start, $ for end) (Bağlayıcılar: ^ başlangıç, $ bitiş için)
# Let's search for lines starting with 'The'.
# 'The' ile başlayan satırları bulalım.
pattern = r'^The'
text = "The sun is shining.\nA cat is sleeping.\nThe dog is barking."
matches = re.findall(pattern, text, re.MULTILINE)
print(f"Lines starting with 'The': {matches}")  # ['The', 'The']

# Example 8: Grouping and capturing (Gruplama ve yakalama)
# We can group parts of a pattern using parentheses and refer back to them.
# Kalıbın parçalarını parantez içine alarak gruplayabilir ve onlara geri dönebiliriz.
pattern = r'(\d{3})-(\d{4})'
phone_number = re.search(pattern, text_with_numbers)
if phone_number:
    print(f"Phone number area code: {phone_number.group(1)}")  # 555
    print(f"Phone number main part: {phone_number.group(2)}")  # 1234

# Example 9: Substitution (Değiştirme)
# Let's replace all occurrences of a pattern.
# Tüm eşleşen kalıpları değiştirelim.
text = "I love cats. Cats are great pets!"
pattern = r'cats'
updated_text = re.sub(pattern, 'dogs', text, flags=re.IGNORECASE)
print(f"Updated text: {updated_text}")  # 'I love dogs. Dogs are great pets!'

# Example 10: Lookahead and lookbehind (İleriye ve geriye bakış)
# Lookahead checks if a pattern is followed by another without consuming the characters.
# Lookbehind checks if a pattern is preceded by another.
# Lookahead, bir kalıbın başka bir şeyle takip edilip edilmediğini kontrol eder.
# Lookbehind ise bir kalıbın öncesinde başka bir şey olup olmadığını kontrol eder.

# Positive lookahead (Pozitif ileriye bakış): Match 'foo' followed by 'bar'.
# 'bar' ile takip edilen 'foo'yu eşleştirir.
pattern = r'foo(?=bar)'
text = "foobar foo"
matches = re.findall(pattern, text)
print(f"Matches with lookahead: {matches}")  # ['foo']

# Negative lookbehind (Negatif geriye bakış): Match 'bar' not preceded by 'foo'.
# 'foo' ile başlamayan 'bar'ları eşleştirir.
pattern = r'(?<!foo)bar'
text = "foobar bar"
matches = re.findall(pattern, text)
print(f"Matches with negative lookbehind: {matches}")  # ['bar']
# Advanced Regular Expressions Examples (Gelişmiş Düzenli İfadeler Örnekleri)
# This script demonstrates advanced uses of regular expressions.
# Bu betik, düzenli ifadelerin ileri seviye kullanım alanlarını göstermektedir.

# Example 1: Greedy vs Non-Greedy Matching (Hırslı ve Hırssız Eşleşme)
# By default, regex performs greedy matching, meaning it matches the longest possible string.
# Varsayılan olarak, regex en uzun eşleşmeyi yapar (hırslı eşleşme).
# Non-greedy matching can be done by adding a question mark (?) after the quantifier.
# Hırssız eşleşme için kümeleme operatöründen sonra '?' eklenir.

text = "The price is $100, but with a discount it's $75."
# Greedy matching
greedy_pattern = r'\$.*'
greedy_match = re.search(greedy_pattern, text)
print(f"Greedy match: {greedy_match.group()}")  # '$100, but with a discount it's $75.'

# Non-greedy matching (Hırssız eşleşme)
non_greedy_pattern = r'\$.*?'
non_greedy_match = re.search(non_greedy_pattern, text)
print(f"Non-greedy match: {non_greedy_match.group()}")  # '$100'

# Example 2: Backreferences (Geriye Referanslar)
# Backreferences allow you to refer to previously captured groups.
# Geriye referanslar, önceki gruplara referans vermenize olanak tanır.

text = "abab abab"
# This pattern captures a group and then checks if the same group repeats.
# Bu desen, bir grubu yakalar ve aynı grubun tekrar edip etmediğini kontrol eder.
backref_pattern = r'(ab)\1'
backref_match = re.search(backref_pattern, text)
if backref_match:
    print(f"Backreference match: {backref_match.group()}")  # 'abab'

# Example 3: Flags (Bayraklar)
# Flags modify the behavior of regex patterns.
# Bayraklar, regex kalıplarının davranışını değiştirir.

# re.IGNORECASE (Case-insensitive matching)
# Büyük/küçük harf duyarsız eşleşme
text = "Cats are wonderful. CATS are adorable."
pattern = r'cats'
ignore_case_matches = re.findall(pattern, text, flags=re.IGNORECASE)
print(f"Case-insensitive matches: {ignore_case_matches}")  # ['Cats', 'CATS']

# re.MULTILINE (Multi-line matching)
# Birden fazla satırda eşleşme
text = "First line\nSecond line\nThird line"
multiline_pattern = r'^\w+'
multiline_matches = re.findall(multiline_pattern, text, flags=re.MULTILINE)
print(f"Multiline matches (start of each line): {multiline_matches}")  # ['First', 'Second', 'Third']

# re.DOTALL (Allow dot (.) to match newline)
# Dot'un (\n) ile eşleşmesini sağlar.
text = "First line.\nSecond line."
dotall_pattern = r'First.*Second'
dotall_match = re.search(dotall_pattern, text, flags=re.DOTALL)
if dotall_match:
    print(f"Dotall match: {dotall_match.group()}")  # 'First line.\nSecond line.'

# Example 4: Unicode Character Matching (Unicode Karakter Eşleşmesi)
# In regex, \w typically matches letters, digits, and underscores, but not non-ASCII letters.
# Regex'de \w genellikle harfler, sayılar ve alt çizgilerle eşleşir, ancak ASCII dışı harflerle eşleşmez.
# Let's match Unicode characters such as accented letters.
# Unicode karakterleri (örn. aksanlı harfler) ile eşleşme yapalım.
text = "Café Müller is a popular spot."
unicode_pattern = r'\w+'
matches = re.findall(unicode_pattern, text, flags=re.UNICODE)
print(f"Unicode word matches: {matches}")  # ['Café', 'Müller', 'is', 'a', 'popular', 'spot']

# Example 5: Recursive Patterns (Özyineli Desenler)
# Recursive patterns allow matching nested structures like parentheses.
# Özyineli desenler, iç içe geçmiş yapılarla eşleşmeye olanak tanır (örneğin parantezler).
# Python's 're' module does not directly support recursion, but we can simulate it in specific cases.
# Python'un 're' modülü özyinelemeyi doğrudan desteklemez, ancak belirli durumlarda bunu simüle edebiliriz.
# For example, matching balanced parentheses:
text = "(a(b(c)d)e)"
# Matches balanced parentheses by using lookahead assertions.
# Dengeyi sağlayan parantezlerle eşleşme.
balanced_pattern = r'\(([^()]*(?:\([^()]*\))*[^()]*)\)'
balanced_match = re.search(balanced_pattern, text)
if balanced_match:
    print(f"Balanced parentheses match: {balanced_match.group()}")  # '(a(b(c)d)e)'

# Example 6: Advanced Assertions (Gelişmiş Koşullar)
# Positive Lookahead: Match 'foo' followed by 'bar' without consuming 'bar'.
# Pozitif İleriye Bakış: 'foo'dan sonra gelen 'bar' ile eşleşir, ancak 'bar'ı tüketmez.
text = "foo bar"
lookahead_pattern = r'foo(?= bar)'
lookahead_match = re.search(lookahead_pattern, text)
if lookahead_match:
    print(f"Lookahead match: {lookahead_match.group()}")  # 'foo'

# Negative Lookahead: Match 'foo' only if it's not followed by 'bar'.
# Negatif İleriye Bakış: 'foo' sadece 'bar' ile takip edilmezse eşleşir.
text = "foo baz"
neg_lookahead_pattern = r'foo(?! bar)'
neg_lookahead_match = re.search(neg_lookahead_pattern, text)
if neg_lookahead_match:
    print(f"Negative lookahead match: {neg_lookahead_match.group()}")  # 'foo'

# Example 7: Capturing vs Non-capturing Groups (Yakalama ve Yakalamayan Gruplar)
# Capturing groups store the matched part, non-capturing groups don't.
# Yakalama grupları eşleşen kısmı saklar, yakalamayan gruplar saklamaz.
text = "apple orange"
capturing_pattern = r'(apple|orange)'
capturing_match = re.findall(capturing_pattern, text)
print(f"Capturing groups: {capturing_match}")  # ['apple', 'orange']

# Non-capturing group (Yakalamayan grup)
non_capturing_pattern = r'(?:apple|orange)'
non_capturing_match = re.findall(non_capturing_pattern, text)
print(f"Non-capturing groups: {non_capturing_match}")  # ['apple', 'orange']

# Example 8: Inline Comments for Readability (Satır İçi Yorumlar)
# You can add comments to your regex for better readability using (?x).
# Daha anlaşılır regex yazmak için (?x) kullanarak yorum ekleyebilirsiniz.
pattern = r'''(?x)        # Enable verbose mode for comments (Yorumlar için detaylı mod)
    \b                    # Word boundary (Kelime sınırı)
    \d{3}                 # Match exactly 3 digits (Tam olarak 3 basamak eşleşir)
    -                     # Match a hyphen (Bir tire ile eşleşir)
    \d{4}                 # Match exactly 4 digits (Tam olarak 4 basamak eşleşir)
'''
text = "My number is 123-4567."
inline_comment_match = re.search(pattern, text)
if inline_comment_match:
    print(f"Inline comment match: {inline_comment_match.group()}")  # '123-4567'