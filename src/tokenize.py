import re
import string
from nltk.stem.lancaster import LancasterStemmer
from counter_backport import Counter
import nltk

stopwords = nltk.corpus.stopwords.words('english')
stemmer = LancasterStemmer()
stems = {}
nonword_regex, word_regex, whitespace_regex = re.compile('\W'), re.compile('\w'), re.compile('\s')

def cachedStem(word):
  cache = stems.get(word, None)
  if cache is None:
    stemmed = stemmer.stem(word)
    stems[word] = stemmed
    return stemmed
  else:
    return cache


class ExpandedQuery:
  def __init__(self, number, tokens, counter):
    self.number, self.tokens, self.counter = number, tokens, counter

class StringTokenizer:
  '''Takes an input string in the format:
    '1 foo bar baz, the cow jumped.'

    and returns:
      number: 1
      tokens: ['foo', 'bar', 'baz', 'the', 'cow', 'jumped']
  '''
  def __init__(self, input_string,
    stem = False, remove_stopwords = False, split_and_merge = False, token_correction = False, include_3grams = False, repeat_titles = False
  ):
    lowered_input = input_string.lower()
    tokens = [w for w in nonword_regex.split(lowered_input) if word_regex.match(w)]

    if include_3grams:
      tokens = tokens + [string.join([tokens[i-2], tokens[i-1], tokens[i]]) for i in range(2, len(tokens))]

    if split_and_merge:
      def should_merge(word):
        '''Word should start and end in word characters but contain at least one nonword char'''
        if not(len(word) >= 3 and word_regex.match(word[0]) and word_regex.match(word[-1])): return False
        return nonword_regex.search(word[1:-1])

      tokens = tokens + [t for t in whitespace_regex.split(lowered_input) if should_merge(t)]

    if repeat_titles:
      tokens = tokens + [t for t in nonword_regex.split(input_string) if t.istitle()]

    if token_correction:
      digit_to_string = {
        '0' : 'zero',
        '1' : 'one',
        '2' : 'two',
        '3' : 'three',
        '4' : 'four',
        '5' : 'five',
        '6' : 'six',
        '7' : 'seven',
        '8' : 'eight',
        '9' : 'nine',
        '10' : 'ten'
      }
      tokens = [tokens[0]] + [digit_to_string[t] if digit_to_string.get(t, None) is not None else t for t in tokens[1:]]

    if remove_stopwords:
      tokens = [tokens[0]] + [t for t in tokens[1:] if not t in stopwords]

    if stem:
      tokens = [tokens[0]] + [cachedStem(w) for w in tokens[1:]]

    self.number, self.tokens = int(tokens[0]), tokens[1:]
    self.counter = Counter(self.tokens)

class FileTokenizer:
  '''Takes an input filename where the file is a list of numbered lines
  and runs it through the Tokenizer'''
  def __init__(self, input_filename,
    stem = False, remove_stopwords = False, split_and_merge = False, token_correction = False, include_3grams = False, repeat_titles = False
  ):
    self.split_and_merge, self.token_correction, self.include_3grams = split_and_merge, token_correction, include_3grams
    self.stem, self.remove_stopwords, self.repeat_titles = stem, remove_stopwords, repeat_titles
    with open(input_filename, 'r') as input_file:
      contents = input_file.read().strip()
    self.lines = re.compile('\n').split(contents)

  def all(self):
    return (StringTokenizer(line, self.stem, self.remove_stopwords, self.split_and_merge, self.token_correction, self.include_3grams, self.repeat_titles) for line in self.lines)
