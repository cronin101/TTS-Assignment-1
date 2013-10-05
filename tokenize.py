import re
import string

class StringTokenizer:
  '''Takes an input string in the format:
    '1 foo bar baz, the cow jumped.'

    and returns:
      sample_number: 1
      tokens: ['foo', 'bar', 'baz', 'the', 'cow', 'jumped']
  '''
  def __init__(self, input_string, split_and_merge = False):
    lowered_input = input_string.lower()
    nonword_regex, word_regex, whitespace_regex = re.compile('\W'), re.compile('\w'), re.compile('\s')
    split_string = nonword_regex.split(lowered_input)
    tokens = filter(lambda w: word_regex.match(w), split_string)
    if split_and_merge:
      def should_merge(word):
        '''Word should start and end in word characters but contain at least one nonword char'''
        if len(word) < 3: return False
        if not(word_regex.match(word[0]) and word_regex.match(word[-1])): return False
        return nonword_regex.search(word[1:-1])

      merged = filter(should_merge, whitespace_regex.split(lowered_input))
      tokens = tokens + merged

    self.sample_number, self.tokens = int(tokens[0]), tokens[1:]

class FileTokenizer:
  '''Takes an input filename where the file is a list of numbered lines
  and runs it through the Tokenizer'''
  def __init__(self, input_filename, split_and_merge = False):
    self.split_and_merge = split_and_merge
    with open(input_filename, 'r') as input_file:
      contents = input_file.read().strip()
    self.lines = re.compile('\n').split(contents)

  def all(self):
    for line in self.lines:
      yield StringTokenizer(line, self.split_and_merge)
