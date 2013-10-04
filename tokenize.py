import re
import string

class StringTokenizer:
  '''Takes an input string in the format:
    '1 foo bar baz, the cow jumped.'

    and returns:
      sample_number: 1
      tokens: ['foo', 'bar', 'baz', 'the', 'cow', 'jumped']
  '''
  def __init__(self, input_string):
    nonword_regex, word_regex = re.compile('\W'), re.compile('\w')
    split_string = nonword_regex.split(input_string.lower())
    no_punctuation = filter(lambda w: word_regex.match(w), split_string)
    self.sample_number, self.tokens = int(no_punctuation[0]), no_punctuation[1:]

class FileTokenizer:
  '''Takes an input filename where the file is a list of numbered lines
  and runs it through the Tokenizer'''
  def __init__(self, input_filename):
    with open(input_filename, 'r') as input_file:
      contents = input_file.read().strip()
    self.lines = re.compile('\n').split(contents)

  def all(self):
    for line in self.lines:
      yield StringTokenizer(line)
