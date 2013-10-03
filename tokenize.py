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
    for symbol in string.punctuation:
      input_string = input_string.replace(symbol, '')
    split_string = re.compile('\s').split(input_string.lower())
    self.sample_number = int(split_string[0])
    self.tokens = split_string[1:]

class FileTokenizer:
  '''Takes an input filename where the file is a list of numbered lines
  and runs it through the Tokenizer'''
  def __init__(self, input_filename):
    with open(input_filename, 'r') as input_file:
      contents = input_file.read().strip()
    self.lines = re.compile('\n').split(contents)

  def tokenized_lines(self):
    for line in self.lines:
      yield StringTokenizer(line)
