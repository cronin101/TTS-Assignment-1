from tokenize import *
import os
from itertools import chain

class TFIDFScorer:
  def __init__(self, filename, queries_file, documents_file):
    self.filename = filename
    self.queries = FileTokenizer(queries_file).tokenized_lines()
    self.documents = FileTokenizer(documents_file).tokenized_lines()
    corpus = chain(self.queries, self.documents)
    unique_words = set(chain.from_iterable(map(lambda l: l.tokens, corpus)))

    self.word_id = dict(map(lambda (word, i): (i, word), enumerate(unique_words, 1)))

if __name__ == "__main__":
  TFIDFScorer('derp', './qrys.txt', './docs.txt')
