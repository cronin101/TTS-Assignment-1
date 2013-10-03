from tokenize import *
import os
import itertools

class TFIDFScorer:
  def __init__(self, filename, queries_file, documents_file):
    self.filename = filename
    self.queries = FileTokenizer(queries_file).tokenized_lines()
    self.documents = FileTokenizer(documents_file).tokenized_lines()
    corpus = itertools.chain(self.queries, self.documents)
    unique_words = set(itertools.chain(map(lambda l: l.tokens, corpus)))

TFIDFScorer('derp', './qrys.txt', './docs.txt')
