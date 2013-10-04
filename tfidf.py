from tokenize import *
import os
from itertools import chain

class TFIDFScorer:
  def __init__(self, filename, queries_file, documents_file, k):
    self.filename = filename
    self.queries = FileTokenizer(queries_file)
    self.documents = FileTokenizer(documents_file)
    self.k = k
    corpus = chain(self.queries.all(), self.documents.all())
    self.unique_words = set(chain.from_iterable(map(lambda l: l.tokens, corpus)))
    self.word_id = dict(map(lambda t: reversed(t), enumerate(self.unique_words, 1)))
    self.document_by_id = dict(map(lambda d: (d.sample_number, d), self.documents.all()))

  def crunch_numbers(self):
    self.compute_k_d_over_avg_d()
    self.compute_term_frequency()

  def compute_k_d_over_avg_d(self):
    '''Document length normalisation factor'''
    self.kd_o_avd = {}
    document_lengths = map(lambda d: len(d.tokens), self.documents.all())
    average_doc_len = sum(document_lengths) / float(len(document_lengths))
    for document in self.documents.all():
      self.kd_o_avd[document.sample_number] = (self.k * len(document.tokens)) / average_doc_len

  def get_tf(self, word_id, document_id):
    '''Occurrences of word i in document j'''
    self.term_frequency.get((word_id, document_id), 0)

  def compute_term_frequency(self):
    '''For each document, record the number of times that each word appears'''
    self.term_frequency = {}
    for document in self.documents.all():
      for word in document.tokens:
        existing = self.term_frequency.get((self.word_id[word], document.sample_number), 0)
        self.term_frequency[(self.word_id[word], document.sample_number)] = existing + 1

if __name__ == "__main__":
  TFIDFScorer('derp', './qrys.txt', './docs.txt', 2.0).crunch_numbers()
