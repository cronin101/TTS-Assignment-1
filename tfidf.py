from tokenize import *
import os
from math import log
from itertools import chain

class TFIDFScorer:
  def __init__(self, filename, queries_file, documents_file, k):
    self.filename, self.k = filename, k
    self.queries, self.documents  = FileTokenizer(queries_file), FileTokenizer(documents_file)
    corpus = chain(self.queries.all(), self.documents.all())
    self.unique_words = set(chain.from_iterable(map(lambda l: l.tokens, corpus)))
    self.word_id = dict(map(lambda t: reversed(t), enumerate(self.unique_words, 1)))
    self.document_by_id = dict(map(lambda d: (d.sample_number, d), self.documents.all()))
    self.C = len(list(self.documents.all()))

  def crunch_numbers(self):
    self.compute_k_d_over_avg_d()
    self.compute_term_frequency()
    return self

  def __format_line(self, query_num, doc_num, score):
    values = map(lambda x: str(x), [query_num, doc_num, score])
    return string.join(values, ' 0 ') + ' 0' + os.linesep

  def dump(self):
    with open(self.filename, 'w') as scores:
      for query in self.queries.all():
        for doc_id in range(1, self.C):
          score = self.query_score(query, doc_id)
          if score > 0:
            line = self.__format_line(query.sample_number, doc_id, score)
            scores.write(line)

  def compute_k_d_over_avg_d(self):
    '''Document length normalisation factor'''
    self.kd_o_avd = {}
    document_lengths = map(lambda d: len(d.tokens), self.documents.all())
    average_doc_len = sum(document_lengths) / float(len(document_lengths))
    for document in self.documents.all():
      self.kd_o_avd[document.sample_number] = (self.k * len(document.tokens)) / average_doc_len

  def get_tf(self, word_id, document_id):
    '''Occurrences of word i in document j'''
    return self.term_frequency.get((word_id, document_id), 0)

  def get_df(self, word_id):
    '''Occurrences of word i in any document'''
    return self.document_frequency.get(word_id, 0)

  def tf(self, word_id, document_id):
    tf = self.get_tf(word_id, document_id)
    return tf / (tf + self.kd_o_avd[document_id])

  def idf(self, word_id):
    return log(self.C / 1.0 + self.get_df(word_id), 10)

  def query_score(self, query, document_id):
    query_words = query.tokens
    q_word_ids = map(lambda w: self.word_id[w], query_words)
    j = document_id
    return sum(map(lambda i: q_word_ids.count(i) * self.tf(i,j) * self.idf(i), q_word_ids))

  def compute_term_frequency(self):
    '''For each document, record the number of times that each word appears'''
    self.term_frequency = {}
    self.document_frequency = {}
    for document in self.documents.all():
      doc_id = document.sample_number
      for word in document.tokens:
        word_id = self.word_id[word]
        existing_tf = self.get_tf(word_id, doc_id)
        self.term_frequency[(word_id, doc_id)] = existing_tf + 1
        existing_df = self.get_df(word_id)
        self.document_frequency[word_id] = existing_df + 1

if __name__ == "__main__":
  TFIDFScorer('tfidf.top', './qrys.txt', './docs.txt', 2.0).crunch_numbers().dump()