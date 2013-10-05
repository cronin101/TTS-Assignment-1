from tokenize import *
import os
from math import log
from itertools import chain

class TFIDFScorer:
  def __init__(self, filename, queries, documents, k):
    self.filename, self.k = filename, k
    self.queries, self.documents  = list(queries), list(documents)
    corpus = chain(self.queries, self.documents)
    self.unique_words = self.flatten_and_unique(corpus)
    self.word_id = dict(map(lambda t: reversed(t), enumerate(self.unique_words, 1)))
    self.document_by_id = dict(map(lambda d: (d.sample_number, d), self.documents))
    self.C = len(self.documents)

  def flatten_and_unique(self, list_of_lists):
    return set(chain.from_iterable(map(lambda l: l.tokens, list_of_lists)))

  def crunch_numbers(self):
    self.compute_term_frequency()
    self.compute_k_d_over_avg_d()
    return self

  def __format_line(self, query_num, doc_num, score):
    values = map(lambda x: str(x), [query_num, doc_num, score])
    return string.join(values, ' 0 ') + ' 0' + os.linesep

  def dump(self):
    with open(self.filename, 'w') as scores:
      for query in self.queries:
        for doc_id in range(1, self.C):
          score = self.query_score(query, doc_id)
          if score > 0:
            line = self.__format_line(query.sample_number, doc_id, score)
            scores.write(line)

  def average_doc_len(self):
    '''Average length of documents in collection C'''
    document_lengths = map(lambda d: len(d.tokens), self.documents)
    return sum(document_lengths) / float(len(document_lengths))

  def compute_k_d_over_avg_d(self):
    '''Document length normalisation factor'''
    self.kd_o_avd = {}
    average_doc_len = self.average_doc_len()
    for document in self.documents:
      self.kd_o_avd[document.sample_number] = (self.k * len(document.tokens)) / average_doc_len

  def get_tf(self, word_id, document_id):
    '''Occurrences of word i in document j'''
    return self.term_frequency.get((word_id, document_id), 0)

  def get_df(self, word_id):
    '''Number of documents that word i appears in'''
    return self.document_frequency.get(word_id, 0)

  def tf(self, word_id, document_id):
    '''Normalised term frequency component:
    Repetitions less important, except in long documents'''
    tf = self.get_tf(word_id, document_id)
    return tf / (tf + self.kd_o_avd[document_id])

  def idf(self, word_id):
    '''Inverse document frequency:
    Rare words are considered more important for ranking'''
    return log(self.C / (1.0 + self.get_df(word_id)), 2)

  def q_tf(self, word_id, query_words):
    '''Query term frequency:
    Words that appear more often in the query are ranked higher'''
    return map(lambda w: self.word_id[w], query_words).count(word_id)

  def query_score(self, query, document_id):
    '''Sum over all query words i of qtf . tf . idf'''
    query_words = query.tokens
    q_word_ids = set(map(lambda w: self.word_id[w], query_words))

    def qtf_tf_idf(word_id):
      i = word_id
      j = document_id
      return self.q_tf(i, query_words) * self.tf(i, j) * self.idf(i)

    return sum(map(qtf_tf_idf, q_word_ids))

  def compute_term_frequency(self):
    '''For each document, record the number of times that each word appears'''
    self.term_frequency = {}
    self.document_frequency = {}
    for document in self.documents:
      doc_id = document.sample_number
      for word in document.tokens:
        word_id = self.word_id[word]
        existing_tf = self.get_tf(word_id, doc_id)
        self.term_frequency[(word_id, doc_id)] = existing_tf + 1
        if existing_tf == 0: # Record the first occurrence of each word by incrementing df_i
          existing_df = self.get_df(word_id)
          self.document_frequency[word_id] = existing_df + 1

if __name__ == "__main__":
  TFIDFScorer(
    'tfidf.top',
    FileTokenizer('./qrys.txt').all(),
    FileTokenizer('./docs.txt').all(),
    2.0
  ).crunch_numbers().dump()
