from tokenize import *
import os
from math import log
from itertools import chain
import sys

class BestScorer:
  def __init__(self, filename, queries, documents, k=2.0, prf_num_top=4, prf_num_words=60):
    self.filename, self.k, self.prf_num_top, self.prf_num_words = filename, k, prf_num_top, prf_num_words
    self.queries, self.documents  = list(queries), list(documents)
    self.unique_words = set(self.flatten_and_unique(chain(self.queries, self.documents)))
    self.word_id = dict((reversed(t) for t in enumerate(self.unique_words, 1)))
    self.document_by_id = [0] + self.documents
    self.C = len(self.documents)

  def flatten_and_unique(self, list_of_lists):
    return set(chain.from_iterable((l.tokens for l in list_of_lists)))

  def crunch_numbers(self):
    return self.compute_term_frequency().compute_k_d_over_avg_d().compute_query_scores()

  def __format_line(self, query_num, doc_num, score):
    values = (str(x) for x in [query_num, doc_num, score])
    return string.join(values, ' 0 ') + ' 0' + os.linesep

  def dump(self):
    with open(self.filename, 'w') as score_file:
      scores = ((q.sample_number, d, self.query_score[q.sample_number - 1][d - 1]) for q in self.queries for d in xrange(1, self.C))
      for (query, document, score) in scores:
        if score > 0: score_file.write(self.__format_line(query, document, score))

  def average_doc_len(self):
    '''Average length of documents in collection C'''
    return sum(len(d.tokens) for d in self.documents) / float(len(self.documents))

  def compute_k_d_over_avg_d(self):
    '''Document length normalisation factor'''
    average_doc_len = self.average_doc_len()
    self.kd_o_avd = [0] + [(self.k * len(d.tokens)) / average_doc_len for d in self.documents]
    return self

  def get_tf(self, word_id, document_id):
    '''Occurrences of word i in document j'''
    return self.term_frequency[word_id - 1][document_id - 1]

  def get_df(self, word_id):
    '''Number of documents that word i appears in'''
    return self.document_frequency[word_id - 1]

  def tf(self, word_id, document_id):
    '''Normalised term frequency component:
    Repetitions less important, except in long documents'''
    tf = self.get_tf(word_id, document_id)
    return tf / (tf + self.kd_o_avd[document_id])

  def idf(self, word_id):
    '''Inverse document frequency:
    Rare words are considered more important for ranking'''
    return log(self.C / (1.0 + self.get_df(word_id)), 2)

  def compute_query_scores(self):
    self.query_score = [[0] * len(self.documents) for word in self.unique_words]
    for query in self.queries:
      scores = [(doc_id, self.get_query_score(query, doc_id)) for doc_id in xrange(1, self.C)]
      scores.sort(key=lambda (d, s): -s)
      top_documents = (d for (d, s) in scores[:self.prf_num_top])

      relevant_contents = reduce(lambda x, y: x + y, (self.document_by_id[j].counter for j in top_documents))
      word_scores = [(key, relevant_contents[key] * self.idf(self.word_id[key])) for key in relevant_contents.keys()]
      word_scores.sort(key=lambda (w, c): -c)
      new_tokens = word_scores[:self.prf_num_words]
      expanded = ExpandedQuery(
        query.sample_number,
        query.tokens + [w for (w, c) in new_tokens],
        query.counter + Counter(dict((w, 1) for (w, c) in new_tokens))
      )

      for doc_id in xrange(1, self.C):
        self.query_score[query.sample_number - 1][doc_id - 1] = self.get_query_score(expanded, doc_id) / len(expanded.tokens)
    return self

  def get_query_score(self, query, document_id):
    '''Sum over all query words i of qtf . tf . idf'''
    q_tf, tf, idf, j = query.counter, self.tf, self.idf, document_id
    return sum(q_tf[w] * tf(i, j) * idf(i) for (i, w) in ((self.word_id[key], key) for key in query.counter.keys()))

  def compute_term_frequency(self):
    '''For each document, record the number of times that each word appears'''
    self.term_frequency, self.document_frequency = [[0] * len(self.documents) for word in self.unique_words], [0] * len(self.unique_words)
    for (doc_id, counts) in ((doc.sample_number, doc.counter) for doc in self.documents):
      for (word_id, word) in ((self.word_id[key], key) for key in counts.keys()):
        self.term_frequency[word_id - 1][doc_id - 1] += counts[word]
        self.document_frequency[word_id - 1] += 1
    return self

if __name__ == "__main__":
  BestScorer(
    'best.top',
    FileTokenizer('./qrys.txt', remove_stopwords=True, stem=True, split_and_merge=True, token_correction=True, include_3grams=False, repeat_titles=True).all(),
    FileTokenizer('./docs.txt', remove_stopwords=True, stem=True, split_and_merge=True, token_correction=True, include_3grams=False, repeat_titles=True).all(),
    k=1.7
  ).crunch_numbers().dump()
