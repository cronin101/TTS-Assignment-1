from tokenize import *
import os
from math import log
from itertools import chain
import sys

class BestScorer:
  def __init__(self, filename, queries, documents, k=2.0, prf_num_top=4, prf_num_words=65):
    self.filename, self.k, self.prf_num_top = filename, k, prf_num_top
    self.prf_num_words, self.queries, self.documents  = prf_num_words, list(queries), list(documents)

    def flatten_and_unique(list_of_lists):
      return set(chain.from_iterable((l.tokens for l in list_of_lists)))
    self.unique_words = set(flatten_and_unique(chain(self.queries, self.documents)))

    self.word_id = dict((reversed(t) for t in enumerate(self.unique_words, 1)))
    self.document_by_id = [0] + self.documents
    self.C = len(self.documents)

  def dump(self):
    def format_line(query_num, doc_num, score):
      values = (str(x) for x in [query_num, doc_num, score])
      return string.join(values, ' 0 ') + ' 0' + os.linesep

    with open(self.filename, 'w') as score_file:
      def _s(query_num, doc_num):
        return self.query_score[query_num - 1][doc_num - 1]
      queries = self.queries
      scores = ((q.number, d, _s(q.number, d)) for q in queries for d in xrange(1, self.C))
      for (query, document, score) in scores:
        if score > 0: score_file.write(format_line(query, document, score))

  def crunch(self):
    def compute_term_frequency():
      num_words, num_docs = len(self.unique_words), len(self.documents)
      self.term_frequency, self.document_frequency = [[0] * num_docs for word in xrange(num_words)], [0] * num_words
      for (doc_id, counts) in ((doc.number, doc.counter) for doc in self.documents):
        for (word_id, word) in ((self.word_id[key], key) for key in counts.keys()):
          self.term_frequency[word_id - 1][doc_id - 1] += counts[word]
          self.document_frequency[word_id - 1] += 1

    def compute_k_d_over_avg_d():
      average_doc_len = sum(len(d.tokens) for d in self.documents) / float(len(self.documents))
      self.kd_o_avd = [0] + [(self.k * len(d.tokens)) / average_doc_len for d in self.documents]

    def compute_query_scores():
      def tf(word_id, document_id):
        tf = self.term_frequency[word_id - 1][document_id - 1]
        return tf / (tf + self.kd_o_avd[document_id])

      def idf(word_id):
        return log(self.C / (1.0 + self.document_frequency[word_id - 1]), 2)

      def get_query_score(query, document_id):
        q_tf, j = query.counter, document_id
        return sum(q_tf[w] * tf(i, j) * idf(i) for (i, w) in ((self.word_id[key], key) for key in query.counter.keys()))

      self.query_score = [[0] * len(self.documents) for word in self.unique_words]
      for query in self.queries:
        scores = [(doc_id, get_query_score(query, doc_id)) for doc_id in xrange(1, self.C)]
        scores.sort(key=lambda (d, s): -s)
        top_documents = (d for (d, s) in scores[:self.prf_num_top])

        relevant_contents = reduce(lambda x, y: x + y, (self.document_by_id[j].counter for j in top_documents))
        word_scores = [(key, relevant_contents[key] * idf(self.word_id[key])) for key in relevant_contents.keys()]
        word_scores.sort(key=lambda (w, c): -c)
        new_tokens = word_scores[:self.prf_num_words]
        expanded = ExpandedQuery(
          query.number,
          query.tokens + [w for (w, c) in new_tokens],
          query.counter + Counter(dict((w, 1) for (w, c) in new_tokens))
        )

        for doc_id in xrange(1, self.C):
          self.query_score[query.number - 1][doc_id - 1] = get_query_score(expanded, doc_id) / len(expanded.tokens)

    compute_term_frequency()
    compute_k_d_over_avg_d()
    compute_query_scores()
    return self

if __name__ == "__main__":
  BestScorer(
    '../rankings/best.top',
    FileTokenizer('../data/qrys.txt',
      remove_stopwords=True, stem=True, split_and_merge=True,
      token_correction=True, include_3grams=False, repeat_titles=True
    ).all(),
    FileTokenizer('../data/docs.txt',
      remove_stopwords=True, stem=True, split_and_merge=True,
      token_correction=True, include_3grams=False, repeat_titles=True
    ).all(),
    k=1.7
  ).crunch().dump()
