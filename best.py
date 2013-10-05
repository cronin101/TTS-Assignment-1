from tokenize import *
import os
from math import log
from itertools import chain
import nltk
import sys
from nltk.stem.lancaster import LancasterStemmer

class BestScorer:
  def __init__(self, filename, queries, documents, k=2.0, prf_num_top=4, prf_num_words=40):
    self.filename, self.k, self.prf_num_top, self.prf_num_words = filename, k, prf_num_top, prf_num_words
    self.queries, self.documents  = list(queries), list(documents)
    corpus = chain(self.queries, self.documents)
    self.stemmer = LancasterStemmer()
    stemmed_words = (self.stemmer.stem(w) for w in self.flatten_and_unique(corpus))
    self.unique_words = set(stemmed_words)
    self.word_id = dict((reversed(t) for t in enumerate(self.unique_words, 1)))
    self.document_by_id = dict(((d.sample_number, d) for d in self.documents))
    self.C = len(self.documents)
    self.stopwords = nltk.corpus.stopwords.words('english')

  def flatten_and_unique(self, list_of_lists):
    return set(chain.from_iterable((l.tokens for l in list_of_lists)))

  def crunch_numbers(self):
    self.compute_term_frequency()
    self.compute_k_d_over_avg_d()
    self.compute_query_scores()
    return self

  def __format_line(self, query_num, doc_num, score):
    values = (str(x) for x in [query_num, doc_num, score])
    return string.join(values, ' 0 ') + ' 0' + os.linesep

  def dump(self):
    with open(self.filename, 'w') as score_file:
      scores = ((q.sample_number, d, self.query_score[(q.sample_number, d)]) for q in self.queries for d in range(1, self.C))
      format_line = self.__format_line
      write_out = score_file.write
      for (query, document, score) in scores:
        if score > 0: write_out(format_line(query, document, score))

  def average_doc_len(self):
    '''Average length of documents in collection C'''
    return sum(len(d.tokens) for d in self.documents) / float(len(self.documents))

  def compute_k_d_over_avg_d(self):
    '''Document length normalisation factor'''
    self.kd_o_avd = {}
    average_doc_len = self.average_doc_len()
    kd_o_avd = self.kd_o_avd
    for document in self.documents:
      kd_o_avd[document.sample_number] = (self.k * len(document.tokens)) / average_doc_len

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
    return list(self.word_id[w] for w in query_words).count(word_id)

  def compute_query_scores(self):
    base_query_score = {}
    self.query_score = {}
    get_query_score = self.get_query_score
    for query in self.queries:
      for doc_id in range(1, self.C):
        base_query_score[(query.sample_number, doc_id)] = get_query_score(query, doc_id)

      scores = [(doc_id, self.get_query_score(query, doc_id)) for doc_id in range(1, self.C)]
      scores.sort(key=lambda (d, s): -s)
      top_documents = (d for (d, s) in scores[:self.prf_num_top])
      relevant_contents = list(chain.from_iterable(self.document_by_id[j].tokens for j in top_documents))
      filtered_contents = [w for w in relevant_contents if not w in self.stopwords]
      word_scores = [(word, relevant_contents.count(word) * self.idf(self.word_id[self.stemmer.stem(word)])) for word in set(filtered_contents)]
      word_scores.sort(key=lambda (w, c): -c)
      influential_words = (w for (w,c) in word_scores[:self.prf_num_words])
      extended_query = StringTokenizer(
        query.original_input + ' ' + string.join(list(influential_words)),
        split_and_merge=True, token_correction=True, include_3grams=False
      )
      for doc_id in range(1, self.C):
        self.query_score[(query.sample_number, doc_id)] = get_query_score(extended_query, doc_id) / len(extended_query.tokens)

  def get_query_score(self, query, document_id):
    '''Sum over all query words i of qtf . tf . idf'''
    query_words = query.tokens
    filtered_query = (w for w in query_words if not w in self.stopwords)
    processed_query = list(self.stemmer.stem(w) for w in filtered_query)
    q_word_ids = set((self.word_id[w] for w in processed_query))

    def qtf_tf_idf(word_id):
      i = word_id
      j = document_id
      return self.q_tf(i, processed_query) * self.tf(i, j) * self.idf(i)

    return sum(qtf_tf_idf(i) for i in q_word_ids)

  def compute_term_frequency(self):
    '''For each document, record the number of times that each word appears'''
    self.term_frequency = {}
    self.document_frequency = {}
    stem = self.stemmer.stem
    tf = self.get_tf
    df = self.get_df
    word2id = self.word_id
    for document in self.documents:
      doc_id = document.sample_number
      for word in document.tokens:
        word_id = word2id[stem(word)]
        existing_tf = tf(word_id, doc_id)
        self.term_frequency[(word_id, doc_id)] = existing_tf + 1
        if existing_tf == 0: # Record the first occurrence of each word by incrementing df_i
          self.document_frequency[word_id] = df(word_id) + 1

if __name__ == "__main__":
  BestScorer(
    'best.top',
    FileTokenizer('./qrys.txt', split_and_merge=True, token_correction=True, include_3grams=False).all(),
    FileTokenizer('./docs.txt', split_and_merge=True, token_correction=True, include_3grams=False).all(),
    k=float(sys.argv[1])
  ).crunch_numbers().dump()
