from tokenize import StringTokenizer
from tfidf import TFIDFScorer
from math import log
import unittest

class TestTFIDFScorer(unittest.TestCase):
  def test_C_constant(self):
    '''C should be set equal to the number of documents in the collection'''
    def tokenize(line): return StringTokenizer(line)

    document_lines = [
      '1 Peeling vegetables is great',
      '2 Learn python the hard way'
    ]
    documents = map(tokenize, document_lines)
    two_doc_scorer = TFIDFScorer('./output', [], documents, 2.0)
    self.assertEqual(two_doc_scorer.C, 2)

    document_lines = [
      '1 Peeling vegetables is great',
      '2 Learn python the hard way',
      '3 Nobody can hear you scream in space',
      '4 I like shorts they are comfortable and easy to wear'
    ]
    documents = map(tokenize, document_lines)
    four_doc_scorer = TFIDFScorer('./output', [], documents, 2.0)
    self.assertEqual(four_doc_scorer.C, 4)

  def test_unique_words_found(self):
    '''The list of unique words in the union of queries and documents should be correct'''
    def tokenize(line): return StringTokenizer(line)

    query_lines = [
     '1 one two three four',
     "2 thunderbirds are go"
    ]
    queries = map(tokenize, query_lines)
    document_lines = [
     '1 two plus two is four',
     '2 player one go'
    ]
    documents = map(tokenize, document_lines)
    scorer = TFIDFScorer('./output', queries, documents, 2.0)
    self.assertEqual(scorer.unique_words, set([
      'thunderbirds', 'three', 'two', 'four', 'player', 'plus', 'are', 'go', 'one', 'is'
    ]))

  def test_query_term_frequency(self):
    '''Query_term_frequency(i, q) is the number of times word i appears in query tokens q'''
    def tokenize(line): return StringTokenizer(line)

    query_lines = [
      '1 bees bees bees',
      '2 who let the dogs out?'
    ]
    queries = map(tokenize, query_lines)
    scorer = TFIDFScorer('./output', queries, [StringTokenizer('1 cats tho')], 2.0)

    bees_query_words = queries[0].tokens
    dogs_query_words = queries[1].tokens

    bees_id = scorer.word_id['bees']
    self.assertEqual(scorer.q_tf(bees_id, bees_query_words), 3)
    self.assertEqual(scorer.q_tf(bees_id, dogs_query_words), 0)

    dogs_id = scorer.word_id['dogs']
    self.assertEqual(scorer.q_tf(dogs_id, bees_query_words), 0)
    self.assertEqual(scorer.q_tf(dogs_id, dogs_query_words), 1)

    cats_id = scorer.word_id['cats']
    self.assertEqual(scorer.q_tf(cats_id, bees_query_words), 0)
    self.assertEqual(scorer.q_tf(cats_id, dogs_query_words), 0)

  def test_term_frequency(self):
    '''Term_frequency(i, j) is the number of times word i appears in document j'''
    def tokenize(line): return StringTokenizer(line)

    document_lines = [
      '1 bees bees bees bees',
      '2 romeo romeo why for art thou romeo',
      '3 romeo was stung by the bees'
    ]
    documents = map(tokenize, document_lines)
    scorer = TFIDFScorer('./output', [StringTokenizer('1 dogs')], documents, 2.0)
    scorer.crunch_numbers()

    bees_id = scorer.word_id['bees']
    self.assertEqual(scorer.get_tf(bees_id, 1), 4)
    self.assertEqual(scorer.get_tf(bees_id, 2), 0)
    self.assertEqual(scorer.get_tf(bees_id, 3), 1)

    romeo_id = scorer.word_id['romeo']
    self.assertEqual(scorer.get_tf(romeo_id, 1), 0)
    self.assertEqual(scorer.get_tf(romeo_id, 2), 3)
    self.assertEqual(scorer.get_tf(romeo_id, 3), 1)

    dogs_id = scorer.word_id['dogs']
    self.assertEqual(scorer.get_tf(dogs_id, 1), 0)
    self.assertEqual(scorer.get_tf(dogs_id, 2), 0)
    self.assertEqual(scorer.get_tf(dogs_id, 3), 0)

  def test_document_frequency(self):
    '''Document_ frequency(i) is the number of documents that word i appears in.
    Inverse_document_frequency(i) is log_2 of:
      The number of documents (C) divided by document frequency of word i (plus 1)'''
    def tokenize(line): return StringTokenizer(line)

    document_lines = [
      '1 bees bees bees bees',
      '2 romeo romeo why for art thou romeo',
      '3 romeo was stung by the bees',
      '4 bees are not your friend '
    ]
    documents = map(tokenize, document_lines)
    scorer = TFIDFScorer('./output', [StringTokenizer('1 cats')], documents, 2.0)
    scorer.crunch_numbers()

    bees_id = scorer.word_id['bees']
    self.assertEqual(scorer.get_df(bees_id), 3)
    inverse_bees_frequency = scorer.C / 1 + 3
    self.assertEqual(scorer.idf(bees_id), log(inverse_bees_frequency, 2))

    romeo_id = scorer.word_id['romeo']
    self.assertEqual(scorer.get_df(romeo_id), 2)
    inverse_romeo_frequency = scorer.C / 1 + 2
    self.assertEqual(scorer.idf(romeo_id), log(inverse_romeo_frequency, 2))

    why_id = scorer.word_id['why']
    self.assertEqual(scorer.get_df(why_id), 1)
    inverse_why_frequency = scorer.C / 1 + 1
    self.assertEqual(scorer.idf(why_id), log(inverse_why_frequency, 2))

    cats_id = scorer.word_id['cats']
    self.assertEqual(scorer.get_df(cats_id), 0)
    inverse_cats_frequency = scorer.C / 1 + 0
    self.assertEqual(scorer.idf(cats_id), log(inverse_cats_frequency, 2))

  def test_average_document_length(self):
    def tokenize(line): return StringTokenizer(line)

    document_lines = [
      '1 bees bees bees bees',
      '2 romeo romeo romeo romeo',
      '4 bees are your friends'
    ]
    documents = map(tokenize, document_lines)
    length_four_scorer = TFIDFScorer('./output', [], documents, 2.0)
    self.assertEqual(length_four_scorer.average_doc_len(), 4.0)

    document_lines = [
      '1 bees bees bees bees',
      '2 romeo romeo',
    ]
    documents = map(tokenize, document_lines)
    length_three_scorer = TFIDFScorer('./output', [], documents, 2.0)
    self.assertEqual(length_three_scorer.average_doc_len(), 3.0)

if __name__ == '__main__':
      unittest.main()
