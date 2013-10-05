from tokenize import StringTokenizer
from tfidf import TFIDFScorer
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

    def test_term_frequency(self):
      '''Term frequency(i, j) is the number of times word i appears in document j'''
      def tokenize(line): return StringTokenizer(line)

      document_lines = [
        '1 bees bees bees bees',
        '2 romeo romeo why for art thou romeo',
        '3 romeo was stung by the bees'
      ]
      documents = map(tokenize, document_lines)
      scorer = TFIDFScorer('./output', [], documents, 2.0)

      bees_id = scorer.word_id['bees']
      self.assertEqual(scorer.get_tf(bees_id, 1), 4)
      self.assertEqual(scorer.get_tf(bees_id, 2), 0)
      self.assertEqual(scorer.get_tf(bees_id, 3), 1)

      romeo_id = scorer.word_id['romeo']
      self.assertEqual(scorer.get_tf(romeo_id, 1), 0)
      self.assertEqual(scorer.get_tf(romeo_id, 2), 3)
      self.assertEqual(scorer.get_tf(romeo_id, 3), 1)

if __name__ == '__main__':
      unittest.main()
