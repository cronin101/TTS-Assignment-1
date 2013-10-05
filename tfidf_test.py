from tokenize import StringTokenizer
from tfidf import TFIDFScorer
import unittest

class TestTFIDFScorer(unittest.TestCase):
  def test_C_constant(self):
    '''C should be set equal to the number of documents in the collection'''
    document_lines = [
      '1 Peeling vegetables is great',
      '2 Learn python the hard way'
    ]
    documents = map(lambda l: StringTokenizer(l), document_lines)
    two_doc_scorer = TFIDFScorer('./output', [], documents, 2.0)
    self.assertEqual(two_doc_scorer.C, 2)

    document_lines = [
      '1 Peeling vegetables is great',
      '2 Learn python the hard way',
      '3 Nobody can hear you scream in space',
      '4 I like shorts they are comfortable and easy to wear'
    ]
    documents = map(lambda l: StringTokenizer(l), document_lines)
    four_doc_scorer = TFIDFScorer('./output', [], documents, 2.0)
    self.assertEqual(four_doc_scorer.C, 4)


if __name__ == '__main__':
      unittest.main()
