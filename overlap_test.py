from tokenize import StringTokenizer
from overlap import OverlapScorer
import unittest
from flexmock import flexmock

class TestOverlapScorer(unittest.TestCase):
  def test_no_overlaps(self):
    '''Queries share no words with documents so the scores should be 0'''
    query_lines = [
      '1 How to roll up pants?',
      '2 How r babby formed?',
    ]
    queries = map(lambda l: StringTokenizer(l), query_lines)
    document_lines = [
      '1 Peeling vegetables is great',
      '2 Learn python the hard way'
    ]
    documents = map(lambda l: StringTokenizer(l), document_lines)
    scorer = OverlapScorer('./output', queries, documents)

    for query in queries:
      for document in documents:
        self.assertEqual(scorer.query_score(query, document), 0)

  def test_with_overlaps(self):
    '''Should be scored correctly regardless of puctuation'''
    scorer = OverlapScorer('./output', [], [])
    self.assertEqual(scorer.query_score(
      StringTokenizer('66 How to clean combine-harvester?'),
      StringTokenizer("88 I've got a brand new combine harvester")
    ), 2)
    self.assertEqual(scorer.query_score(
      StringTokenizer('4 I ran MapReduce in the cloud and now I have too much data'),
      StringTokenizer('9 Not a cloud in sight as Usain Bolt ran the 100m sprint')
    ), 4)

if __name__ == '__main__':
      unittest.main()
