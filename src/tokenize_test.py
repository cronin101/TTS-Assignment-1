from tokenize import StringTokenizer
import unittest

class TestStringTokenizer(unittest.TestCase):
  def test_tokenizing_just_words(self):
    '''The leading number should become the sample number and
    the remaining tokens should be downcased'''
    just_words = StringTokenizer('1 The cow jumped over the moon')
    tokens = ['the', 'cow', 'jumped', 'over', 'the', 'moon']
    self.assertEqual(just_words.number, 1)
    self.assertEqual(just_words.tokens, tokens)

  def test_splitting_on_punctuation(self):
    '''Words containing punctuation should be split into a token for each continuous segment'''
    with_punc = StringTokenizer(
      "9 Swedish House Mafia's Don't You Worry Child, I can't stop over-playing that song."
    )
    tokens = [
      'swedish', 'house', 'mafia', 's', 'don', 't', 'you', 'worry', 'child',
      'i', 'can', 't', 'stop', 'over', 'playing', 'that', 'song'
    ]
    self.assertEqual(with_punc.number, 9)
    self.assertEqual(with_punc.tokens, tokens)

  def test_split_and_merge(self):
    '''Words containing punctuation should be split into tokens but also added in full'''
    split_and_merge = StringTokenizer(
      "16 Message-passing is great, ***** don't forget to keep data local though!",
      split_and_merge = True
    )
    tokens = [
      'message', 'passing', 'is', 'great', 'don', 't', 'forget', 'to',
      'keep', 'data', 'local', 'though', 'message-passing', "don't"
    ]
    self.assertEqual(split_and_merge.number, 16)
    self.assertEqual(split_and_merge.tokens, tokens)

if __name__ == '__main__':
      unittest.main()
