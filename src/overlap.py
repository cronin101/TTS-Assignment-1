from tokenize import *
import os

class OverlapScorer:
  def __init__(self, filename, queries, documents):
    self.filename = filename
    self.queries = list(queries)
    self.documents = list(documents)

  def __format_line(self, query_num, doc_num, score):
    values = map(lambda x: str(x), [query_num, doc_num, score])
    return string.join(values, ' 0 ') + ' 0' + os.linesep

  def query_score(self, query, document):
    return len(set(query.tokens).intersection(set(document.tokens)))

  def dump(self):
    with open(self.filename, 'w') as overlaps:
      for query in self.queries:
        for document in self.documents:
          score = self.query_score(query, document)
          line = self.__format_line(query.sample_number, document.sample_number, score)
          overlaps.write(line)

if __name__ == "__main__":
  OverlapScorer(
    '../rankings/overlap.top',
    FileTokenizer('../data/qrys.txt').all(),
    FileTokenizer('../data/docs.txt').all()
  ).dump()
