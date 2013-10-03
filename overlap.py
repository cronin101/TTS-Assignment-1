from tokenize import *
import os

class OverlapScorer:
  def __init__(self, filename, queries_file, documents_file):
    self.filename = filename
    self.queries = FileTokenizer(queries_file)
    self.documents = FileTokenizer(documents_file)

  def __format_line(self, query_num, doc_num, score):
    values = map(lambda x: str(x), [query_num, doc_num, score])
    return string.join(values, ' 0 ') + ' 0' + os.linesep

  def dump(self):
    with open(self.filename, 'w') as overlaps:
      for query in self.queries.tokenized_lines():
        for document in self.documents.tokenized_lines():
          score = len(set(query.tokens).intersection(set(document.tokens)))
          line = self.__format_line(query.sample_number, document.sample_number, score)
          overlaps.write(line)

if __name__ == "__main__":
  OverlapScorer('./overlap.top', './qrys.txt', './docs.txt').dump()
