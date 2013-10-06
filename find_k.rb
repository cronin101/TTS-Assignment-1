#!/usr/bin/env ruby
(10..30).map{ |i| i / 10.0 }.each do |k|
  `python best.py #{k}`
  precision = `./trec_eval.8.1/trec_eval -o -c -M1000 truth.rel best.top | grep 'Average precision' -A 1`.split("\n").last.split(/\s/).last
  `echo '#{k} => #{precision}' >> k_find.txt`
end
