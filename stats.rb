#!/usr/bin/ruby

%w{overlap tfidf best}.each do |ranker|
  puts ranker.capitalize << ' ranking:'
  puts "\t" << `./trec_eval -o -c -M1000 ./data/truth.rel ./rankings/#{ranker}.top | grep 'Average precision' -A 1`
end

