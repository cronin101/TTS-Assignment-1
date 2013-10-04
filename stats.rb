#!/usr/bin/env ruby

%w{overlap}.each { |technique| puts "#{technique.capitalize}: #{`./trec_eval.8.1/trec_eval -o -c -M1000 truth.rel #{technique}.top | grep 'Average precision' -A 1`}\n" }
