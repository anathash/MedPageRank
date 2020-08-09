import csv
import datetime
import itertools
import math
import random
import sys
from enum import Enum

import numpy as np
import os
from itertools import combinations, permutations

from metapub import PubMedFetcher

from preprocessing.HIndex import HIndex
from preprocessing.PaperBuilder import PaperBuilder
from preprocessing.PaperCache import PaperCache
from preprocessing.PaperFeatures import PaperFeatures

HINDEX_MAX = 1096

class Features(Enum):
    H_INDEX = 'h_index'
    STANCE_SCORE = 'stance_score'
    CURRENT_SCORE = 'current_score'
    CITATION_COUNT = 'citation_count'
    CONTRADICTED_BY_LATER = 'contradicted_by_later'
    RECENT_WEIGHTED_H_INDEX = 'recent_weighted_h_index'
    RECENT_WEIGHTED_CITATION_COUNT = 'recent_weighted_citation_count'
    CITATIONS_HINDEX_WAVG = 'citations_hindex_wavg'

class FeaturesGenerator:

    def __init__(self, paper_builder):
        self.paper_builder = paper_builder

#    def __init__(self, cache_file_name, hIndex_filename, no_index_filename):
#        paper_cache = PaperCache(cache_file_name)
#        hIndex = HIndex(hIndex_filename)
#        fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
#        self.paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, no_index_filename)

    #compute the average citation score with more weight to recent citations
    def compute_moving_averages(self, paper, review_year, config):
        features_collection_end_year = min(datetime.datetime.now().year, config.review_end_range + review_year)
        review_range = features_collection_end_year - paper.year + 1

        features_collection_start_year = paper.year
        years_hIndex_acc = [0] * review_range
        for pmid in paper.pm_cited:
            citing_paper = self.paper_builder.build_paper(pmid)
            if not citing_paper:
                continue
            citing_paper_year = int(citing_paper.year)
            #sometimes the epub is early
            if citing_paper_year == features_collection_start_year-1:
                print('GAP FIX ' + pmid)
                citing_paper_year = features_collection_start_year
                #pubmed linking erorrs
            if citing_paper_year < features_collection_start_year:
                print(str(citing_paper_year) + '<' + str(features_collection_start_year) + 'for paper ' + paper.pmid
                      + 'and citation ' +  pmid)
                continue
                #assert (citing_paper_year >= features_collection_start_year)

            if citing_paper_year > features_collection_end_year:
                continue
            year_gap = features_collection_end_year - int(citing_paper_year)
            years_hIndex_acc[year_gap] += citing_paper.h_index
        alpha = 0.5
        ewa = 0
        l = len(years_hIndex_acc)
        for i in range(l, 0, -1):
            ewa = alpha*years_hIndex_acc[i-1] + (1-alpha)*ewa
        return ewa

        #return numpy.average(years_hIndex_acc, weights = range(review_range, 0, -1))

    #compute the average citation score with more weight to recent citations
    def compute_moving_averages2(self, paper, review_year, config):
        counter = {}
        years_hIndex_acc = {}
        review_range = config.review_end_range + config.review_start_range + 1
        features_collection_end_year = int(review_year) + config.review_end_range
        features_collection_start_year = int(review_year) - config.review_start_range
        for i in range(0, review_range):
            counter[i] = 0
            years_hIndex_acc[i] = 0
        for pmid in paper.pm_cited:
            citing_paper = self.paper_builder.build_paper(pmid)
            if not citing_paper:
                continue
            citing_paper_year = int(citing_paper.year)
            assert (citing_paper_year >= features_collection_start_year)
            if citing_paper_year > features_collection_end_year:
                continue
            year_gap = features_collection_end_year - int(citing_paper.year)
            counter[year_gap] += 1 #CHECK
            years_hIndex_acc[year_gap] += citing_paper.h_index
        avg_hIndex = {}
        for year, acc in years_hIndex_acc.items():
            avg_hIndex[year] = 0 if acc ==0 else acc / counter[year]
        hIndex_wavg = 0
        wavg = 0
        for year_gap, avg_hIndex_per_year in avg_hIndex.items():
            hIndex_wavg += (review_range - year_gap +1)*avg_hIndex[year_gap]
            wavg += (review_range - year_gap +1)*counter[year_gap]
        range_sum = sum(x for x in range(0, review_range))
        return hIndex_wavg/range_sum, wavg/range_sum

    def is_contradicted_by_later(self, paper, papers):
        for citing_pmid in paper.pm_cited:
            if citing_pmid in papers.keys():
                citing_score = papers[citing_pmid].stance_score
                if citing_score - paper.stance_score < -1:
                    return int(True)
        return int(False)

    def get_citation_count(self, config, review_year, paper):
        count = 0
        features_collection_end_year = min(datetime.datetime.now().year, config.review_end_range + review_year)
        for pmid in paper.pm_cited:
            citing_paper = self.paper_builder.build_paper(pmid)
            if not citing_paper:
                continue
            citing_paper_year = int(citing_paper.year)
            if citing_paper_year > features_collection_end_year:
                continue
            count +=1
        return count

    def generate_dummy_feaures(self, stance, reverse =False):
        h_index = random.randrange(0, HINDEX_MAX/2)
        if reverse:
            h_index = random.randrange(0, HINDEX_MAX / 4)

        h_index_normed = h_index * 100 / HINDEX_MAX
        features = {Features.H_INDEX: h_index_normed, Features.STANCE_SCORE: stance}
        current_score = random.randrange(0, 10)
        features[Features.CURRENT_SCORE] = current_score
        features[Features.RECENT_WEIGHTED_H_INDEX] = current_score * h_index_normed  # paper.h_index

        if reverse:
            citation_count = random.randrange(0, 100)
        else:
            citation_count = random.randrange(0, 250)
        features[Features.RECENT_WEIGHTED_CITATION_COUNT] = (current_score * citation_count)
        features[Features.CITATION_COUNT] = citation_count
        return features


    def single_paper_feature_generator(self, paper, review_year, config, papers):
        #features = PaperFeatures(paper.h_index, paper.stance_score)
        if  paper.h_index <= 1:
            print ('NO HINDEX FOR  PAPER WITH PMID ' + paper.pmid + ' published in ' + paper.journal )
        h_index_normed = paper.h_index *100/HINDEX_MAX
        features = {Features.H_INDEX: paper.h_index *100/HINDEX_MAX, Features.STANCE_SCORE: paper.stance_score}
        review_year = int(review_year)
        year_gap = int(review_year + config.review_end_range - int(paper.year))
        review_range = config.review_end_range + config.review_start_range + 1
        current_score = review_range - year_gap
#        if not 0 <= year_gap <= config.cochrane_search_range:
#            print(paper.pmid)
#        assert (0 <= year_gap <= config.cochrane_search_range)
        features[Features.CURRENT_SCORE] = current_score
        features[Features.RECENT_WEIGHTED_H_INDEX] = current_score*h_index_normed#paper.h_index

        #citation_count = 0 if not paper.pm_cited else len(paper.pm_cited)
        citation_count = 0 if not paper.pm_cited else self.get_citation_count(config, review_year, paper)
        features[Features.RECENT_WEIGHTED_CITATION_COUNT] = (current_score * citation_count)
        features[Features.CITATION_COUNT] = citation_count
        #ewa = self.compute_moving_averages(paper, review_year, config)
        #features[Features.CITATIONS_HINDEX_WAVG] = ewa
        #citations_wavg = self.compute_moving_averages(paper, review_year, config)
        #features[Features.CITATIONS_WAVG].add_citations_hIndex_weighted_feature(hIndex_wavg)
        #features.add_citations_wighted_average_feature(wavg)
        #TODO - remove after Experiment
        #features[Features.CONTRADICTED_BY_LATER] = self.is_contradicted_by_later(paper, papers)
        return features


    def generate_features(self, files, review_year, config):
        featured_papers = []
        papers = {}
        num_ir = 0
        rel = 0
        for file in files:
            if not file.endswith('.csv'):
                filename = file.strip() + '.csv'
            else:
                filename = file
            if not os.path.isfile(filename):
                continue
            with open(filename, encoding='utf-8', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    url = row['document_url']
                    if not url:
                        continue
                    score = row['category']
                    if score:
                        score = int(score)
                        if score <= 0:
                            num_ir += 1
                            continue
                        else:
                            rel += 1
                    pmid = row['document_url'].split('/')[-1].split('\n')[0]
                    paper = self.paper_builder.build_paper(pmid)
                    if not paper:
                        continue
                    paper.set_stance_score(score)
                    papers[pmid] = paper
                examples_collected = 0
        for pmid, paper in papers.items():
            featured_paper = self.single_paper_feature_generator(paper, review_year, config, papers)
            if featured_paper:
                featured_papers.append(featured_paper)
                examples_collected += 1
        return  featured_papers, num_ir, rel
  #          if examples_collected == config.examples_per_file:
  #              break

#        if config.rel == "rel":
#            return label, featured_papers, rel
#        else:
#            return label, featured_papers, num_ir

    def generate_examples_per_query(self, files, review_year, config, coch_pubmed_url):
        return self.generate_features(files, review_year, config)
      #  if config.perm:
      #     return permutations(featured_papers, config.group_size), rel
      #  else:
      #      return featured_papers, rel

    def write_group_csv_file(self, output_dir,name, examples, labels):
        fieldnames = ['query', 'label']
        fieldnames.extend(list(list(examples.values())[0].keys()))
        with open(output_dir + name, 'w', encoding='utf-8', newline='') as outputCsv:
            wr = csv.DictWriter(outputCsv, fieldnames=fieldnames)
            wr.writeheader()
            for query, features in examples.items():
                row = {'query':query, 'label':labels[query]}
                for k, v in features.items():
                    row[k] = str(v)
                wr.writerow(row)
        with open(output_dir + 'labels_' + name , 'w', encoding='utf-8', newline='') as labelsCSv:
            wr = csv.DictWriter(labelsCSv, fieldnames=['query', 'value_label'])
            wr.writeheader()
            for query, features in examples.items():
                row = {'query': query, 'value_label': labels[query]}
                wr.writerow(row)


    def write_csv_file(self, output_dir, query, group_size, examples, label):
        fieldnames = ['label']
        examples = list(examples)
        fields =list(examples[0][0].keys())
        for i in range(0, group_size):
            for field in fields:
                fieldnames.append(field.value + str(i + 1))
        with open(output_dir + query +'.csv', 'w', encoding='utf-8', newline='') as outputCsv:
            wr = csv.DictWriter(outputCsv, fieldnames=fieldnames)
            wr.writeheader()
            for example in examples:
                row = {}
                if group_size > 1:
                    assert (len(example) == group_size)
                    row['label'] = label
                else:
                    row['label'] = math.fabs(int(label) - example.stance_score)
                for i in range(0, group_size):
                    if group_size > 1:
                        attr = example[i]
                    else:
                        attr = example
                    for field in fields:
                        row[field.value + str(i+1)] = attr[field]
                wr.writerow(row)


    def write_readme(self, output_dir, long_dir, short_dir, config):
        with open(output_dir + 'README.txt', 'w', encoding='utf-8', newline='') as readme:
            readme.write('long_dir = ' + long_dir + '\n')
            readme.write('short_dir = ' + short_dir + '\n')
            for att, val in config.__dict__.items():
                readme.write(att + ' = ' + str(val) + '\n')
#            readme.write('review_start_range = ' + str(config.review_start_range) + '\n')
#            readme.write('review_end_range = ' + str(config.review_end_range) + '\n')
#            readme.write('group_size = ' + str(config.group_size) + '\n')
#            readme.write('examples_per_file = ' + str(config.examples_per_file) + '\n')
#            readme.write('include_irrelevant = ' + str(config.include_irrelevant) + '\n')

    def write_pairs_csv_file(self, output_dir, query, examples, fields, get_diff, get_attr, config):
        pairs = permutations(examples, 2)
        fieldnames = ['label']
        if query == 'all':
            fieldnames.extend(['query1','query2'])
        for i in range(0, 2):
            for field in fields:
                if field == Features.STANCE_SCORE and config.remove_stance:
                    continue
                fieldnames.append(field.value + str(i + 1))
        with open(output_dir + query + '.csv', 'w', encoding='utf-8', newline='') as outputCsv:
            wr = csv.DictWriter(outputCsv, fieldnames=fieldnames)
            wr.writeheader()
            for pair in pairs:
                diff1 = get_diff(pair[0])
                diff2 = get_diff(pair[1])
#                diff1 = math.fabs(pair[0][0].stance_score - pair[0][1])
#                diff2 = math.fabs(pair[1][0].stance_score - pair[1][1])
                if diff1 == diff2:
                    pref = 0
                elif  diff1 < diff2:
                    pref = 1
                else:
                    pref = 2
                if query == 'all':
                    row = {'label': pref, 'query1': pair[0][2], 'query2':pair[1][2]}
                else:
                    row = {'label': pref}
                for i in range(0, 2):
                    attr = get_attr(pair[i])#pair[i][0].__dict__
                    for field in fields:
                        if field == Features.STANCE_SCORE and config.remove_stance:
                            continue
                        row[field.value + str(i + 1)] = attr[field]
                wr.writerow(row)

    def generate_examples_by_single(self, output_dir, queries, long_dir, short_dir, config):
        config.group_size = 1
        self.setup_dir(output_dir, long_dir, short_dir, config)
        fields = list(PaperFeatures.__annotations__.keys())
        if config.remove_stance:
            fields.remove('stance_score')
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, rel = self.get_examples(config, row, short_dir)
                self.write_csv_file(output_dir, row['short query'], 1, examples, label, fields = fields)

    def generate_examples_by_right_wrong_groups(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        all_examples = []
        config.group_size = 2
        fields = list(PaperFeatures.__annotations__.keys())
        if config.remove_stance:
            fields.remove('stance_score')
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, rel = self.get_examples(config, row, short_dir)
                all_examples.extend([(e, label, row['short query']) for e in examples])
                # output_dir, query, examples, fields, get_diff, get_attr
                get_diff = lambda x: math.fabs(x.stance_score - int(label))
                get_attr = lambda p: p.__dict__
                self.write_pairs_csv_file(output_dir, row['short query'], examples, fields, get_diff, get_attr,  config)
            get_diff = lambda x: math.fabs(x[0].stance_score - int(x[1]))
            get_attr = lambda p: p[0].__dict__
            self.write_pairs_csv_file(output_dir, 'all', all_examples, fields, get_diff, get_attr, config)


    def generate_examples_by_pairs(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        all_examples = []
        config.group_size = 2
        config.perm = False
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, rel = self.get_examples(config,  row, short_dir)
                all_examples.extend([(e, label, row['short query']) for e in examples])
                #output_dir, query, examples, fields, get_diff, get_attr
                get_diff = lambda x: math.fabs(x[Features.STANCE_SCORE] - int(label))
                get_attr = lambda p: p
                fields = list(examples[0].keys())
                self.write_pairs_csv_file(output_dir, row['short query'],  examples, fields, get_diff, get_attr,  config)
            get_diff = lambda x: math.fabs(x[0][Features.STANCE_SCORE] - int(x[1]))
            get_attr = lambda p: p[0]
            self.write_pairs_csv_file(output_dir, 'all', all_examples, fields, get_diff, get_attr, config)

    def generate_examples(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, __ = self.get_examples(config,  row, short_dir)
                self.write_csv_file(output_dir, row['short query'], config.group_size, examples, label)

    def get_class_old(self, score):  # TODO - define welll
        if score < 3:
            return 1
        elif score < 5:
            # return 3
            return 3
        else:
            return 5


    def get_class(self, score):  # TODO - define welll
        if score < 3:
            return 1
        elif score < 5:
            # return 3
            return 2
        else:
            return 3


    def gen_dist_features_with_maj(self, majority_file, output_dir):
        group_features = {}
        labels = {}
        categories = [-1, 1, 2, 3, 4, 5]
        with open(majority_file, encoding='utf-8', newline='') as majority_csv:
            reader = csv.DictReader(majority_csv)
            for row in reader:
                query = row['query']
                dist_features = {i: int(row[str(i)]) for i in categories}
                sum_votes = sum(dist_features.values())
                group_features[query] = {'sigma' + str(i): dist_features[i] / sum_votes for i in categories}
                labels[query] = row['label']
                group_features[query]['sigmaz'] = (group_features[query]['sigma5'] - group_features[query][
                    'sigma-1']) / 6
        self.write_group_csv_file(output_dir, 'dist.csv', group_features, labels)

    def gen_dist_features(self, majority_file, output_dir):
        group_features = {}
        labels = {}
        categories = [1,2,3,4,5]
        with open(majority_file, encoding='utf-8', newline='') as majority_csv:
            reader = csv.DictReader(majority_csv)
            for row in reader:
                query = row['query']
                dist_features = {i : int(row[str(i)]) for i in range(1,6)}
                sum_votes = sum(dist_features.values())
                group_features[query] = {'theta'+str(i):dist_features[i]/sum_votes for i in range(1,6)}
                labels[query] = row['label']
                group_features[query]['thetaz'] = (group_features[query]['theta5'] - group_features[query]['theta1']) / 5
        self.write_group_csv_file(output_dir, 'dist.csv', group_features, labels)

    def gen_majority_vote(self, output_dir,  queries, long_dir,  short_dir, config):
        config.group_size = 1
        config.perm = False
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        stance_shrinking = {1: 1, 2: 1, 3: 3, 4: 5, 5: 5}
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                query = row['long query']
                examples, label, num_ir, rel = self.get_examples(config,  row, short_dir, [])
                if not examples:
                    continue
                labels[query] = label
                group_features_list = {1: 0, 2:0, 3: 0, 4:0, 5: 0}
                group_features[query] = {}

                for example in examples:
                    for k, v in example.items():
                        if k == Features.STANCE_SCORE:
                            group_features_list[v] += 1
                            stance = stance_shrinking[v]
                            #group_features_list[stance] += 1

                sorted_stance = sorted(group_features_list.items(), key=lambda kv: kv[1], reverse=True)
                majority = sorted_stance[0][0]
                majority_class = self.get_class(majority)
                print(query)
                int_label = int(label)
                label_class_group =  self.get_class(int_label)
                group_features[query]['majority_value'] = majority
                group_features[query]['majority_class'] = majority_class

                group_features[query]['1'] = group_features_list[1]
                group_features[query]['2'] = group_features_list[2]
                group_features[query]['3'] = group_features_list[3]
                group_features[query]['4'] = group_features_list[4]
                group_features[query]['5'] = group_features_list[5]


                group_features[query]['accuracy'] = int(label_class_group == majority_class)
                group_features[query]['error'] = math.fabs(int_label - majority)


        self.write_group_csv_file(output_dir, 'majority.csv', group_features, labels)


    def gen_majority_with_ir(self, output_dir,  queries, long_dir,  short_dir, config):
        config.group_size = 1
        config.perm = False
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        stance_shrinking = {1: 1, 2: 1, 3: 3, 4: 5, 5: 5}
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                query = row['long query']
                examples, label, rel = self.get_examples(config,  row, short_dir, [])
                if not examples:
                    continue
                labels[query] = label
                group_features_list = {-1:0, 1: 0, 2:0, 3: 0, 4:0, 5: 0}
                group_features[query] = {}

                for example in examples:
                    for k, v in example.items():
                        if k == Features.STANCE_SCORE:
                            group_features_list[v] += 1
                            #stance = stance_shrinking[v]
                            #group_features_list[stance] += 1

                sorted_stance = sorted(group_features_list.items(), key=lambda kv: kv[1], reverse=True)
                majority = sorted_stance[0][0]
                majority_class = self.get_class(majority)
                print(query)
                int_label = int(label)
                label_class_group =  self.get_class(int_label)
                group_features[query]['majority_value'] = majority
                group_features[query]['majority_class'] = majority_class

                group_features[query]['1'] = group_features_list[1]
                group_features[query]['2'] = group_features_list[2]
                group_features[query]['3'] = group_features_list[3]
                group_features[query]['4'] = group_features_list[4]
                group_features[query]['5'] = group_features_list[5]


                group_features[query]['accuracy'] = int(label_class_group == majority_class)
                group_features[query]['error'] = math.fabs(int_label - majority)


        self.write_group_csv_file(output_dir, 'majority.csv', group_features, labels)

    def generate_examples_by_group(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
      #  stance_shrinking = {1: 1, 2: 1, 3: 3, 4: 5, 5: 5}
        config.group_size = 3
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                query = row['short query']
                examples, label, num_ir, rel = self.get_examples(config,  row, short_dir)
                if not examples:
                    continue
                labels[query] = label
                group_features_list = {x:[] for x in examples[0].keys()}
                group_features[query] = {}
                for example in examples:
                    for k,v in example.items():
                        group_features_list[k].append(v)
#                        if k != Features.STANCE_SCORE:
#                            group_features_list[k].append(v)
#                        else:
#                            group_features_list[k].append(v)
       #                     group_features_list[k].append(stance_shrinking[v])
                for k, vals in group_features_list.items():
                    if k == Features.STANCE_SCORE:
                        continue
                    group_features[query][k.value + '_mean'] = np.mean(vals)
                    group_features[query][k.value + '_var'] = np.var(vals)
                group_features[query]['rel'] = rel
                group_features[query]['num_ir'] = num_ir
        self.write_group_csv_file(output_dir, 'group_features.csv', group_features, labels)

    def gen_features_for_examples(self, query, examples, group_features, rel, suffix, features):
        empty_dict = lambda __=None: {x: [] for x in features if x != Features.STANCE_SCORE}
        group_features_list = {x: empty_dict() for x in range(1,6)}

        for example in examples:
            stance = example[Features.STANCE_SCORE]
            if not stance or int(stance) < 0:
                continue
            for k, v in example.items():
                if k != Features.STANCE_SCORE:
                    group_features_list[stance][k].append(v)
        for stance in range(1,6):
            for k, vals in group_features_list[stance].items():
                group_features[query][k.value + str(stance) + '_mean_' + suffix ] = 0 if not vals else np.mean(vals)
        group_features[query]['sample_size_' + suffix] = rel

    def generate_examples_by_group_paper_type(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        config.group_size = 3

        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                long_query = row['long query']
                backup_label = row['label'].strip()
                print(long_query)
                short_query = row['short query']
                rev_file = short_dir + long_query + "\\" + short_query + "_rev"
                clinical_file = short_dir + long_query + "\\" + short_query +"_clinical"
                rest_file = short_dir + long_query + "\\" + short_query+ "_rest"
                date = row['date']
                coch_pubmed_url = None
                review_year = date.split('.')[2].strip()
                label_r, rev_examples, rev_num_ir, rev_sample_size = self.generate_examples_per_query([rev_file], review_year, config, coch_pubmed_url)
                label_c, clinical_examples, clinical__num_ir,  clinical_sample_size = self.generate_examples_per_query([clinical_file], review_year, config, coch_pubmed_url)
                label_rest, rest_examples,  rest_num_ir, rest_sample_size = self.generate_examples_per_query([rest_file], review_year, config, coch_pubmed_url)

                if not rev_examples and not clinical_examples and not rest_examples:
                    print ('NO EXAMLES FOR ' + long_query)
                    continue
                if label_c == -1 and label_r == -1 and label_rest == -1:
                    labels[long_query] = backup_label
                elif label_r > -1:
                    labels[long_query] = label_r
                elif label_c > -1:
                    labels[long_query] = label_c
                else:
                    labels[long_query] = label_rest

                group_features[long_query] = {}
                if rev_examples:
                    features = rev_examples[0].keys()
                elif clinical_examples:
                    features = clinical_examples[0].keys()
                else:
                    features = rest_examples[0].keys()
                self.gen_features_for_examples(long_query, rev_examples, group_features, rev_sample_size, 'rev', features)
                self.gen_features_for_examples(long_query, clinical_examples, group_features, clinical_sample_size, 'clinical', features)
                self.gen_features_for_examples(long_query, rest_examples, group_features, rest_sample_size, 'rest', features)
        self.write_group_csv_file(output_dir, 'group_features_by_paper_type.csv', group_features, labels)

    def generate_examples_by_group_and_stance_and_majority(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        # stance_shrinking = {1: 1, 2: 1, 3: 3, 4: 5, 5: 5}
        config.group_size = 3
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, rel = self.get_examples(config, row, short_dir)
                if not examples:
                    continue
                query = row['long query']
                labels[query] = label
                empty_dict = lambda __=None: {x: [] for x in examples[0].keys() if x != Features.STANCE_SCORE}
                group_features_list = {1: empty_dict(),
                                       2: empty_dict(),
                                       3: empty_dict(),
                                       4: empty_dict(),
                                       5: empty_dict(),
                                       'all': empty_dict()}

                group_features[query] = {'stance_1': 1,
                                         'stance_2': 2,
                                         'stance_3': 3,
                                         'stance_4': 4,
                                         'stance_5': 5}
                stance_num = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                for example in examples:
                    # stance = stance_shrinking[example[Features.STANCE_SCORE]]
                    stance = example[Features.STANCE_SCORE]
                    stance_num[stance] += 1
                    for k, v in example.items():
                        if k != Features.STANCE_SCORE:
                            group_features_list[stance][k].append(v)
                            group_features_list['all'][k].append(v)

                for stance in range(1, 6):
                    group_features[query][str(stance)+'_num'] = stance_num[stance]
                    for k, vals in group_features_list[stance].items():
                        group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)

                #               for k, vals in group_features_list['all'].items():
                #                       group_features[query][k.value + 'all_mean'] = 0 if not vals else np.mean(vals)
                #                       group_features[query][k.value + 'all_std'] = 0 if not vals else np.std(vals)
                sorted_stance = sorted(stance_num.items(), key=lambda kv: kv[1], reverse=True)
                majority = sorted_stance[0][0]
                group_features[query]['rel'] = rel
                group_features[query]['majority'] = majority

        self.write_group_csv_file(output_dir, 'group_features_by_stance_and_num.csv', group_features, labels)

    def generate_examples_by_group_and_class(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        stance_shrinking = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
        config.group_size = 3
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, rel = self.get_examples(config, row, short_dir)
                if not examples:
                    continue
                query = row['long query']
                labels[query] = stance_shrinking[label]
                empty_dict = lambda __=None: {x: [] for x in examples[0].keys() if x != Features.STANCE_SCORE}
                group_features_list = {1: empty_dict(),
                                       2: empty_dict(),
                                       3: empty_dict()
                                     }


                for example in examples:
                    stance = stance_shrinking[example[Features.STANCE_SCORE]]
                    for k, v in example.items():
                        if k != Features.STANCE_SCORE:
                            group_features_list[stance][k].append(v)
                for stance in range(1, 6):
                    for k, vals in group_features_list[stance].items():
                        group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)
                #               for k, vals in group_features_list['all'].items():
                #                       group_features[query][k.value + 'all_mean'] = 0 if not vals else np.mean(vals)
                #                       group_features[query][k.value + 'all_std'] = 0 if not vals else np.std(vals)
                group_features[query]['rel'] = rel
        output_filename = 'group_features_by_stance_citation_range_' + str(config.review_end_range) + '.csv'
        self.write_group_csv_file(output_dir, output_filename, group_features, labels)

    def generate_examples_by_group_and_stance_only_rel(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        config.group_size = 3
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, rel = self.get_examples(config,  row, short_dir,  exclude = [])
                if not examples:
                    continue
                query = row['long query']
                labels[query] = label

                empty_dict = lambda __=None:  {x: [] for x in examples[0].keys()}
                group_features_list = {1: empty_dict(),
                                       2: empty_dict(),
                                       3: empty_dict(),
                                       4: empty_dict(),
                                       5: empty_dict()}

                group_features[query] = {}
                for example in examples:
                    stance = example[Features.STANCE_SCORE]
                    for k, v in example.items():
                        if k != Features.STANCE_SCORE:
                            group_features_list[stance][k].append(v)
                for stance in range(1,6):
                    for k, vals in group_features_list[stance].items():
                        group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)
                group_features[query]['rel'] = rel
        output_filename = 'rel_only_group_features_by_stance_citation_range_' + str(config.review_end_range) + '.csv'
        self.write_group_csv_file(output_dir,output_filename, group_features, labels)

    def generate_examples_by_group_and_stance_old(self, output_dir, queries, long_dir, short_dir, config, shrink=False):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        stance_shrinking = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
        config.group_size = 3
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, rel = self.get_examples(config, row, short_dir, exclude=[])
                if not examples:
                    continue
                query = row['long query']
                if shrink:
                    labels[query] = stance_shrinking[int(label)]
                else:
                    labels[query] = label

                empty_dict = lambda __=None: {x: [] for x in examples[0].keys() if x != Features.STANCE_SCORE}
                group_features_list = {1: empty_dict(),
                                       2: empty_dict(),
                                       3: empty_dict(),
                                       4: empty_dict(),
                                       5: empty_dict(),
                                       'all': empty_dict()}

                group_features[query] = {'stance_1': 1,
                                         'stance_2': 2,
                                         'stance_3': 3,
                                         'stance_4': 4,
                                         'stance_5': 5}
                for example in examples:
                    # stance = stance_shrinking[example[Features.STANCE_SCORE]]
                    stance = example[Features.STANCE_SCORE]
                    for k, v in example.items():
                        if k != Features.STANCE_SCORE:
                            group_features_list[stance][k].append(v)
                            group_features_list['all'][k].append(v)
                for stance in range(1, 6):
                    for k, vals in group_features_list[stance].items():
                        group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)
                #             for k, vals in group_features_list['all'].items():
                #                     group_features[query][k.value + 'all_mean'] = 0 if not vals else np.mean(vals)
                #                     group_features[query][k.value + 'all_std'] = 0 if not vals else np.std(vals)
                group_features[query]['rel'] = rel
        if shrink:
            output_filename = 'rel_only_group_features_by_stance_citation_range_' + str(
                config.review_end_range) + '_shrink.csv'
        else:
            output_filename = 'rel_=only_group_features_by_stance_old_large.csv'
        self.write_group_csv_file(output_dir, output_filename, group_features, labels)

    def generate_examples_by_group_and_stance_orig(self, output_dir, queries, long_dir, short_dir, config, include_dist_features = True, label_shrink = False):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        config.group_size = 3
        config.review_end_range = 1
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, num_ir, rel = self.get_examples(config, row, short_dir, exclude=[])
                if not examples:
                    continue
                query = row['long query']
                stance_shrinking = {1: 1, 2: 1, 3: 3, 4: 4, 5: 5}
                if label_shrink:
                    label = stance_shrinking[int(label)]
                labels[query] = label

                empty_dict = lambda __=None: {x: [] for x in examples[0].keys()}
                group_features_list = {1: empty_dict(),
                                       2: empty_dict(),
                                       3: empty_dict(),
                                       4: empty_dict(),
                                       5: empty_dict(),
                                       'all': empty_dict()}


                count_votes = {x:0 for x in range(1,6)}
                for example in examples:
                    stance = example[Features.STANCE_SCORE]
                    count_votes[stance] += 1
                    for k, v in example.items():
                        group_features_list[stance][k].append(v)
                        group_features_list['all'][k].append(v)
                sum_votes = sum(count_votes.values())

                group_features[query] = {}
                if include_dist_features:
                    dist_features = {'sigma' + str(i): count_votes[i] / sum_votes for i in range(1, 6)}
                    dist_features['sigmaz'] = (dist_features['sigma5'] - dist_features['sigma1']) / 5
                    group_features[query].update(dist_features)

                for stance in range(1, 6):
                    for k, vals in group_features_list[stance].items():
                        group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)
                h_index_all_sum = 0
                h_index_all_acc = 0
                citations_all_sum = 0
                citations_all_acc = 0
                for stance in range(1, 6):
                    stance_hIndex = group_features[query][Features.H_INDEX.value + str(stance) + '_mean']
                    stance_citations = group_features[query][Features.CITATION_COUNT.value + str(stance) + '_mean']
                    h_index_all_sum += stance_hIndex
                    citations_all_sum += stance_citations
                    h_index_all_acc += stance * stance_hIndex
                    citations_all_acc += stance * stance_citations

                for k, vals in group_features_list['all'].items():
                    group_features[query][k.value + 'all_mean'] = 0 if not vals else np.mean(vals)

                group_features[query]['rel'] = rel
#                group_features[query]['num_ir'] = num_ir
                group_features[query]['h_index_stance_avg'] = h_index_all_acc /h_index_all_sum
                group_features[query]['citations_stance_avg'] = 1 if citations_all_sum == 0 else  citations_all_acc /citations_all_sum

        if label_shrink:
            output_filename = 'group_features_by_stance_shrink.csv'
        else:
            if include_dist_features:
                output_filename = 'dist_features_group_features_by_stance.csv'
            else:
                output_filename = 'group_features_by_stance.csv'
        self.write_group_csv_file(output_dir, output_filename, group_features, labels)


    def gen_query_features(self, group_features, query, label, examples,rel,labels,
                           include_dist_features=True, label_shrink=False, normalize = False):
        stance_shrinking = {1: 1, 2: 1, 3: 3, 4: 3, 5: 5}
        if label_shrink:
            label = stance_shrinking[int(label)]

        labels[query] = label

        empty_dict = lambda __=None: {x: [] for x in examples[0].keys()}
        group_features_list = {1: empty_dict(),
                               2: empty_dict(),
                               3: empty_dict(),
                               4: empty_dict(),
                               5: empty_dict(),
                               'all': empty_dict()}

        count_votes = {x: 0 for x in range(1, 6)}
        for example in examples:
            stance = example[Features.STANCE_SCORE]
            count_votes[stance] += 1
            for k, v in example.items():
                group_features_list[stance][k].append(v)
                group_features_list['all'][k].append(v)
        sum_votes = sum(count_votes.values())

        group_features[query] = {}
        if include_dist_features:
            dist_features = {'sigma' + str(i): count_votes[i] / sum_votes for i in range(1, 6)}
            dist_features['sigmaz'] = (dist_features['sigma5'] - dist_features['sigma1']) / 5
            group_features[query].update(dist_features)

        # stance features
        sums = {x: 0 for x in group_features_list[1].keys()}
        for stance in range(1, 6):
            group_features[query]['stance_' + str(stance) + '_votes'] = count_votes[stance]
            for k, vals in group_features_list[stance].items():
                mean_val = 0 if not vals else np.mean(vals)
                sums[k] += mean_val
                group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)
                group_features[query][k.value + str(stance) + '_sum'] = 0 if not vals else np.sum(vals)

        h_index_all_sum = 0
        h_index_all_acc = 0
        citations_all_sum = 0
        citations_all_acc = 0
        if normalize:
            for stance in range(1, 6):
                for k, vals in group_features_list[stance].items():
                    normed_val = sys.maxsize if sums[k] == 0 else group_features[query][
                                                                      k.value + str(stance) + '_mean'] / sums[k]
                    group_features[query][k.value + str(stance) + '_normed_mean'] = normed_val
        for stance in range(1, 6):
            stance_hIndex = group_features[query][Features.H_INDEX.value + str(stance) + '_mean']
            stance_citations = group_features[query][Features.CITATION_COUNT.value + str(stance) + '_mean']
            h_index_all_sum += stance_hIndex
            citations_all_sum += stance_citations
            h_index_all_acc += stance * stance_hIndex
            citations_all_acc += stance * stance_citations

        for k, vals in group_features_list['all'].items():
            group_features[query][k.value + 'all_mean'] = 0 if not vals else np.mean(vals)

        group_features[query]['rel'] = rel
        #                group_features[query]['num_ir'] = num_ir
        group_features[query]['h_index_stance_avg'] = h_index_all_acc / h_index_all_sum
        group_features[query][
            'citations_stance_avg'] = 1 if citations_all_sum == 0 else citations_all_acc / citations_all_sum


    def generate_examples_by_group_and_stance_with_dummy(self, output_dir, queries, long_dir, short_dir, config,
                                              include_dist_features=True, label_shrink=False, normalize = False):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        config.group_size = 3
        config.review_end_range = 1
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, num_ir, rel = self.get_examples(config, row, short_dir, exclude=[])

                if not examples:
                    continue
                query = row['long query']
                self.gen_query_features(group_features, query, label, examples,rel,labels,
                           include_dist_features, label_shrink, normalize)

        for i in [5]:
            for j in range(0,5):
                examples, label, num_ir, rel = self.gen_dummy_examples(i)
                query ='dummy'+str(i)+'_'+str(j)
                self.gen_query_features( group_features, query, label, examples, rel, labels,
                                        include_dist_features, label_shrink, normalize)

        rev = {1:5,2:5,3:5,3:4,5:1,5:2,5:3,5:4}
        for j in range(0, 5):
            for i,j in rev.items():
                examples, label, num_ir, rel = self.gen_dummy_examples(i,j)
                query ='dummy'+str(i)+'_vs_'+str(j)
                self.gen_query_features( group_features, query, label, examples, rel, labels,
                                        include_dist_features, label_shrink, normalize)


        if label_shrink:
            output_filename = 'group_features_by_stance_shrink.csv'
        else:
            if include_dist_features:
                output_filename = 'dist_features_group_features_by_stance.csv'
            else:
                output_filename = 'dummy_added_group_features_by_stance.csv'
        self.write_group_csv_file(output_dir, output_filename, group_features, labels)




    def generate_examples_by_group_and_stance(self, output_dir, queries, long_dir, short_dir, config,
                                              include_dist_features=True, label_shrink=False, normalize = False):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        config.group_size = 3
        config.review_end_range = 1
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label, num_ir, rel = self.get_examples(config, row, short_dir, exclude=[])
                if not examples:
                    continue
                query = row['long query']
                stance_shrinking = {1: 1, 2: 1, 3: 3, 4: 3, 5: 5}
                if label_shrink:
                    label = stance_shrinking[int(label)]
                labels[query] = label

                empty_dict = lambda __=None: {x: [] for x in examples[0].keys()}
                group_features_list = {1: empty_dict(),
                                       2: empty_dict(),
                                       3: empty_dict(),
                                       4: empty_dict(),
                                       5: empty_dict(),
                                       'all': empty_dict()}

                count_votes = {x: 0 for x in range(1, 6)}
                for example in examples:
                    stance = example[Features.STANCE_SCORE]
                    count_votes[stance] += 1
                    for k, v in example.items():
                        group_features_list[stance][k].append(v)
                        group_features_list['all'][k].append(v)
                sum_votes = sum(count_votes.values())

                group_features[query] = {}
                if include_dist_features:
                    dist_features = {'sigma' + str(i): count_votes[i] / sum_votes for i in range(1, 6)}
                    dist_features['sigmaz'] = (dist_features['sigma5'] - dist_features['sigma1']) / 5
                    group_features[query].update(dist_features)

                #stance features
                sums = { x:0 for x in group_features_list[1].keys()}
                for stance in range(1, 6):
                    group_features[query]['stance_' + str(stance) + '_votes'] =count_votes[stance]
                    for k, vals in group_features_list[stance].items():
                        mean_val = 0 if not vals else np.mean(vals)
                        sums[k] += mean_val
                        group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)
                        group_features[query][k.value + str(stance) + '_sum'] = 0 if not vals else np.sum(vals)


                h_index_all_sum = 0
                h_index_all_acc = 0
                citations_all_sum = 0
                citations_all_acc = 0
                if normalize:
                    for stance in range(1, 6):
                        for k, vals in group_features_list[stance].items():
                            normed_val = sys.maxsize if sums[k] == 0 else  group_features[query][k.value + str(stance) + '_mean'] / sums[k]
                            group_features[query][k.value + str(stance) + '_normed_mean'] = normed_val
                for stance in range(1, 6):
                    stance_hIndex = group_features[query][Features.H_INDEX.value + str(stance) + '_mean']
                    stance_citations = group_features[query][Features.CITATION_COUNT.value + str(stance) + '_mean']
                    h_index_all_sum += stance_hIndex
                    citations_all_sum += stance_citations
                    h_index_all_acc += stance * stance_hIndex
                    citations_all_acc += stance * stance_citations

                for k, vals in group_features_list['all'].items():
                    group_features[query][k.value + 'all_mean'] = 0 if not vals else np.mean(vals)

                group_features[query]['rel'] = rel
                #                group_features[query]['num_ir'] = num_ir
                group_features[query]['h_index_stance_avg'] = h_index_all_acc / h_index_all_sum
                group_features[query][
                    'citations_stance_avg'] = 1 if citations_all_sum == 0 else citations_all_acc / citations_all_sum

        if label_shrink:
            output_filename = 'group_features_by_stance_shrink.csv'
        else:
            if include_dist_features:
                output_filename = 'dist_features_group_features_by_stance.csv'
            else:
                output_filename = 'group_features_by_stance.csv'
        self.write_group_csv_file(output_dir, output_filename, group_features, labels)


    def setup_dir(self, output_dir, long_dir, short_dir, config):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.write_readme(output_dir, long_dir, short_dir, config)

    def get_examples(self, config, row, short_dir, exclude):
        label_dict = {'does not help':1,'inconclusive':2, 'helps':3}
        date = row['date']
        backup_label = row['label'].strip()
        coch_pubmed_url = None
        review_year = date.split('/')[2].strip()
        label = label_dict[row['label'].strip()]
        q_dir = short_dir + row['long query'].strip()
        files = []
        if not os.path.isdir(q_dir):
            return [], -1, 0, 0
        for f in os.listdir(q_dir):
            sp = f.split('.csv')[0].split('_')
            if len(sp) > 1:
                suffix = sp[1]
                if suffix in exclude:
                    continue
            files.append(q_dir + '\\' + f)

        examples, num_ir,  rel = self.generate_examples_per_query(files, review_year, config, coch_pubmed_url)
        if label == -1:
            print(' no label for ' + row['long query'] )
        return examples, label, num_ir, rel


    def gen_dummy_examples(self, label, add_reverse = False):
        examples = []
        num_files = random.randrange(1,20)
        rev_files = 0
        if add_reverse:
            rev_files = random.randrange(0, num_files)
        for i in range(0, num_files - rev_files):
            examples.append(self.generate_dummy_feaures(label))
        if rev_files> 0:
            for i in range(0, rev_files):
                examples.append(self.generate_dummy_feaures(label, True))
        rel = num_files
        num_ir_max = max(0,20-num_files)
        num_ir = random.randrange(0,num_ir_max)
        return examples, label,num_ir, rel



    def generate_features2(self, files, review_year, config):
        featured_papers = []
        papers = {}
        for file in files:
            examples_collected = 0
            with open(file.strip()+'_bestMatch.csv', encoding='utf-8', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    url = row['document_url']
                    if not url:
                        continue
                    score = row['numeric']
                    if not score:
                        continue
                    score = int(score)
                    if score < -1 or (not config.include_irrelevant and score <0):
                        continue
                    pmid = row['document_url'].split('/')[-1].split('\n')[0]
                    paper = self.paper_builder.build_paper(pmid)
                    if not paper:
                        continue
                    papers[pmid] = paper
                    featured_paper = self.single_paper_feature_generator(paper, score, review_year, config)
                    if featured_paper:
                        featured_papers.append(featured_paper)
                        examples_collected += 1
                    if examples_collected == config.examples_per_file:
                        break
        return featured_papers

    def write_pairs_csv_file2(self, output_dir, query, examples, label, fields):
        pairs = permutations(examples, 2)
        fieldnames = ['label']
        for i in range(0, 2):
            for field in fields:
                fieldnames.append(field + str(i + 1))
        with open(output_dir + query + '.csv', 'w', encoding='utf-8', newline='') as outputCsv:
            wr = csv.DictWriter(outputCsv, fieldnames=fieldnames)
            wr.writeheader()
            for pair in pairs:
                diff1 = math.fabs(pair[0].stance_score - label)
                diff2 = math.fabs(pair[1].stance_score - label)
                pref = 1 if diff1 < diff2 else 2
                row = {}
                row['label'] = pref
                for i in range(0, 2):
                    attr = pair[i].__dict__
                    for field in fields:
                        row[field + str(i + 1)] = attr[field]
                wr.writerow(row)



class FeaturesGenerationConfig:
    def __init__(self, include_irrelevant, examples_per_file, review_start_range, review_end_range, group_size,
                 cochrane_search_range, remove_stance, perm, rel ="rel"):
        self.include_irrelevant = include_irrelevant
        self.examples_per_file = examples_per_file
        self.review_start_range = review_start_range
        self.review_end_range = review_end_range
        self.group_size = group_size
        self.cochrane_search_range = cochrane_search_range
        self.remove_stance = remove_stance
        self.perm = perm
        self.rel = rel

def gen_all_groups(fg):

    # fg = FeaturesGenerator('../resources/fg_cache3.json', '../resources/scimagojr 2018.csv','../resources/fg_noindex.json')

    output_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\normed\\'
    queries = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\queries.csv'
    short_dir = 'C:\\research\\falseMedicalClaims\\examples\\short queries\\pubmed\\classified\\'




## GROUPS of 5
    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=5, group_size=5, cochrane_search_range=15, remove_stance=True,
                                      perm = True)
    fg.generate_examples(output_dir + 'group1\\', queries, '', short_dir, config)
    config.include_irrelevant = True
    fg.generate_examples(output_dir + 'group2\\', queries, '', short_dir, config)


## GROUPS OF 3
    config.include_irrelevant = False
    config.group_size = 3
    fg.generate_examples(output_dir + 'group3\\', queries, '', short_dir, config)
    config.include_irrelevant = True
    fg.generate_examples(output_dir + 'group4\\', queries, '', short_dir, config)

#PAIRS

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=5, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm=False)
    fg.generate_examples_by_pairs(output_dir + 'group5\\', queries, '', short_dir, config)

#GROUP
    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=5, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm = False)
    fg.generate_examples_by_group(output_dir + 'group6\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance(output_dir + 'group7\\', queries, '', short_dir, config)

def gen_Yael_old(fg):
    output_dir = 'C:\\research\\falseMedicalClaims\ECAI\\model input\\Yael\\'
    queries = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries_all.csv'
    #queries = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries1_2.csv'
    #queries = 'C:\\research\\falseMedicalClaims\\examples\\to_classify_20_YAEL\\to_classify_20\\queries.csv'
    #short_dir = 'C:\\research\\falseMedicalClaims\\examples\\short queries\\pubmed\\CAM\\classified\\'
    #short_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\Yael\\sample1_and_2\\'
    short_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\Yael\\all\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=1, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm = False)

    fg.gen_majority_vote(output_dir +  'by_group\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config)
    #fg.generate_examples_by_group_and_stance_and_majority(output_dir +  'by_group\\', queries, '', short_dir, config)
#    fg.generate_examples_by_group_paper_type(output_dir + 'by_group\\', queries, '', short_dir, config)


def gen_group(fg):
    output_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\normed\\'
    queries = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\queries.csv'
    short_dir = 'C:\\research\\falseMedicalClaims\\examples\\short queries\\pubmed\\classified\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=1, group_size=3, cochrane_search_range=20, remove_stance=False,
                                      perm = False)
    fg.gen_majority_vote(output_dir +  'by_group\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config)
   # fg.generate_examples_by_group_paper_type(output_dir + 'by_group\\', queries, '', short_dir, config)

def gen_Yael(fg):
    output_dir = 'C:\\research\\falseMedicalClaims\ECAI\\model input\\Yael\\'
    queries = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries_all.csv'
    #queries = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries1_2.csv'
    #queries = 'C:\\research\\falseMedicalClaims\\examples\\to_classify_20_YAEL\\to_classify_20\\queries.csv'
    #short_dir = 'C:\\research\\falseMedicalClaims\\examples\\short queries\\pubmed\\CAM\\classified\\'
    #short_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\Yael\\sample1_and_2\\'
    short_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\Yael\\all\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=1, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm = False)

    fg.gen_majority_vote(output_dir +  'by_group\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config)
    #fg.generate_examples_by_group_and_stance_and_majority(output_dir +  'by_group\\', queries, '', short_dir, config)
#    fg.generate_examples_by_group_paper_type(output_dir + 'by_group\\', queries, '', short_dir, config)

def gen_Yael_sigal_Irit(fg):
    output_dir = 'C:\\research\\falseMedicalClaims\ECAI\\model input\\Yael_sigal_Irit\\'
    queries = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries1_2.csv'
    short_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\Yael_sigal_Irit\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=None, review_start_range=15,
                                      review_end_range=1, group_size=None, cochrane_search_range=15, remove_stance=None,
                                      perm=None)
    #fg.gen_majority_vote(output_dir + 'by_group\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config)

def gen_sample_1_2_all(fg, short_dir, output_dir):

    queries = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries1_2.csv'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=None, review_start_range=15,
                                      review_end_range=1, group_size=None, cochrane_search_range=15, remove_stance=None,
                                      perm=None)
    fg.gen_majority_vote(output_dir + 'by_group\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config, False)


def gen_cls_sample_1_2(fg, cls, short):
    output_dir = 'C:\\research\\falseMedicalClaims\ECAI\\model input\\' + cls +'\\'
    queries = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries1_2.csv'
    short_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\'+ cls +'\\' + short+'\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=1, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm = False)

    fg.gen_majority_vote(output_dir + 'by_group\\reports\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance_old(output_dir + 'by_group\\', queries, '', short_dir, config)
    #fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config)
    #fg.generate_examples_by_group_and_stance_and_majority(output_dir +  'by_group\\', queries, '', short_dir, config)
#    fg.generate_examples_by_group_paper_type(output_dir + 'by_group\\', queries, '', short_dir, config)


def ijcai():
    paper_cache = PaperCache('../resources/fg_cache3.json')
    hIndex = HIndex('../resources/scimagojr 2018.csv')
    fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
    paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, '../resources/fg_noindex.csv')
    fg = FeaturesGenerator(paper_builder)

#official:
#    output_dir = 'C:\\research\\falseMedicalClaims\IJCAI\\model input\\all\\'
#    queries = 'C:\\research\\falseMedicalClaims\\IJCAI\\query files\\queries_ijcai_pos_added.csv'
#    short_dir = 'C:\\research\\falseMedicalClaims\\IJCAI\\merged_annotations\\all\\'

    output_dir = 'C:\\research\\falseMedicalClaims\IJCAI\\model input\\ns\\'
  #  queries = 'C:\\research\\falseMedicalClaims\\IJCAI\\query files\\queries_ijcai_pos_added.csv'
    queries = 'C:\\research\\falseMedicalClaims\\IJCAI\\query files\\queries_ijcai_pos_neg_added.csv'
    short_dir = 'C:\\research\\falseMedicalClaims\\IJCAI\\merged_annotations\\ns\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=1, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm = False)

    #fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config, include_dist_features = False, label_shrink = False,  normalize = True)
    fg.generate_examples_by_group_and_stance_with_dummy(output_dir + 'by_group\\', queries, '', short_dir, config, include_dist_features = False, label_shrink = False,  normalize = True)
    fg.gen_majority_vote(output_dir + 'by_group\\reports\\', queries, '', short_dir, config)
    fg.gen_dist_features(output_dir + 'by_group\\reports\\majority.csv', output_dir)

def ecai():
    paper_cache = PaperCache('../resources/fg_cache3.json')
    hIndex = HIndex('../resources/scimagojr 2018.csv')
    fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
    paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, '../resources/fg_noindex.csv')
    fg = FeaturesGenerator(paper_builder)
    output_dir = 'C:\\research\\falseMedicalClaims\IJCAI\\model input\\ecai_query\\'
    short_dir = 'C:\\research\\falseMedicalClaims\\IJCAI\\merged_annotations\\ecai_query\\'
    queries = 'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries1_2.csv'
    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=None, review_start_range=15,
                                  review_end_range=1, group_size=None, cochrane_search_range=15, remove_stance=None,
                                  perm=None)
    fg.gen_majority_vote(output_dir + 'by_group\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config, False)
    fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config, False)

def w_and_h_fg():
    paper_cache = PaperCache('../../../resources/fg_cache_wh.json')
    hIndex = HIndex('../../../resources/scimagojr 2018.csv')
    fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
    paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, '../../../resources/fg_noindex_wh.csv')
    fg = FeaturesGenerator(paper_builder)

    queries = 'C:\\research\\falseMedicalClaims\\White and Hassan\\truth_detailed_comma.csv'
    output_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\'
    short_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\merged_annotations\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=1, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm=False)

#    fg.generate_examples_by_group_and_stance(output_dir + 'by_group\\', queries, '', short_dir, config, include_dist_features = False, label_shrink = False,  normalize = True)
    fg.gen_majority_vote(output_dir + 'by_group\\reports\\', queries, '', short_dir, config)
    fg.gen_dist_features(output_dir + 'by_group\\reports\\majority.csv', output_dir)


def main():
    #ecai()
    w_and_h_fg()
    return


if __name__ == '__main__':
    main()