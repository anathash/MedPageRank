import csv
import datetime
import itertools
import math
import random
import sys
from enum import Enum
import collections
import numpy as np
import os
from itertools import combinations, permutations

from metapub import PubMedFetcher

from preprocessing.HIndex import HIndex
from preprocessing.PaperBuilder import PaperBuilder
from preprocessing.PaperCache import PaperCache
from preprocessing.PaperFeatures import PaperFeatures

HINDEX_MAX = 1096
stance_shrinking = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
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

    def single_paper_feature_generator_old(self, paper, review_year, config, papers):
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

    def single_paper_mult_feature_generator(self, paper, review_year, config):
        # features = PaperFeatures(paper.h_index, paper.stance_score)
        if paper.h_index <= 1:
            print('NO HINDEX FOR  PAPER WITH PMID ' + paper.pmid + ' published in ' + paper.journal)
        h_index_normed = paper.h_index * 100 / HINDEX_MAX
        features = {Features.H_INDEX: paper.h_index * 100 / HINDEX_MAX, Features.STANCE_SCORE: paper.stance_score}
        review_year = int(review_year)
        year_gap = int(review_year + config.review_end_range - int(paper.year))
        review_range = config.review_end_range + config.review_start_range + 1
        current_score = review_range - year_gap
        #        if not 0 <= year_gap <= config.cochrane_search_range:
        #            print(paper.pmid)
        #        assert (0 <= year_gap <= config.cochrane_search_range)
        features[Features.CURRENT_SCORE] = current_score
        features[Features.RECENT_WEIGHTED_H_INDEX] = current_score * h_index_normed  # paper.h_index

        # citation_count = 0 if not paper.pm_cited else len(paper.pm_cited)
        citation_count = 0 if not paper.pm_cited else self.get_citation_count(config, review_year, paper)
        features[Features.RECENT_WEIGHTED_CITATION_COUNT] = (current_score * citation_count)
        features[Features.CITATION_COUNT] = citation_count
        # ewa = self.compute_moving_averages(paper, review_year, config)
        # features[Features.CITATIONS_HINDEX_WAVG] = ewa
        # citations_wavg = self.compute_moving_averages(paper, review_year, config)
        # features[Features.CITATIONS_WAVG].add_citations_hIndex_weighted_feature(hIndex_wavg)
        # features.add_citations_wighted_average_feature(wavg)
        # TODO - remove after Experiment
        # features[Features.CONTRADICTED_BY_LATER] = self.is_contradicted_by_later(paper, papers)
        return features


    def single_paper_posterior_feature_generator(self, paper, review_year, config):
        # features = PaperFeatures(paper.h_index, paper.stance_score)
        if paper.h_index <= 1:
            print('NO HINDEX FOR  PAPER WITH PMID ' + paper.pmid + ' published in ' + paper.journal)
        review_year = int(review_year)
        citation_count = 0 if not paper.pm_cited else self.get_citation_count(config, review_year, paper)

        year_gap = int(review_year + config.review_end_range - int(paper.year))
        review_range = config.review_end_range + config.review_start_range + 1
        current_score = review_range - year_gap
        features = {Features.H_INDEX: paper.h_index , Features.STANCE_SCORE: paper.stance_score,
                    Features.CURRENT_SCORE: current_score, Features.CITATION_COUNT: citation_count}
        return features

    def norm_features(self, featured_papers):
        if not featured_papers:
            return featured_papers
        features_names = [x for x in featured_papers[0].keys() if x!= Features.STANCE_SCORE]
        sum_vals = {}
        for f in features_names:
            sum_vals[f] = sum([x[f] for x in featured_papers])
        for p in featured_papers:
            for f in features_names:
                if sum_vals[f] == 0:
                    p[f] = 0
                else:
                    p[f] /= sum_vals[f]
        return featured_papers



    def generate_features(self, files, review_year, config, coch_pubmed_url, mode, norm):
        label = None
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
                    if url == coch_pubmed_url:
                        label = int(row['category'])
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
            if mode == "MULT":
                featured_paper = self.single_paper_mult_feature_generator(paper, review_year, config)
            else:
                featured_paper = self.single_paper_posterior_feature_generator(paper, review_year, config)
            if featured_paper:
                featured_papers.append(featured_paper)
                examples_collected += 1
    #
        if norm:
            return self.norm_features(featured_papers), num_ir, rel, label
        else:
            return  featured_papers, num_ir, rel, label


    def generate_examples_per_query(self, files, review_year, config, coch_pubmed_url, mode, norm):
        return self.generate_features(files, review_year, config, coch_pubmed_url, mode,norm)


    def open_group_csv_file(self, output_dir,name, fieldnames):

        with open(output_dir + name, 'w', encoding='utf-8', newline='') as outputCsv:
            wr = csv.DictWriter(outputCsv, fieldnames=fieldnames)
            wr.writeheader()

        with open(output_dir + 'weighted_posterior.csv', 'w', encoding='utf-8', newline='') as postCsv:
            wr = csv.DictWriter(postCsv, fieldnames=['query','label','W1','W2','W3'])
            wr.writeheader()

        with open(output_dir + 'reports/majority.csv', 'w', encoding='utf-8', newline='') as majCsv:
            wr = csv.DictWriter(majCsv, fieldnames=['query','label','majority_value','majority_class','1','2','3','accuracy','error'])
            wr.writeheader()

        with open(output_dir + 'labels_' + name , 'w', encoding='utf-8', newline='') as labelsCSv:
            wr = csv.DictWriter(labelsCSv, fieldnames=['query', 'value_label'])
            wr.writeheader()

    def write_output_row(self, output_dir,name, query, features, label, fieldnames, votes, weights):
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        majority_value = sorted_votes[0][0]

        with open(output_dir + name, 'a', encoding='utf-8', newline='') as outputCsv:
            wr = csv.DictWriter(outputCsv, fieldnames)
            row = {'query': query, 'label': label}
            for k, v in features.items():
                row[k] = str(v)
            wr.writerow(row)
        with open(output_dir + 'weighted_posterior.csv', 'a', encoding='utf-8', newline='') as postCsv:
            wr = csv.DictWriter(postCsv, fieldnames=['query','label','W1','W2','W3'])
            row = {'query': query, 'label': label,
                    'W1': weights[1],
                    'W2': weights[2],
                    'W3': weights[3]}
            wr.writerow(row)

        with open(output_dir + '/reports/majority.csv', 'a', encoding='utf-8', newline='') as majCsv:
            wr = csv.DictWriter(majCsv, fieldnames=['query','label','majority_value','majority_class','1','2','3','accuracy','error'])

            row = {'query': query, 'label': label,
                    '1': votes[1],
                    '2': votes[2],
                    '3': votes[3],
                    'majority_value': majority_value,
                    'majority_class': majority_value,
                    'error':abs(label - majority_value),
                   'accuracy': int(label == majority_value)}
            wr.writerow(row)

        with open(output_dir + 'labels_' + name, 'a', encoding='utf-8', newline='') as labelsCSv:
            wr = csv.DictWriter(labelsCSv, fieldnames=['query', 'value_label'])
            row = {'query': query, 'value_label': label}
            wr.writerow(row)

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
        class_label = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                query = row['long query']
                label, votes = self.get_example_votes( row, short_dir, [])
                if not votes:
                    continue
                labels[query] = label

                group_features[query] = {}
                classes = [class_label[x] for x in votes]
                count = collections.Counter(classes)
                sorted_count = sorted(count.items(), key=lambda item: item[1], reverse=True)
                majority = sorted_count[0][0]
                print(query)
                int_label = int(label)
                majority_class = class_label[majority]
                group_features[query]['majority_value'] = majority
                group_features[query]['majority_class'] = majority_class

                group_features[query]['1'] = count[1]
                group_features[query]['2'] = count[2]
                group_features[query]['3'] = count[3]
                group_features[query]['4'] = count[4]
                group_features[query]['5'] = count[5]


                group_features[query]['accuracy'] = int(int_label == majority_class)
                group_features[query]['error'] = math.fabs(int_label - majority)


        self.write_group_csv_file(output_dir, 'majority.csv', group_features, labels)


    def group_features_by_stance(self, stance_shrink, examples):
        empty_dict = lambda __=None: {x: [] for x in examples[0].keys()}
        votes = {1:0,2:0,3:0}
        weights = {1:0,2:0,3:0}
        if stance_shrink:
            group_features_list = {1: empty_dict(),
                                   2: empty_dict(),
                                   3: empty_dict(),
                                   'all': empty_dict()}
        else:
            group_features_list = {1: empty_dict(),
                                   2: empty_dict(),
                                   3: empty_dict(),
                                   4: empty_dict(),
                                   5: empty_dict(),
                                   'all': empty_dict()}


        for example in examples:
            if stance_shrinking:
                stance = stance_shrinking[example[Features.STANCE_SCORE]]
            else:
                stance = example[Features.STANCE_SCORE]
            votes[stance_shrinking[example[Features.STANCE_SCORE]]] += 1
            for k, v in example.items():
                group_features_list[stance][k].append(v)
                group_features_list['all'][k].append(v)
                weights[stance] += v *1/(len (example.keys())-1)
        sum_weights = sum(weights.values())
        weights = {k:v/sum_weights for k,v in weights.items()}
        return votes, weights, group_features_list

    def get_wh_labels(self, queries):
        labels = {}
        label_dict = {'does not help': 1, 'inconclusive': 2, 'helps': 3}
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                label = row ['label'].strip()
                if label and not label.isdigit():
                    query = row['long query'].strip()
                    labels[query] = label_dict[label]
        return labels


    def gen_model_input(self, output_dir, output_filename, queries, long_dir, short_dir, config, mode,
                                              stance_shrink=False, exclude = [], norm = False, outlayers = []):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        config.group_size = 3
        config.review_end_range = 1

        fieldnames = None
        wh_labels = self.get_wh_labels(queries)
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                query = row['long query']
                if query in outlayers:
                    continue

                examples, label, num_ir, rel = self.get_examples(wh_labels,row, config, short_dir, exclude=exclude, mode = mode, norm = norm)
                if not examples:
                    continue


                votes, weights, group_features_list = self.group_features_by_stance(stance_shrink, examples)

                group_features[query] = {}

                #stance features
                sums = { x:0 for x in group_features_list[1].keys()}
                max_stance = 4
                for stance in range(1, max_stance):
                    for k, vals in group_features_list[stance].items():
                        if k == Features.STANCE_SCORE:
                            continue
                        mean_val = 0 if not vals else np.mean(vals)
              #          sums[k] += mean_val
                    #    group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)
                        group_features[query][k.value + str(stance) + '_sum'] = 0 if not vals else np.sum(vals)
                if not fieldnames:
                    fieldnames = ['query', 'label']
                    fieldnames.extend(list( group_features[query].keys()))
                    self.open_group_csv_file(output_dir, output_filename, fieldnames)

                self.write_output_row(output_dir, output_filename, query, group_features[query], label, fieldnames, votes, weights)


        #self.write_group_csv_file(output_dir, output_filename, group_features, labels)

    def get_feature_names(self, keys):
        feature_names = {}
        for k in keys:
            if k.endswith('_1'):
                feature_names.append(k[2:])
        return feature_names, int((len(keys) -2)/len(feature_names))

    def gen_posterior_beliefs(self, output_dir, feature_file, weights):
        feature_names = None
        num_stances = None
        rows = []
        with open(feature_file, encoding='utf-8', newline='') as feature_csv:
            reader = csv.DictReader(feature_csv)
            for row in reader:
                dict = {'query':row['query'],'label':row['label']}
                if not feature_names:
                    feature_names, num_stances = self.get_feature_names(row.keys())
                for i in range(1, num_stances+1):
                    for f in feature_names:
                        val = float(row[f+'_'+str(i)])
                        if weights:
                            dict['S'+str(i)] += val* weights[f]
                        else:
                            dict['S' + str(i)] += val * (1/len(feature_names))
                rows.append(dict)


        with open(output_dir + 'posterior_features.csv', 'w', encoding='utf-8', newline='') as postCSv:
            wr = csv.DictWriter(postCSv, fieldnames=['query', 'value_label'])
            wr.writeheader()
            for row in rows:
                wr.writerow(row)


    def setup_dir(self, output_dir, long_dir, short_dir, config):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.write_readme(output_dir, long_dir, short_dir, config)

    def get_examples(self, labels, row,  config, short_dir, exclude = [] , mode = "MULT", norm = False):
        query = row['long query'].strip()
        print(query)
        date = row['date']
        coch_pubmed_url = None
        if 'pubmed' in row:
            coch_pubmed_url =  row['pubmed'].strip()
        else:
            coch_pubmed_url = None
        review_year = date.split('/')[2].strip()
        row_label = row['label'].strip()


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

        examples, num_ir,  rel, coc_label = self.generate_examples_per_query(files, review_year, config, coch_pubmed_url, mode, norm)
        if coc_label:
            label = stance_shrinking[coc_label]
        elif row_label and row_label.isdigit():
            label = stance_shrinking[int(row_label)]
        else:
            label = labels[query]
        return examples, label, num_ir, rel


    def get_example_votes(self, row, short_dir, exclude):
        label_dict = {'does not help':1,'inconclusive':2, 'helps':3}
        label = label_dict[row['label'].strip()]
        q_dir = short_dir + row['long query'].strip()
        files = []
        if not os.path.isdir(q_dir):
            return -1, []
        for f in os.listdir(q_dir):
            sp = f.split('.csv')[0].split('_')
            if len(sp) > 1:
                suffix = sp[1]
                if suffix in exclude:
                    continue
            files.append(q_dir + '\\' + f)

        votes = []
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
                            continue
                    votes.append(score)
        return label, votes





def read_outlayer_file(outlayers_file):
    outlayers = []
    with open(outlayers_file, encoding='utf-8', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            outlayers.append(row['query'])
    return outlayers


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


def posterior_fg():
    outtlayers = read_outlayer_file(
        'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\outlayers.csv')
    paper_cache = PaperCache('../../../resources/fg_cache_wh.json')
    hIndex = HIndex('../../../resources/scimagojr 2018.csv')
    fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
    paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, '../../../resources/fg_noindex_wh.csv')
    fg = FeaturesGenerator(paper_builder)

   # queries = 'C:\\research\\falseMedicalClaims\\White and Hassan\\truth_detailed_comma.csv'
    queries = 'C:\\research\\falseMedicalClaims\\White and Hassan\\queries.csv'
    output_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\\posterior\\'
    short_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\merged_annotations_all\\'
    output_filename = 'posterior_not_normed.csv'
    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=1, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm=False)

    fg.gen_model_input(output_dir,output_filename,  queries, '', short_dir, config, 'POST', stance_shrink= True,  exclude = [], norm = False,  outlayers=outtlayers)
 #   fg.gen_majority_vote(output_dir + 'by_group\\reports\\', queries, '', short_dir, config)
#    fg.gen_dist_features(output_dir + 'by_group\\reports\\majority.csv', output_dir)

def mult_features_fg():
    paper_cache = PaperCache('../../../resources/fg_cache_wh.json')
    hIndex = HIndex('../../../resources/scimagojr 2018.csv')
    fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
    paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, '../../../resources/fg_noindex_wh.csv')
    fg = FeaturesGenerator(paper_builder)

    queries = 'C:\\research\\falseMedicalClaims\\White and Hassan\\truth_detailed_unicode_to_utf.csv'
    #queries = 'C:\\research\\falseMedicalClaims\\White and Hassan\\queries.csv'
    output_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\'
    short_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\merged_annotations\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=1, group_size=3, cochrane_search_range=15, remove_stance=False,
                                      perm=False)
    output_filename = 'mult_features_stance_shrink.csv'
    fg.gen_model_input(output_dir + 'mult_features\\',output_filename,  queries, '', short_dir, config,"MULT", True, norm = False)

#    fg.gen_dist_features(output_dir + 'mult_features\\reports\\majority.csv', output_dir)


def main():
    #ecai()
    posterior_fg()
    return


if __name__ == '__main__':
    main()