import csv
import datetime
import itertools
import math
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

    def single_paper_feature_generator(self, paper, review_year, config, papers):
        #features = PaperFeatures(paper.h_index, paper.stance_score)
        if  paper.h_index <= 1:
            print ('NO HINDEX FOR  PAPER WITH PMID ' + paper.pmid + ' published in ' + paper.journal )
        features = {Features.H_INDEX: paper.h_index *100/HINDEX_MAX, Features.STANCE_SCORE: paper.stance_score}
        review_year = int(review_year)
        year_gap = int(review_year + config.review_end_range - int(paper.year))
        review_range = config.review_end_range + config.review_start_range + 1
        current_score = review_range - year_gap
#        if not 0 <= year_gap <= config.cochrane_search_range:
#            print(paper.pmid)
#        assert (0 <= year_gap <= config.cochrane_search_range)
        features[Features.CURRENT_SCORE] = current_score
        features[Features.RECENT_WEIGHTED_CITATION_COUNT] = current_score*paper.h_index


        citation_count = 0 if not paper.pm_cited else len(paper.pm_cited)
        features[Features.RECENT_WEIGHTED_H_INDEX] = (current_score * citation_count)
        features[Features.CITATION_COUNT] = citation_count
        #hIndex_wavg, wavg = self.compute_moving_averages(paper, review_year, config)
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
                    score = row['numeric']
                    if score:
                        score = int(score)
                        if score == -1:
                            num_ir += 1
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
  #          if examples_collected == config.examples_per_file:
  #              break
        if config.rel == "rel":
            return featured_papers, rel
        else:
            return featured_papers, num_ir

    def generate_examples_per_query(self, files, review_year, config):
        featured_papers, rel = self.generate_features(files, review_year, config)
        if config.perm:
            return permutations(featured_papers, config.group_size), rel
        else:
            return featured_papers, rel

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
                query = row['short query']
                examples, label, rel = self.get_examples(config,  row, short_dir)
                labels[query] = label
                group_features_list = {1: 0, 3: 0, 5: 0}
                group_features[query] = {}
                for example in examples:
                    for k, v in example.items():
                        if k == Features.STANCE_SCORE:
                            stance = stance_shrinking[v]
                            group_features_list[stance] += 1

                sorted_stance = sorted(group_features_list.items(), key=lambda kv: kv[1], reverse=True)
                majority = sorted_stance[0][0]
                group_features[query]['majority'] = majority
                group_features[query]['accuracy'] = int(int(label) == majority)
                group_features[query]['1'] = group_features_list[1]
                group_features[query]['3'] = group_features_list[3]
                group_features[query]['5'] = group_features_list[5]

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
                examples, label, rel = self.get_examples(config,  row, short_dir, ['rest'])
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
        self.write_group_csv_file(output_dir, 'group_features.csv', group_features, labels)

    def gen_features_for_examples(self, query, examples, group_features, rel, suffix):
        empty_dict = lambda __=None: {x: [] for x in examples[0].keys() if x != Features.STANCE_SCORE}
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
                query = row['short query']
                rev_file = short_dir + query + "\\" + query + "_rev"
                clinical_file = short_dir + query + "\\" + query +"_clinical"
                rest_file = short_dir + query + "\\" + query+ "_rest"
                label = row['label']
                date = row['date']
                review_year = date.split('.')[2].strip()
                rev_examples, rev_sample_size = self.generate_examples_per_query([rev_file], review_year, config)
                clinical_examples, clinical_sample_size = self.generate_examples_per_query([clinical_file], review_year, config)
                rest_examples, rest_sample_size = self.generate_examples_per_query([rest_file], review_year, config)
                labels[query] = label
                group_features[query] = {}
                self.gen_features_for_examples(query, rev_examples, group_features, rev_sample_size, 'rev')
                self.gen_features_for_examples(query, clinical_examples, group_features, clinical_sample_size, 'clinical')
                self.gen_features_for_examples(query, rest_examples, group_features, rest_sample_size, 'rest')
        self.write_group_csv_file(output_dir, 'group_features_by_paper_type.csv', group_features, labels)

    def generate_examples_by_group_and_stance(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        group_features = {}
        labels = {}
        stance_shrinking = {1: 1, 2: 1, 3: 3, 4: 5, 5: 5}
        config.group_size = 3
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                query = row['short query']
                examples, label, rel = self.get_examples(config,  row, short_dir)
                labels[query] = label
                empty_dict = lambda __=None:  {x: [] for x in examples[0].keys() if x != Features.STANCE_SCORE}
                group_features_list = {1: empty_dict(), 3: empty_dict(), 5: empty_dict(), 'all': empty_dict()}
                group_features[query] = {'stance_1': 1,'stance_3': 3,'stance_5': 5}

                for example in examples:
                    stance = stance_shrinking[example[Features.STANCE_SCORE]]
                    for k, v in example.items():
                        if k != Features.STANCE_SCORE:
                            group_features_list[stance][k].append(v)
                            group_features_list['all'][k].append(v)
                for stance in [1, 3, 5]:
                    for k, vals in group_features_list[stance].items():
                        group_features[query][k.value + str(stance) + '_mean'] = 0 if not vals else np.mean(vals)
 #               for k, vals in group_features_list['all'].items():
 #                       group_features[query][k.value + 'all_mean'] = 0 if not vals else np.mean(vals)
 #                       group_features[query][k.value + 'all_std'] = 0 if not vals else np.std(vals)
                group_features[query]['rel'] = rel
        self.write_group_csv_file(output_dir, 'group_features_by_stance.csv', group_features, labels)


    def setup_dir(self, output_dir, long_dir, short_dir, config):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.write_readme(output_dir, long_dir, short_dir, config)

    def get_examples(self, config, row, short_dir, exclude = []):
        label = row['label']
        date = row['date']
        review_year = date.split('.')[2].strip()
        q_dir = short_dir + row['short query']
        files = []
        for (dirpath, dirnames, filenames) in os.walk(q_dir):
            for f in filenames:
                sp = f.split('.csv')[0].split('_')
                if len(sp) > 1:
                    suffix = sp[1]
                    if suffix in exclude:
                        continue
                files.append(q_dir + '\\' + f)

        examples, rel = self.generate_examples_per_query(files, review_year, config)
        return examples, label, rel

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

def gen_all(fg):

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

def gen_one(fg):
    output_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\CAM\\'
    queries = 'C:\\research\\falseMedicalClaims\\examples\\short queries\\pubmed\\CAM\\classified\\queries.csv'
    short_dir = 'C:\\research\\falseMedicalClaims\\examples\\short queries\\pubmed\\CAM\\classified\\'


    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=5, group_size=1, cochrane_search_range=25, remove_stance=False,
                                      perm = False)

    fg.generate_examples_by_group(output_dir + 'by_group\\', queries, '', short_dir, config)
    #fg.generate_examples_by_group_paper_type(output_dir + 'by_group\\', queries, '', short_dir, config)


def gen_group(fg):
    output_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\normed\\'
    queries = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\queries.csv'
    short_dir = 'C:\\research\\falseMedicalClaims\\examples\\short queries\\pubmed\\classified\\'

    config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=15,
                                      review_end_range=5, group_size=3, cochrane_search_range=20, remove_stance=False,
                                      perm = False)
    fg.generate_examples_by_group(output_dir + 'group7\\', queries, '', short_dir, config)
    fg.generate_examples_by_group_and_stance(output_dir + 'group7\\', queries, '', short_dir, config)


def main():
    include_irrelevant = [True, False]
    examples_per_file = [10,15,20]
    review_start_range =[10,15]
    review_end_range = [1,5,10]
    group_size = [1,3,5]

    paper_cache = PaperCache('../resources/fg_cache3.json')
    hIndex = HIndex('../resources/scimagojr 2018.csv')
    fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
    paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, '../resources/fg_noindex.json')
    fg = FeaturesGenerator(paper_builder)
    gen_one(fg)
    #gen_all(fg)
    #gen_group(fg)
    #fg.generate_examples(output_dir + 'group4\\', queries, '', short_dir, config)
#    fg.generate_examples_by_single(output_dir+'group1\\',queries,'',short_dir, config)


if __name__ == '__main__':
    main()