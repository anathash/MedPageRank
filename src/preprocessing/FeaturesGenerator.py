import csv
import itertools
import math
import numpy
import os
from itertools import combinations, permutations

from metapub import PubMedFetcher

from preprocessing.HIndex import HIndex
from preprocessing.PaperBuilder import PaperBuilder
from preprocessing.PaperCache import PaperCache
from preprocessing.PaperFeatures import PaperFeatures


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
        review_range = config.review_end_range + config.review_start_range + 1
        features_collection_end_year = int(review_year) + config.review_end_range
        features_collection_start_year = int(review_year) - config.review_start_range
        years_hIndex_acc = [0] * review_range
        for pmid in paper.pm_cited:
            citing_paper = self.paper_builder.build_paper(pmid)
            if not citing_paper:
                continue
            citing_paper_year = int(citing_paper.year)
            assert (citing_paper_year >= features_collection_start_year)
            if citing_paper_year > features_collection_end_year:
                continue
            year_gap = features_collection_end_year - int(citing_paper.year)
            years_hIndex_acc[year_gap] += citing_paper.h_index
        return numpy.average(years_hIndex_acc, weights = range(review_range, 0, -1))

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
        features = PaperFeatures(paper.h_index, paper.stance_score)
        review_year = int(review_year)
        year_gap = int(review_year + config.review_end_range - int(paper.year))
        if not 0 <= year_gap <= config.cochrane_search_range:
            print(paper.pmid)
#        assert (0 <= year_gap <= config.cochrane_search_range)
        features.add_year_gap_feature(year_gap)
        citation_count = 0 if not paper.pm_cited else len(paper.pm_cited)
        features.add_citation_feature(citation_count)
        #hIndex_wavg, wavg = self.compute_moving_averages(paper, review_year, config)
        hIndex_wavg = self.compute_moving_averages(paper, review_year, config)
        features.add_citations_hIndex_weighted_feature(hIndex_wavg)
        #features.add_citations_wighted_average_feature(wavg)
        features.set_contradicted_by_later(self.is_contradicted_by_later(paper, papers))
        return features


    def generate_features(self, files, review_year, config):
        featured_papers = []
        papers = {}
        for file in files:
             with open(file.strip() + '_bestMatch.csv', encoding='utf-8', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    url = row['document_url']
                    if not url:
                        continue
                    score = row['numeric']
                    if not score:
                        continue
                    score = int(score)
                    if score < -1 or (not config.include_irrelevant and score < 0):
                        continue
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
                    if examples_collected == config.examples_per_file:
                        break
        return featured_papers


    def generate_examples_per_query(self, files, review_year, config):
        featured_papers = self.generate_features(files, review_year, config)
        if config.group_size > 2:
            return permutations(featured_papers, config.group_size)
        else:
            return featured_papers


    def write_csv_file(self, output_dir, query, group_size, examples, label, fields):
        fieldnames = ['label']
        for i in range(0, group_size):
            for field in fields:
                fieldnames.append(field + str(i+1))
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
                        attr = example[i].__dict__
                    else:
                        attr = example.__dict__
                    for field in fields:
                        row[field + str(i+1)] = attr[field]
                wr.writerow(row)

    def write_readme(self, output_dir, long_dir, short_dir, config):
        with open(output_dir + 'README.txt', 'w', encoding='utf-8', newline='') as readme:
            readme.write('examples_per_file = ' + str(config.examples_per_file) + '\n')
            readme.write('include_irrelevant = ' + str(config.include_irrelevant) + '\n')
            readme.write('long_dir = ' + long_dir + '\n')
            readme.write('short_dir = ' + short_dir + '\n')
            readme.write('review_start_range = ' + str(config.review_start_range) + '\n')
            readme.write('review_end_range = ' + str(config.review_end_range) + '\n')
            readme.write('group_size = ' + str(config.group_size) + '\n')

    def write_pairs_csv_file(self, output_dir, query, examples, fields, get_diff, get_attr):
        pairs = permutations(examples, 2)
        fieldnames = ['label']
        for i in range(0, 2):
            for field in fields:
                fieldnames.append(field + str(i + 1))
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
                row = {'label' : pref}
                for i in range(0, 2):
                    attr = get_attr(pair[i])#pair[i][0].__dict__
                    for field in fields:
                        row[field + str(i + 1)] = attr[field]
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
                examples, label = self.get_examples(config, long_dir, row, short_dir)
                self.write_csv_file(output_dir, row['short query'], 1, examples, label,
                                    fields = PaperFeatures.__annotations__.keys())


    def generate_examples_by_pairs(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        all_examples = []
        config.group_size = 2
        fields = list(PaperFeatures.__annotations__.keys())
        if config.remove_stance:
            fields.remove('stance_score')
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label = self.get_examples(config, long_dir, row, short_dir)
                all_examples.extend([(e, label) for e in examples])
                #output_dir, query, examples, fields, get_diff, get_attr
                get_diff = lambda x: math.fabs(x.stance_score - int(label))
                get_attr = lambda p: p.__dict__
                self.write_pairs_csv_file(output_dir, row['short query'],  examples, fields, get_diff, get_attr)
            get_diff = lambda x: math.fabs(x[0].stance_score - int(x[1]))
            get_attr = lambda p: p[0].__dict__
            self.write_pairs_csv_file(output_dir, row['short query'], all_examples, fields, get_diff, get_attr)

    def generate_examples(self, output_dir, queries, long_dir, short_dir, config):
        self.setup_dir(output_dir, long_dir, short_dir, config)
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                examples, label = self.get_examples(config, long_dir, row, short_dir)
                self.write_csv_file(output_dir, row['short query'], config.group_size, examples, label,
                                    fields = PaperFeatures.__annotations__.keys())

    def setup_dir(self, output_dir, long_dir, short_dir, config):
        assert (not os.path.exists(output_dir))
        os.makedirs(output_dir)
        self.write_readme(output_dir, long_dir, short_dir, config)

    def get_examples(self, config, long_dir, row, short_dir):
        label = row['label']
        date = row['date']
        review_year = date.split('.')[2].strip()
        files = []
        if short_dir:
            files.append(short_dir + row['short query'])
        if long_dir:
            files.append(long_dir + row['long query'])
        examples = self.generate_examples_per_query(files, review_year, config)
        return examples, label

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
                 cochrane_search_range, remove_stance):
        self.include_irrelevant = include_irrelevant
        self.examples_per_file = examples_per_file
        self.review_start_range = review_start_range
        self.review_end_range = review_end_range
        self.group_size = group_size
        self.cochrane_search_range = cochrane_search_range
        self.remove_stance = remove_stance


def main():
    include_irrelevant = [True, False]
    examples_per_file = [10,15,20]
    review_start_range =[10,15]
    review_end_range = [1,5,10]
    group_size = [1,3,5]
    config = FeaturesGenerationConfig(include_irrelevant=False,examples_per_file=20,review_start_range=15,
                                      review_end_range=1,group_size=5, cochrane_search_range=15, remove_stance=True)

    paper_cache = PaperCache('../resources/fg_cache3.json')
    hIndex = HIndex('../resources/scimagojr 2018.csv')
    fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
    paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, '../resources/fg_noindex.json')
    fg = FeaturesGenerator(paper_builder)
    #fg = FeaturesGenerator('../resources/fg_cache3.json', '../resources/scimagojr 2018.csv','../resources/fg_noindex.json')

    output_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\'
    queries = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\queries.csv'
    short_dir = 'C:\\research\\falseMedicalClaims\\examples\\short queries\\pubmed\\classified\\'

    #fg.generate_examples_by_single(output_dir+'group1\\',queries,'',short_dir, config)
    #fg.generate_examples_by_pairs(output_dir+'group2\\', queries, '', short_dir, config)
    fg.generate_examples(output_dir+'group4\\', queries, '', short_dir, config)


if __name__ == '__main__':
    main()