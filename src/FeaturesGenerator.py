import csv
from collections import Counter
from itertools import permutations

from metapub import PubMedFetcher

from HIndex import HIndex
from PaperBuilder import PaperBuilder
from PaperCache import PaperCache
from PaperFeatures import PaperFeatures
from Paper_Network import P_N
import page_rank_alg as pr
from pagerank_orig import orig_powerIteration


class FeaturesGenerator:
    def __init__(self, cache_file_name, hIndex_filename, no_index_filename):
        paper_cache = PaperCache(cache_file_name)
        hIndex = HIndex(hIndex_filename)
        fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
        self.paper_builder = PaperBuilder(hIndex, paper_cache, fetcher, no_index_filename)

    def compute_moving_averages(self, paper, review_year, range):
        counter = {}
        years_hIndex_acc = {}
        for i in range (0,range +1):
            counter[i] = 0
            years_hIndex_acc[i] = 0
        for pmid in paper.pm_cited:
            citing_paper = self.paper_builder.build_paper(pmid)
            year_gap = int(review_year) - int(citing_paper.year)
            counter[year_gap] += 1 #CHECK
            years_hIndex_acc[year_gap] += paper.h_index
        avg_hIndex = {x: sum / counter[x] for x in years_hIndex_acc.keys()}#
#       for year, sum in years_hIndex_acc.items():
#           avg_hIndex[year] = sum / counter[year]
        hIndex_wavg = 0
        wavg = 0
        for year_gap, avg_hIndex_per_year in avg_hIndex.items():
            hIndex_wavg += (range - year_gap +1)*avg_hIndex[year_gap]
            wavg += (range - year_gap +1)*counter[year_gap]
        return hIndex_wavg, wavg



#h_index, year, page_rank, label, citations_mavg, total_citations
    #h_index, year, page_rank, label, citations_mavg, total_citations
    def single_paper_feature_generator(self, pmid, stance_score, review_year, review_range):
        paper = self.paper_builder.build_paper(pmid)
        features = PaperFeatures(paper.h_index, stance_score)
        year_gap = int(review_year - paper.year)
        features.add_year_gap_feature(year_gap)
        citation_count = 0 if not paper.pm_cited else len(paper.pm_cited)
        features.add_citation_feature(citation_count)
        hIndex_wavg, wavg = self.compute_moving_averages(paper, review_year, review_range)
        features.add_citations_hIndex_weighted_feature(hIndex_wavg)
        features.add_citations_wighted_average_feature(wavg)
        return features

    def generate_features(self, files, examples_per_file,
                          include_irrelevant, review_year, cochrane_label, review_range):
        featured_papers = []
        for file in files:
            examples_collected = 0
            with open(file, encoding='utf-8', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    url =  row['document_url']
                    if not url:
                        continue
                    score = row['numeric']
                    if not score:
                        continue
                    score = int(score)
                    if score < -1 or (not include_irrelevant and score <0):
                        continue
                    pmid = row['document_url'].split('/')[-1].split('\n')[0]
                    featured_paper = self.single_paper_feature_generator(pmid, score, review_year,
                                                                         cochrane_label, review_range)
                    featured_papers.append(featured_paper)
                    examples_collected += 1
                    if examples_collected == examples_per_file:
                        break
        return  featured_papers

    def generate_examples_per_query(self, files, examples_per_file, include_irrelevant, review_year,
                          cochrane_label, review_range, group_size):
        featured_papers = self.generate_features(files, examples_per_file, include_irrelevant, review_year,
                                                 cochrane_label, review_range)
        return permutations(featured_papers, group_size)

    def write_csv_file(self, output_dir, query, group_size, examples):
        fields = PaperFeatures.__dict__.keys()
        fieldnames = []
        for i in range(0,group_size):
            for field in fields:
                fieldnames.append(field + str(i+1))
        with open(output_dir + query, 'w', encoding='utf-8', newline='') as outputCsv:
            wr = csv.DictWriter(outputCsv, fieldnames=fieldnames)
            for example_set in examples:
                assert(len(example_set) == group_size)
                for i in range(0, group_size):
                    row = {}
                    attr = example_set[i].__dict__
                    for field in fields:
                        row[field + str(i+1)] = attr[field]
                    wr.writerow(row)

    def write_readme(self, output_dir, examples_per_file,
                     include_irrelevant, long_dir, short_dir, review_range, group_size):
        with open(output_dir + 'README.md', 'w', encoding='utf-8', newline='') as readme:
            readme.write('examples_per_file = ' + examples_per_file + '/n')
            readme.write('include_irrelevant = ' + include_irrelevant + '/n')
            readme.write('long_dir = ' + long_dir + '/n')
            readme.write('short_dir = ' + short_dir + '/n')
            readme.write('review_range = ' + review_range + '/n')
            readme.write('group_size = ' + group_size + '/n')

    def generate_examples(self,output_dir, examples_per_file, include_irrelevant, queries, long_dir,
                          short_dir, review_range, group_size):
        self.write_readme(output_dir, examples_per_file, include_irrelevant, long_dir, short_dir, review_range, group_size)
        with open(queries, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                label = row['label']
                date = label['date']
                review_year = date.split('.')[2].strip()
                files = []
                if short_dir:
                    files.append(short_dir + row['short query'])
                if long_dir:
                    files.append(long_dir + row['long query'])
                examples = self.generate_examples_per_query(files, examples_per_file, include_irrelevant,  review_year,
                                                            label, review_range, group_size)
                self.write_csv_file(output_dir, examples)


def main():
    fg = FeaturesGenerator()
    fg.generate_features()

if __name__ == '__main__':
    main()