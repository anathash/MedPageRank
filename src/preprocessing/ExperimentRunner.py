import csv

from metapub import PubMedFetcher

from preprocessing.HIndex import HIndex
from preprocessing.PaperBuilder import PaperBuilder
from preprocessing.PaperCache import PaperCache
from preprocessing.Paper_Network import P_N
from preprocessing import page_rank_alg as pr
from preprocessing.pagerank_orig import orig_powerIteration


class ExperimentRunner:
    def __init__(self, cache_file_name, hIndex_filename, no_index_filename, label_filename, label_ratio_thresh, output_filename, workers, rsp):
        self.paper_info = {}
        self.workers = workers
        self.label_filename = label_filename
        if workers:
            self.label_data_with_workers(label_filename)
        else:
            self.label_data(label_filename)
        self.paper_cache = PaperCache(cache_file_name)
        self.hIndex = HIndex(hIndex_filename)
        self.label_ratio_thresh = label_ratio_thresh
        self.output_filename = output_filename
        self.fetcher = PubMedFetcher(email='anat.hashavit@gmail.com')
        paper_builder = PaperBuilder(self.hIndex, self.paper_cache, self.fetcher, no_index_filename)
        self.papers_network = P_N(list(self.paper_info.keys()), paper_builder)
        for pmid, p in self.papers_network.csv_papers_dict.items():
            self.paper_info[pmid]['Journal hIndex'] = p.h_index
            self.paper_info[pmid]['year'] = p.year
        self.rsp = rsp

    def label_data(self, turk_file):
        reader = csv.DictReader(open(turk_file))
        for row in reader:
            pmid = row['document_url'].split('/')[-1].split('\n')[0]
            self.paper_info[pmid] = {'document_url' : row['document_url'].strip(),
                                     'correct_label': row['correct_label'].strip()}

    def label_data_with_workers(self, turk_file):
        reader = csv.DictReader(open(turk_file))
        for row in reader:
            pmid = row['document_url'].split('/')[-1].split('\n')[0]
            self.paper_info[pmid] = row
            worker_labels = {'support': int(row['support']), 'not_support': int(row['reject']) + int(row['neutral']),
                             'irrelevant': int(row['irrelevant'])}
            sorted_by_value = list(sorted(worker_labels.items(), key=lambda kv: kv[1], reverse=True))
            top_label = sorted_by_value[0][0]
            top_score = int(sorted_by_value[0][1])
            runner_up_score = int(sorted_by_value[1][1])
            if runner_up_score == 0 or top_score / runner_up_score >= self.label_ratio_thresh:
                self.paper_info[pmid]['label'] = top_label
            else:
                self.paper_info[pmid]['label'] = 'undecided_label'


    def write_ranks_to_file(self, papers_res):
        with open(self.output_filename, 'w', newline='') as csvfile:
            fieldnames = list(papers_res[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for p in papers_res:
                writer.writerow(p)

    @staticmethod
    def write_avgs(csvfile, avg_ranks, fieldnames):
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in avg_ranks.items():
            row = v
            row['setting'] = k
            writer.writerow(row)


    def write_params_to_file(self, csvfile):
        csvfile.write('\n\n PARAMETERS \n')
        csvfile.write('rsp = ' + str(self.rsp) + '\n')
        csvfile.write('label_ratio_thresh = ' + str(self.label_ratio_thresh) + '\n')
        csvfile.write('input_file = ' + self.label_filename + '\n')

    def write_results_to_file(self, avg_ranks_worker, avg_ranks_me, papers_res):
        self.write_ranks_to_file(papers_res)
        with open(self.output_filename, 'a', newline='') as csvfile:
            self.write_my_label_results(avg_ranks_me, csvfile)
            if self.workers:
                self.write_worker_lables_results(avg_ranks_worker, csvfile)
            self.write_params_to_file(csvfile)

    def write_worker_lables_results(self, avg_ranks_worker, csvfile):
        worker_fieldnames = ['setting', 'support', 'not_support', 'irrelevant', 'undecided_label']
        csvfile.write('\n\nWORKERS AVERAGES\n')
        self.write_avgs(csvfile, avg_ranks_worker, worker_fieldnames)

    def write_my_label_results(self, avg_ranks_me, csvfile):
        my_field_names = ['setting', 'support', 'reject', 'neutral', 'preliminary', 'irrelevant', 'undecided_label']
        csvfile.write('\n\nCORRECT LABEL AVERAGES\n ')
        self.write_avgs(csvfile, avg_ranks_me, my_field_names)

    def compute_avg_ranks(self, key, ranks, label_filed_name, labels):
        counter = {x:0 for x in labels}
        accumulator = {x:0 for x in labels}
        for pmid, paper in self.paper_info.items():
            rank = ranks[key][pmid]
            paper[key + '_rank'] = rank
            label = paper[label_filed_name]
            accumulator[label] += rank
            counter[label] += 1
        return {x: 0 if not counter[x] else accumulator[x] / counter[x] for x in accumulator.keys()}


    def add_citation_info(self):
        for pmid, p in self.papers_network.csv_papers_dict.items():
            for i in range(0, 20):
                str_i = str(i)
                k = str_i + ' year cit'
                if str_i in p.citations:
                    self.paper_info[pmid][k] = p.citations[str_i][0]/p.citations[str_i][1]
                else:
                    self.paper_info[pmid][k] = 0
            self.paper_info[pmid]['year'] = p.year

    def report_exp_results(self, ranks):
        self.add_citation_info()
        avg_ranks_worker = None
        if self.workers:
            avg_ranks_worker = {key:self.compute_avg_ranks(key, ranks, 'label', ['support', 'not_support', 'irrelevant', 'undecided_label']) for key in ranks.keys()}
        avg_ranks_me = {key: self.compute_avg_ranks(key, ranks, 'correct_label', ['support', 'reject', 'neutral', 'preliminary', 'irrelevant', 'undecided_label']) for key in ranks.keys()}
        papers_res = list(self.paper_info.values())
        self.write_results_to_file(avg_ranks_worker, avg_ranks_me, papers_res)

    def print_citing_info(self):
        for pmid in self.papers_network.csv_papers_dict.keys():
            for paper in self.papers_network.csv_papers_dict.values():
                if paper.pm_cited and pmid in paper.pm_cited:
                    print (pmid + ' cited ' + paper.pmid)

    def run_experiment(self, recursion_degree, use_h_index):
        self.papers_network.rest_network()
        self.papers_network.create_network(recursion_degree)
        edge_weights, papers_h_index = self.papers_network.get_network_edges_weights()
        if use_h_index:
            ranks = pr.powerIteration(edge_weights, papers_h_index, rsp=self.rsp)
        else:
            ranks = orig_powerIteration(edge_weights, self.rsp)

        print(ranks)
        print('--')
        for paper in self.papers_network.csv_papers:
            rank = ranks.get(paper)
            print(paper + ' : ' + str(rank))
        return ranks

    def citation_counts(self):
        self.papers_network.rest_network()
        rank = {}
        for pmid, paper in self.papers_network.csv_papers_dict.items():
            cit_count = 0 if not paper.pm_cited else len(paper.pm_cited)
            rank[pmid] = cit_count
        return rank

    def h_index_count(self):
        self.papers_network.rest_network()
        rank = {}
        for pmid, paper in self.papers_network.csv_papers_dict.items():
            rank[pmid] = paper.h_index
        return rank

    def run_experiments(self, recursion_degs):
        results = {'CIT_COUNT': self.citation_counts(), 'H INDEX:': self.h_index_count()}
        for rec_deg in recursion_degs:
            for use_h_index in [True]:  #TODO: if adding flase fix citation info construction bug
                key = 'REC_DEG_' + str(rec_deg) + 'HIndex_' + str(use_h_index)
                results[key] = self.run_experiment(rec_deg, use_h_index)
        self. report_exp_results(results)


def main():
    exp_runner = ExperimentRunner('../resources/cinnamon/cinnamon3__.json',
                                  "../resources/scimagojr 2018.csv",
                                  'C:/research/falseMedicalClaims/turk/cinnamon/no_index.csv',
                                  'C:/research/falseMedicalClaims/turk/cinnamon/clinical_trials.csv',
                                  1.5,
                                  'C:/research/falseMedicalClaims/turk/cinnamon/clinical_trials_res.csv',
                                  False,
                                  0.31)
#    exp_runner.print_citing_info()
    exp_runner.run_experiments([2])

if __name__ == '__main__':
    main()