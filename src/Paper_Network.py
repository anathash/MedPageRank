import csv
import collections

import PaperBuilder


class P_N(object):
    def __init__(self, pmids, paper_builder: PaperBuilder):
        self.papers_dict = {}
        self.csv_papers = []
        self.csv_papers_dict = {}
        self.paper_builder = paper_builder
        self.read_paper_info(pmids)


    def rest_network(self):
        self.papers_dict = self.csv_papers_dict


    def read_paper_info(self, pmids):
        for pmid in pmids:
            paper = self.paper_builder.build_paper(pmid)
            self.papers_dict[paper.pmid] = paper
            self.csv_papers.append(paper.pmid)
            self.csv_papers_dict[paper.pmid] = paper

    def create_network(self, recursion_degree):
        for paper_pmid in self.csv_papers:
            print('BUILDING CITATION NETWORK FOR ' + paper_pmid)
            self.recursion_search_citations(paper_pmid, recursion_degree)
        self.paper_builder.paper_cache.save_cache()


    def recursion_search_citations(self, paper_pmid, k):
        """
        recursion function for search the papers that cited the original paper
        :param paper_pmid: the original paper pmid
        :param k: the number of recursion iterations
        :return: None (append all papers to self.papers_dict)
        """
        if k == 0: return
        print('recursion degree = ' + str(k))
        original_paper = self.papers_dict[paper_pmid]
        if original_paper == None or original_paper.pm_cited == None: return
        print('paper ' + original_paper.pmid + ' has ' + str(len(original_paper.pm_cited))+ ' citations')
        for new_paper_pmid in original_paper.pm_cited:
            if new_paper_pmid not in self.papers_dict:
                citing_paper = self.paper_builder.build_paper(new_paper_pmid)
                self.papers_dict[new_paper_pmid] = citing_paper
            else:
                citing_paper = self.papers_dict[new_paper_pmid]
            citing_paper.add_to_pm_cite(paper_pmid)

            self.recursion_search_citations(citing_paper.pmid, k - 1)

    def get_network_edges_weights(self):
        edgeWeights = collections.defaultdict(lambda: collections.Counter())
        papers_h_index = {}
        for pmid, paper in self.papers_dict.items():
            papers_h_index[pmid] = paper.h_index
            if paper.pm_cited==None:
                edgeWeights[pmid] = collections.Counter()
            else:
                for citing_paper in paper.pm_cited:
                    if citing_paper in self.papers_dict:
                        neighborPaper = self.papers_dict[citing_paper]
                        edgeWeights[pmid][neighborPaper.pmid] = self.papers_dict[pmid].h_index


        return edgeWeights, papers_h_index