import sys
from time import sleep

from preprocessing import PaperCache
from preprocessing.Paper import Paper

H_INDEX_CSV = "scimagojr 2018.csv"
SHORTCUT_JOURNALS_CSV = "jlist.csv"

class PaperBuilder:
    def __init__(self, hIndex, article_cache: PaperCache, fetcher, no_index_filename):
        self.cache_counter = 0
        self.hIndex = hIndex
        self.paper_cache = article_cache
        self.fetch = fetcher
        self.no_index_filename = no_index_filename

    def get_h_index(self, issn, journal):
        if not self.hIndex or not issn:
            return 1
        #issn = '' if not issn else issn.replace('-', '')
        h_index = self.hIndex.get_H_index(issn)
        #if h_index == 0:
        #    with open( self.no_index_filename, 'a') as file:
        #        file.write(journal+',' + issn + ' \n')
        #h_index += 1
        return h_index

    def get_paper_from_cache(self, pmid):
        paper = self.paper_cache.get_paper(pmid)
        if paper:
            if not self.hIndex:
                return paper
            # retry to get hIndex
            if paper.h_index == 1:
                print('No HIndex for journal ' + str(paper.journal) + ' with ISSN ' + str(paper.issn))
                paper.h_index = self.get_h_index(paper.issn, paper.journal)
                if paper.h_index > 1:
                    self.paper_cache.add_paper(paper.pmid, paper)
        #hack fix because of the cache
            if paper.pm_cited == None:
                paper.pm_cited = []
        return paper

    def get_article_from_pubmed(self, pmid):
        if not pmid:
            print('ssss')
            return
        article = None
        while not article:
            try:
                article = self.fetch.article_by_pmid(pmid)
            except:
                print('error fetching  paper for pmid ' + pmid)
                sleep(30)
        return article

    def get_article_citations(self, pmid):
        # pm_cited - which papers cited current paper
        try:
            paper = self.fetch.related_pmids(pmid)
            if 'citedin' not in paper:
                return []
            return paper['citedin']
        except:
            sys.stderr.write('error retrieving related for pubmed id ' + pmid + '/n')
            return []

    def update_cache(self, paper):
        self.paper_cache.add_paper(paper.pmid, paper)
        self.cache_counter += 1
        if self.cache_counter == 20:
            self.paper_cache.save_cache()
            self.cache_counter = 0

    def build_paper(self, pmid):
        print('building paper with pmid ' + pmid)
        paper = self.get_paper_from_cache(pmid)
        if paper:
            return paper
        article = self.get_article_from_pubmed(pmid)
        if not article.year:
            return None
        pm_cited = self.get_article_citations(pmid)
        h_index = self.get_h_index(article.issn, article.journal)
        paper = Paper(pmid, article.title, article.journal, article.authors, pm_cited, h_index, article.issn, int(article.year))
        self.update_cache(paper)
        return paper

    def add_to_pm_cite(self, pmip):
        if pmip not in self.pm_cite:
            self.pm_cite.append(pmip)