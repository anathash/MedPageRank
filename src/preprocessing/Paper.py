import datetime


class Paper(object):
    stance_score: int

    def __init__(self, pmid, title,journal, authors, pm_cited, h_index, issn, year):
        self.pm_cite = []
        self.title = title
        self.pmid = pmid
        self.journal = journal
        self.authors = authors
        self.pm_cited = pm_cited
        self.h_index = h_index
        self.issn = issn
        self.year = year
       # self.year_gap = datetime.datetime.now().year - int(year)
        self.citations = {}

    def add_to_pm_cite(self, pmip):
        if pmip not in self.pm_cite:
            self.pm_cite.append(pmip)

    def set_stance_score(self, stance_score):
        self.stance_score = stance_score


    def add_citation_info(self):

            for i in range(0, 20):
                str_i = str(i)
                k = str_i + ' year cit'
                if str_i in p.citations:
                    self.paper_info[self.pmid][k] = p.citations[str_i][0]/self.citations[str_i][1]
                else:
                    self.paper_info[self.pmid][k] = 0
            self.paper_info[self.pmid]['year'] = p.year
