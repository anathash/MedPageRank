class Paper(object):
    def __init__(self, pmid, title,journal, authors, pm_cited, h_index, issn):
        self.pm_cite = []
        self.title = title
        self.pmid = pmid
        self.journal = journal
        self.authors = authors
        self.pm_cited = pm_cited
        self.h_index = h_index
        self.issn = issn

    def add_to_pm_cite(self, pmip):
        if pmip not in self.pm_cite:
            self.pm_cite.append(pmip)