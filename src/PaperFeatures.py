import datetime


class PaperFeatures(object):
    year_gap: int
    citation_count: int
    h_index: int
    citations_wavg: float
    page_rank: float
    label: int
    citations_hIndex_wavg:float

    def __init__(self, h_index, stance_score):
        self.h_index = h_index
        self.stance_score = stance_score

    def add_year_gap_feature(self, year_gap):
        self.year_gap = year_gap

    def add_citation_feature(self, citation_count):
        self.citation_count = citation_count

    def add_citations_wighted_average_feature(self, citations_mavg):
        self.citations_wavg = citations_mavg

    def add_citations_hIndex_weighted_feature(self, citations_hIndex_mavg):
        self.citations_hIndex_wavg = citations_hIndex_mavg

    def add_pager_rank_feature(self, page_rank):
        self.page_rank = page_rank
