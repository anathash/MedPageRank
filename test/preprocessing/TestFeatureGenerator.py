import unittest
from unittest.mock import patch, MagicMock

from preprocessing.FeaturesGenerator import FeaturesGenerationConfig, FeaturesGenerator
from preprocessing.Paper import Paper
from preprocessing.PaperFeatures import PaperFeatures


def mock_responses(responses, default_response=None):
    return lambda input: responses[input] if input in responses else default_response


class TestFeatureGenerator(unittest.TestCase):

    def test_is_contradicted_by_later(self):
        p1 = Paper('1', 'title1', 'journal1', '', ['2', '3', '4', '5'], 1, '1234', 1987)
        p1.set_stance_score(5)
        p2 = Paper('2', 'title1', 'journal1', '', ['3', '13', '14', '15'], 5, '1234', 1990)
        p2.set_stance_score(1)
        p3 = Paper('3', 'title1', 'journal1', '', ['12', '13', '14', '15'], 6, '1234', 1992)
        p3.set_stance_score(3)
        papers = {'1':p1,'2': p2,'3': p3}
        fg = FeaturesGenerator(None)
        self.assertTrue(fg.is_contradicted_by_later(p1, papers))
        self.assertFalse(fg.is_contradicted_by_later(p2, papers))
        self.assertFalse(fg.is_contradicted_by_later(p3, papers))

    def test_compute_moving_averages(self):
        config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=2,
                                          review_end_range=1, group_size=5, cochrane_search_range=15,
                                          remove_stance=True)
        primary_paper = Paper('1', 'title1', 'journal1', '', ['2', '3', '4', '5'], 1, '1234', 1987)
        p2 = Paper('2', 'title1', 'journal1', '', ['12', '13', '14', '15'], 5, '1234', 1989)
        p3 = Paper('3', 'title1', 'journal1', '', ['12', '13', '14', '15'], 6, '1234', 1991)
        p4 = Paper('4', 'title1', 'journal1', '', ['12', '13', '14', '15'], 4, '1234', 1991)
        p5 = Paper('5', 'title1', 'journal1', '', ['12', '13', '14', '15'], 30, '1234', 1993)
        paper_builder_mock = MagicMock()
        paper_builder_mock.build_paper = mock_responses({'2': p2, '3': p3, '4': p4, '5': p5})
        fg = FeaturesGenerator(paper_builder_mock)
        avg = fg.compute_moving_averages(primary_paper, 1990, config)
        self.assertTrue(avg == 5.0)

    def test_single_paper_feature_generator(self):
        config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=2,
                                          review_end_range=1, group_size=5, cochrane_search_range=15,
                                          remove_stance=True)
        primary_paper = Paper('1', 'title1', 'journal1', '', ['2', '3', '4', '5'], 3, '1234', 1987)
        primary_paper.set_stance_score(5)
        c2 = Paper('2', 'title1', 'journal1', '', ['12', '13', '14', '15'], 5, '1234', 1989)
        c2.set_stance_score(1)
        c3 = Paper('3', 'title1', 'journal1', '', ['12', '13', '14', '15'], 6, '1234', 1991)
        c4 = Paper('4', 'title1', 'journal1', '', ['12', '13', '14', '15'], 4, '1234', 1991)
        c5 = Paper('5', 'title1', 'journal1', '', ['12', '13', '14', '15'], 30, '1234', 1993)
        paper_builder_mock = MagicMock()
        paper_builder_mock.build_paper = mock_responses({'2': c2, '3': c3, '4': c4, '5': c5})
        fg = FeaturesGenerator(paper_builder_mock)
        #paper, review_year, config, papers
        expected_paper = PaperFeatures(3,5)
        expected_paper.set_contradicted_by_later(int(True))
        expected_paper.add_citations_hIndex_weighted_feature(5.0)
        expected_paper.add_citation_feature(4)
        expected_paper.add_year_gap_feature(4)
        actual_paper = fg.single_paper_feature_generator(primary_paper, 1990, config, {'2':c2})
        self.assertEqual(expected_paper, actual_paper)




if __name__ == '__main__':
    unittest.main()
