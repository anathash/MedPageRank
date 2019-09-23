import os
import shutil
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

    def setup_feature_generator(self):
        p1 = Paper('1', 'title1', 'journal1', '', ['12', '13', '14', '15'], 10, '1234', 1987)
        p2 = Paper('2', 'title1', 'journal1', '', ['12', '13', '14', '15'], 20, '1234', 1987)
        p3 = Paper('3', 'title1', 'journal1', '', ['12', '13', '14', '15'], 30, '1234', 1987)
        p4 = Paper('4', 'title1', 'journal1', '', ['12', '13', '14', '15'], 40, '1234', 1987)
        p5 = Paper('5', 'title1', 'journal1', '', ['12', '13', '14', '15'], 50, '1234', 1987)
        p6 = Paper('6', 'title1', 'journal1', '', ['12', '13', '14', '15'], 60, '1234', 1987)
        p7 = Paper('7', 'title1', 'journal1', '', ['12', '13', '14', '15'], 70, '1234', 1987)
        p8 = Paper('8', 'title1', 'journal1', '', ['12', '13', '14', '15'], 80, '1234', 1987)
        p9 = Paper('9', 'title1', 'journal1', '', ['12', '13', '14', '15'], 90, '1234', 1987)

        c2 = Paper('12', 'title1', 'journal1', '', ['22', '23', '24', '25'], 5, '1234', 1989)
        c3 = Paper('13', 'title1', 'journal1', '', ['22', '23', '24', '25'], 6, '1234', 1991)
        c4 = Paper('14', 'title1', 'journal1', '', ['22', '23', '24', '25'], 4, '1234', 1991)
        c5 = Paper('15', 'title1', 'journal1', '', ['22', '23', '24', '25'], 30, '1234', 1993)

        paper_builder_mock = MagicMock()
        paper_builder_mock.build_paper = mock_responses({'1': p1, '2': p2, '3': p3, '4': p4, '5': p5, '6': p6,
                                                         '7': p7, '8': p8, '9': p9, '12':c2, '13':c3, '14':c4, '15':c5})
        return FeaturesGenerator(paper_builder_mock)

    def compare_files(self, actual_dir, expected_dir):
        for actual_name in os.listdir(actual_dir):
            if not actual_name.endswith('.csv'):
                continue
            expect_name = actual_name.split('.')[0] + '_expected.csv'
            with open(expected_dir + expect_name, 'r') as expected, open(actual_dir + actual_name, 'r') as actual:
                expected_lines = expected.readlines()
                actual_lines = actual.readlines()
                self.assertTrue(len(actual_lines) == len(expected_lines))
                for line in actual_lines:
                    self.assertTrue(line in expected_lines)

    def setup_actual_dir(self, actual_dir):
        if os.path.exists(actual_dir):
            shutil.rmtree(actual_dir)

    def test_generate_examples_by_single_remove_stance_exclude_ir(self):
        actual_dir = '../resources/output/single_remove_stance_exclude_ir/actual/'
        self.setup_actual_dir(actual_dir)

        config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=2,
                                          review_end_range=1, group_size=5, cochrane_search_range=2,
                                          remove_stance=True)
        fg = self.setup_feature_generator()
        fg.generate_examples_by_single(actual_dir, '../resources/queries.csv', '', '../resources/examples/', config)
        expected_dir = '../resources/output/single_remove_stance_exclude_ir/expected/'
        self.compare_files(actual_dir,expected_dir)

    def test_generate_examples_by_single_include_stance_exclude_ir(self):
        actual_dir = '../resources/output/single_include_stance_exclude_ir/actual/'
        self.setup_actual_dir(actual_dir)

        config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=2,
                                          review_end_range=1, group_size=5, cochrane_search_range=2,
                                          remove_stance=False)
        fg = self.setup_feature_generator()
        fg.generate_examples_by_single(actual_dir, '../resources/queries.csv', '', '../resources/examples/', config)
        expected_dir = '../resources/output/single_include_stance_exclude_ir/expected/'
        self.compare_files(actual_dir,expected_dir)


    def test_generate_examples_by_pairs_include_stance_exclude_ir(self):
        actual_dir = '../resources/output/pairs_include_stance_exclude_ir/actual/'
        self.setup_actual_dir(actual_dir)

        config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=2,
                                          review_end_range=1, group_size=5, cochrane_search_range=2,
                                          remove_stance=False)
        fg = self.setup_feature_generator()
        fg.generate_examples_by_pairs(actual_dir, '../resources/queries.csv', '', '../resources/examples/', config)
        expected_dir = '../resources/output/pairs_include_stance_exclude_ir/expected/'
        self.compare_files(actual_dir,expected_dir)

    def test_generate_examples_by_pairs_exclude_stance_exclude_ir(self):
        actual_dir = '../resources/output/pairs_exclude_stance_exclude_ir/actual/'
        self.setup_actual_dir(actual_dir)

        config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=2,
                                          review_end_range=1, group_size=5, cochrane_search_range=2,
                                          remove_stance=True)
        fg = self.setup_feature_generator()
        fg.generate_examples_by_pairs(actual_dir, '../resources/queries.csv', '', '../resources/examples/', config)
        expected_dir = '../resources/output/pairs_exclude_stance_exclude_ir/expected/'
        self.compare_files(actual_dir,expected_dir)

    def test_generate_examples_by_group_exclude_ir(self):
        actual_dir = '../resources/output/group_exclude_ir/actual/'
        self.setup_actual_dir(actual_dir)

        config = FeaturesGenerationConfig(include_irrelevant=False, examples_per_file=20, review_start_range=2,
                                          review_end_range=1, group_size=3, cochrane_search_range=2,
                                          remove_stance=True)
        fg = self.setup_feature_generator()
        fg.generate_examples(actual_dir, '../resources/queries.csv', '', '../resources/examples/', config)
        expected_dir = '../resources/output/group_exclude_ir/expected/'
        self.compare_files(actual_dir, expected_dir)

    def test_generate_examples_by_group_include_ir(self):
        actual_dir = '../resources/output/group_include_ir/actual/'
        self.setup_actual_dir(actual_dir)

        config = FeaturesGenerationConfig(include_irrelevant=True, examples_per_file=20, review_start_range=2,
                                          review_end_range=1, group_size=3, cochrane_search_range=2,
                                          remove_stance=True)
        fg = self.setup_feature_generator()
        fg.generate_examples(actual_dir, '../resources/queries.csv', '', '../resources/examples/', config)
        expected_dir = '../resources/output/group_include_ir/expected/'
        self.compare_files(actual_dir, expected_dir)

if __name__ == '__main__':
    unittest.main()
