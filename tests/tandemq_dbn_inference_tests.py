# tests/tandemq_dbn_inference_tests.py
# Description: Unittests for the TandemQ DBN inference module.

# Import libraries
import unittest
import yaml
from src_hypoexp.tandemq_dbn_inference import construct_dbn

class TestTandemQDBN(unittest.TestCase):
    """Test the structure of the constructed DBN."""

    def setUp(self):
        self.config_file = 'tests/test_configs/tandemq_queries_test.yaml'

    def test_edges(self):
        """Test the edges when only service rate is used."""
        experiment_number = 3
        bn_filename = 'tests/data/tandemq_discrete_time/discrete-time-2tbn-exp-1.csv'
        dbn_filename = 'tests/data/tandemq_discrete_time/discrete-time-dbn-exp-1.csv'
        with open(self.config_file, 'r', encoding='utf-8') as file:
            test_configs = yaml.safe_load(file)
        test_config = test_configs[f'experiment_{experiment_number}']

        dbn = construct_dbn(bn_filename, dbn_filename, test_config['dbn_edges'],
                            test_config['maximum_ql'], None)

        if 'mup_lfc_lsc' in test_config['dbn_edges']:
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lfc0')
            ]
            self.assertTrue('Mup0' in parent_vars)
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lsc0')
            ]
            self.assertTrue('Mup0' in parent_vars)
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lfct')
            ]
            self.assertTrue('Mupt' in parent_vars)
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lsct')
            ]
            self.assertTrue('Mupt' in parent_vars)
        elif 'lambdap_lfc_lsc' in test_config['dbn_edges']:
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lfc0')
            ]
            self.assertTrue('Lambdap0' in parent_vars)
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lsc0')
            ]
            self.assertTrue('Lambdap0' in parent_vars)
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lfct')
            ]
            self.assertTrue('Lambdapt' in parent_vars)
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lsct')
            ]
            self.assertTrue('Lambdapt' in parent_vars)
        elif 'lp0_lfct_lsct' in test_config['dbn_edges']:
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lfct')
            ]
            self.assertTrue('Lp0' in parent_vars)
            parent_vars = [
                dbn.variable(parent).name() for parent in dbn.parents('Lsct')
            ]
            self.assertTrue('Lp0' in parent_vars)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
