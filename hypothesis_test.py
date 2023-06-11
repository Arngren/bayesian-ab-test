#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

__author__ = "Morten Arngren"

class Hypothesis_AB_Test:
    """ class for pre-processing data and calculating hypothesis test statistics
    """

    def __init__(self):
        """
        """
        pass

    def calc_sample_size(self, p1: float, p2: float, Z_a: float=1.96, Z_b: float=0.842) -> int:
        """ calc. the sample size to be used in statistical testing

        Args:
            p1 (float): performance of variant A, eg. p1 = 0.35 (ctr)
            p2 (float): performance of variant B, eg. p2 = 0.12 (ctr)
            Z_a (float): confidence level (e.g., Z = 1.96 for 95% confidence)
            Z_b (float): confidence level (e.g., Z = 0.842 for 80% confidence)
        """
        # calc. sample size for statistical testing
        # n_samples = (Z_a+Z_b)**2 * (p1*(1-p1) + p2*(1-p2)) / X**2
        n_samples = int( (Z_a+Z_b)**2 * (p1*(1-p1) + p2*(1-p2)) / (p2-p1)**2 ) + 1
        return n_samples


    def chi2_test(self, n_clicks_a: int, n_impr_a: int, n_clicks_b: int, n_impr_b: int) -> List[float]:
        """ calc. chi-square test

        Args:
            n_clicks_a (int): number of clicks for variant A
            n_impr_a (int): number of impressions for variant A
            n_clicks_b (int): number of clicks for variant B
            n_impr_b (int): number of impressions for variant B
        """
        ct = np.array([[n_clicks_a+1, n_impr_a-n_clicks_a+1], [n_clicks_b+1, n_impr_b-n_clicks_b+1]])
        chi2, p, dof, ex = chi2_contingency(ct)
        return chi2, p, dof, ex


    def transform(self, df: pd.DataFrame) ->  pd.DataFrame:
        """ calc. accumulated staistics for all events

        Args:
            df (pd.DataFrame): dataframe with all observed impression / clicks / conversions
        """
        print(f'CHI2 TEST...')
        # calc. chi2 AA test
        print(f'- calc. chi2 A/A-test...')
        df['chi2_A1A2_ctr'] = df.progress_apply(lambda x: self.chi2_test(x['acc_clicks_a1'], x['acc_impr_a1'], x['acc_clicks_a2'], x['acc_impr_a2']), axis=1)
        df['pvalue_A1A2_ctr'] = df.chi2_A1A2_ctr.apply(lambda x: x[1])
        df['chi2_A1A2_ctr'] = df.chi2_A1A2_ctr.apply(lambda x: x[0])
        df['chi2_B1B2_ctr'] = df.progress_apply(lambda x: self.chi2_test(x['acc_clicks_b1'], x['acc_impr_b1'], x['acc_clicks_b2'], x['acc_impr_b2']), axis=1)
        df['pvalue_B1B2_ctr'] = df.chi2_B1B2_ctr.apply(lambda x: x[1])
        df['chi2_B1B2_ctr'] = df.chi2_B1B2_ctr.apply(lambda x: x[0])

        print(f'- calc. chi2 A/B-test...')
        df['test'] = df.progress_apply(lambda x: self.chi2_test(x['acc_clicks_a'], x['acc_impr_a'], x['acc_clicks_b'], x['acc_impr_b']), axis=1)
        df['chi2_ctr'] = df.test.apply(lambda x: x[0])
        df['pvalue_ctr'] = df.test.apply(lambda x: x[1])
        df = df.drop(columns=['test'])

        # for CpC metric
        df['chi2_A1A2_cpc'] = df.progress_apply(lambda x: self.chi2_test(x['acc_cost_a1'], x['acc_clicks_a1'], x['acc_cost_a2'], x['acc_clicks_a2']), axis=1)
        df['pvalue_A1A2_cpc'] = df.chi2_A1A2_cpc.apply(lambda x: x[1])
        df['chi2_A1A2_cpc'] = df.chi2_A1A2_cpc.apply(lambda x: x[0])
        df['chi2_B1B2_cpc'] = df.progress_apply(lambda x: self.chi2_test(x['acc_cost_b1'], x['acc_clicks_b1'], x['acc_cost_b2'], x['acc_clicks_b2']), axis=1)
        df['pvalue_B1B2_cpc'] = df.chi2_B1B2_cpc.apply(lambda x: x[1])
        df['chi2_B1B2_cpc'] = df.chi2_B1B2_cpc.apply(lambda x: x[0])

        print(f'- calc. chi2 A/B-test...')
        df['test'] = df.progress_apply(lambda x: self.chi2_test(x['acc_cost_a'], x['acc_clicks_a'], x['acc_cost_b'], x['acc_clicks_b']), axis=1)
        df['chi2_cpc'] = df.test.apply(lambda x: x[0])
        df['pvalue_cpc'] = df.test.apply(lambda x: x[1])
        df = df.drop(columns=['test'])

        return df
