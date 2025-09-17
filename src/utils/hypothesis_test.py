#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
from matplotlib.pylab import beta
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, norm

__author__ = "Morten Arngren"

class Hypothesis_AB_Test:
    """ class for pre-processing data and calculating hypothesis test statistics
    """

    def __init__(self):
        """
        """
        pass

    # def calc_sample_size_old(self, p1: float, p2: float, alpha: float=0.05, beta: float=0.2, side: str='single') -> int:
    #     """ calc. the sample size to be used in statistical testing

    #     Args:
    #         p1 (float): performance of variant A, eg. p1 = 0.35 (ctr)
    #         p2 (float): performance of variant B, eg. p2 = 0.12 (ctr)
    #         alpha (float, optional): significance level. Defaults to 0.05.
    #         beta (float, optional): (1-power) of the test. Defaults to 0.2, so power = 80%.
    #         side (str, optional): single or double sided test. Defaults to 'single'.
    #     """
    #     # calculate Z-values
    #     if side == 'single':
    #         Z_a = norm.ppf(1-alpha)
    #         Z_b = norm.ppf(1-beta)
    #     else:
    #         Z_a = norm.ppf(1-alpha/2)
    #         Z_b = norm.ppf(1-beta/2)

    #     # calc. sample size for statistical testing
    #     n_samples = int( (Z_a+Z_b)**2 * (p1*(1-p1) + p2*(1-p2)) / (p2-p1)**2 ) + 1
    #     return n_samples # , Z_a, Z_b


    def calc_sample_size(self, test_type: str, p_a: float, p_b: float, alpha: float=0.05, beta: float=0.2, alpha_side: str='single') -> tuple:
        """ calc. the sample size to be used in statistical testing assuming a chi-square test

        Args:
            test_type (str): type of test to perform, eg. 'Chi-square Test'
            p_a (float): performance of variant A, eg. p_a = 0.35 (ctr)
            p_b (float): performance of variant B, eg. p_b = 0.12 (ctr)
            alpha (float, optional): significance level. Defaults to 0.05.
            beta (float, optional): (1-power) of the test. Defaults to 0.2, so power = 80%.
            alpha_side (str, optional): single or double sided test. Defaults to 'single'.
        """
        # calculate Z-values
        if alpha_side == 'single':
            Z_a = norm.ppf(1-alpha)
        else:
            Z_a = norm.ppf(1-alpha/2)
        
        # Beta is always one-sided
        Z_b = norm.ppf(1-beta)

        # Effect size - use absolute value to handle negative differences
        effect_size = abs(p_b - p_a)

        n_samples = -1 # default value to indicate not calculated

        if test_type == 'Chi-square Test':
            # Handle edge cases
            if effect_size == 0:
                return -1, Z_a, Z_b
            
            # Pooled probability for equal groups
            pooled_prob = (p_a + p_b) / 2
            
            # Variance under null hypothesis (pooled)
            var_null = pooled_prob * (1 - pooled_prob)
            
            # Variance under alternative hypothesis
            var_alt = (p_a * (1 - p_a) + p_b * (1 - p_b)) / 2
            
            # Sample size calculation per group for equal allocation
            # Formula: n = (Z_α√(2*var_null) + Z_β√(2*var_alt))² / effect_size²
            numerator = (Z_a * np.sqrt(2 * var_null) + Z_b * np.sqrt(2 * var_alt))**2
            n_samples = numerator / (effect_size**2)
            
            # Handle infinite or invalid results
            if np.isinf(n_samples) or np.isnan(n_samples) or n_samples <= 0:
                return -1, Z_a, Z_b
            
            # Round up to ensure adequate power
            n_samples = int(np.ceil(n_samples))

        return n_samples, Z_a, Z_b


    def chi2_test(self, n_success_a: int, n_trails_a: int, n_success_b: int, n_trails_b: int) -> List[float]:
        """ calc. chi-square test

        Args:
            n_clicks_a (int): number of clicks for variant A
            n_impr_a (int): number of impressions for variant A
            n_clicks_b (int): number of clicks for variant B
            n_impr_b (int): number of impressions for variant B
        """
        ct = np.array([[n_success_a+1, n_trails_a-n_success_a+1], [n_success_b+1, n_trails_b-n_success_b+1]])
        # print(ct)
        chi2, p, dof, ex = chi2_contingency(ct)
        return chi2, p, dof, ex


    def t_test(self, cost_a: float, n_clicks_a: int, cost_b: float, n_clicks_b: int):
        """ calc. one-sided t-test usign scipy library

        Args:
            cost_a (float): cost for variant A
            n_clicks_a (int): number of clicks for variant A
            cost_b (float): cost for variant B
            n_clicks_b (int): number of clicks for variant B

        Returns:
            float: t-statistic
            float: p-value
        """
        t, p = ttest_ind(cost_a, n_clicks_a, cost_b, n_clicks_b)
        return t, p

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