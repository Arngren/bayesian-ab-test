#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List
import numpy as np
import pandas as pd
import multiprocessing as mp

import scipy
from scipy.stats import beta, gamma, kstest, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.special import betaln

__author__ = "Morten Arngren"

class Bayesian_AB_Test:
    """ class for pre-processing data and calculating Bayesian test statistics
    """

    def __init__(self):
        """ Inits class
        """
        # init
        self.rv = {}

    def set_rv(self, rv, name):
        self.rv.update({name: rv})

    # def B(self, alpha, beta):
    #     """ mapper function for the beta distribution
    #     """
    #     return scipy.special.beta(alpha, beta)

    def p_overlap(self, rv_a, rv_b, metric: str='ks', n_samples: int=10000) -> float:
        """ calc. the overlap between two probability distribution rv_a and rv_b

        Args:
            rv_a (scipy.stats): random variable function for variant A
            rv_b (scipy.stats): random variable function for variant B
            metric (str): metric to use for calc. the overlap
            n_samples (int): number of samples to use for relevant metrics

        Return:
            overlap (float): 
        """
        # Kolmogorov-Smirnov test
        # - operates on samples from the random variables
        # - ref: https://www.mit.edu/~6.s085/notes/lecture5.pdf
        if metric == 'ks':
            ks = kstest(rv_a.rvs(size=n_samples), rv_b.rvs(size=n_samples))
            return 1 - ks.statistic

        # Earth-Mover Distance
        # - operates on the cdf of the random variables
        if metric == 'ws':
            # built-in function - takes samples as input
            # ws = 1 - wasserstein_distance(rv_a.rvs(n_samples), rv_b.rvs(n_samples))

            # manual - to illustrate math behind
            x = np.linspace(0, 1, n_samples)
            ws = np.abs( rv_a.cdf(x) - rv_b.cdf(x) )
            ws = 1 - (np.sum(ws) / n_samples)
            return ws

        # Jensen-Shannon Distance
        # - operates on the pdf of the random variables
        if metric == 'jsd':
            x = np.linspace(0, 1, 101)
            pdf_a = [rv_a.pdf(_) for _ in x]
            pdf_b = [rv_b.pdf(_) for _ in x]
            jsd = jensenshannon( pdf_a, pdf_b )
            return jsd
    

    def calc_sample_size(self, perf_A: float, lift: float, threshold: float, metric: str, precision: str = 'Low') -> int:
        """ calc. the sample size to be used in statistical testing
        
        Args:
            perf_A (float): performance of variant A, eg. p_a = 0.35 (ctr)
            lift (float): expected lift in performance, eg. lift = 0.1 (10%)
            threshold (float): threshold for the probability of B > A
            metric (str): metric to use for calc. the sample size
            precision (str): precision of the sample size calculation

        Returns:
            impr_thr (int): number of impressions needed to reach the threshold
            impr_list (List): list of impressions
            PA (List): list of probabilities that B > A
        """

        def __calc_threshold(impr_list, perf_A, perf_B, n_samples=20000):
            """ Internal function to calc. the threshold

            Args:
                impr_list (List): list of impressions
                perf_A (float): performance of variant A
                perf_B (float): performance of variant B
                n_samples (int): number of samples

            Returns:
                impr_thr (int): number of impressions needed to reach the threshold
                PA (List): list of probabilities that B > A
            """
            PA =[]
            for impr in impr_list:
                # if metric in  ['CTR, CVR']:
                success_a = int(impr * perf_A)
                success_b = int(impr * perf_B)                    
                rv_a, rv_b = beta(success_a+1, impr-success_a+1), beta(success_b+1, impr-success_b+1)
                # if metric ==  ['CpC, CpA']:
                #     rv_a, rv_b = beta(success_a+1, impr-success_a+1), beta(success_b+1, impr-success_b+1)

                # calc. prob. of B > A
                P, _ = self.p_ab_loss( [rv_a, rv_b], best='max', n_samples=n_samples )
                PA += [ P[1] ] # extract P(B>A)

            # identify 0.95 in PA
            id_thr = np.argmax(np.array(PA) > threshold)
            impr_thr = impr_list[id_thr]
            return impr_thr, PA

        # precision
        if precision == 'Low':
            n_impr, n_samples = 1_00, 1_000
        if precision == 'Medium':
            n_impr, n_samples = 1_000, 20_000
        if precision == 'High':
            n_impr, n_samples = 1_000, 100_000

        # calc. CTR_B
        perf_B = perf_A * (1+lift/100)

        # Sweep through the number of impressions and identify the threshold
        impr_list = np.linspace(1, 1_000_000, 100, dtype=int)
        impr_thr, PA = __calc_threshold(impr_list, perf_A, perf_B, n_samples=10000)

        # Refine the threshold by zooming in on the threshold interval
        if impr_thr > 1: # success
            impr_list = np.linspace(1, impr_thr*1.1, n_impr, dtype=int)
            impr_thr, PA = __calc_threshold(impr_list, perf_A, perf_B, n_samples=n_samples)
            # round to nearest 100
            impr_thr = int(np.ceil(impr_thr/100)*100)
        else:
            impr_thr = -1 # to indidate that the threshold is not reached

        return impr_thr, impr_list, PA


    def p_ab_loss(self, rvs: List, best: str='max', thr: float=1, n_samples: int=10_000):
        """ Calc. probability that all variants are better than the rest and corresponding loss

        Args:
            rvs (List): list of scipy.stats objects
            best (str): best variant to be max or min
            thr (float): threshold
            n_samples (int): number of samples

        Returns:
            P_ab_thr (List): list of probabilities that each variant is better than the rest
            loss (List): list of losses for each variant
        """
        # Generate samples from all variants
        samples = np.array( [rv.rvs(size=n_samples) for rv in rvs] )

        # Calc. probability that a variant is better than the rest
        P_ab_thr, loss = [], []
        for id_ref in range(len(rvs)):
            # Identify the rest of the variants
            id_rest = [i for i in range(len(rvs)) if i != id_ref]
            
            # calc. Bayesian metrics for best being max. or min.
            if best == 'max':
                # extract most 'competitive' samples
                samples_best_of_rest = np.max(samples[id_rest], axis=0)
                # calc. probability ratio that ref is better than the rest
                P_ratio = samples[id_ref] / samples_best_of_rest
                P_ab_thr += [ (P_ratio>thr).sum() / n_samples ]
                # calc. loss from the most 'competitive' sample
                loss += [ np.mean( np.maximum(samples_best_of_rest - samples[id_ref], 0) ) ]
            if best == 'min':
                # extract most 'competitive' samples
                samples_best_of_rest = np.min(samples[id_rest], axis=0)
                # calc. probability ratio that ref is better than the rest
                P_ratio = samples[id_ref] / samples_best_of_rest
                P_ab_thr += [ (P_ratio<thr).sum() / n_samples ]
                # calc. loss from the most 'competitive' sample
                loss += [ np.mean( np.maximum(samples[id_ref] - samples_best_of_rest, 0) ) ]

        return P_ab_thr, loss

    def hpdr(self, rv, thr: float=0.95):
        """ calc. the highest posterior density region (HPDR) for a given threshold
        
        Args:
            rv (scipy.stats): random variable function
            thr (float): threshold for the HPDR

        Returns:
            hpdr (List): highest posterior density region
        """
        # import scipy.stats as stats
        lower_bound = rv.ppf((1 - thr) / 2)
        upper_bound = rv.ppf(1 - (1 - thr) / 2)
        return (lower_bound, upper_bound)
    

    def power_analysis(self, ctr, lift, n_samples=1000):
        """ calc. power analysis for different sample sizes
        """
        # define
        impr_list = np.arange(0, 50_000, 100)

        for impr in impr_list:
            # clicks
            clicks_a = impr * ctr
            clicks_b = impr * ctr * (1+lift/100)

            # calc. prob. of B > A
            PA_p_ba = [ self.p_ba(beta(clicks_a+1, impr-clicks_a+1), beta(clicks_b+1, impr-clicks_b+1) ) ]

    # def p_ab_loss_api(self):
    #     rvs
    #     return self.p_ab_loss(rvs: List, best: str='max', thr: float=1, n_samples: int=10_000):

    def agg_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """ calc. aggregated statistics for all events, accumulated

        Args:
            df (DataFrame): dataframe with all observed impression / clicks / conversions
        """
        # pre-calc. parameters for CTR modelling using the Beta distributions
        df['alpha_a1'] = df.acc_clicks_a1 + 1
        df['beta_a1'] = df.acc_impr_a1 - df.acc_clicks_a1 + 1
        df['alpha_a2'] = df.acc_clicks_a2 + 1
        df['beta_a2'] = df.acc_impr_a2 - df.acc_clicks_a2 + 1
        df['alpha_a'] = df.acc_clicks_a + 1
        df['beta_a'] = df.acc_impr_a - df.acc_clicks_a + 1
        df['alpha_b1'] = df.acc_clicks_b1 + 1
        df['beta_b1'] = df.acc_impr_b1 - df.acc_clicks_b1 + 1
        df['alpha_b2'] = df.acc_clicks_b2 + 1
        df['beta_b2'] = df.acc_impr_b2 - df.acc_clicks_b2 + 1
        df['alpha_b'] = df.acc_clicks_b + 1
        df['beta_b'] = df.acc_impr_b - df.acc_clicks_b + 1

        # pre-calc. parameters for CpC modelling using the Gamma distributions
        df['a_a1'] = df.acc_cost_a1+1
        df['scale_a1'] = 1 / df.acc_clicks_a1
        df['a_a2'] = df.acc_cost_a2+1
        df['scale_a2'] = 1 / df.acc_clicks_a2
        df['a_a'] = df.acc_cost_a+1
        df['scale_a'] = 1 / df.acc_clicks_a
        df['a_b1'] = df.acc_cost_b1+1
        df['scale_b1'] = 1 / df.acc_clicks_b1
        df['a_b2'] = df.acc_cost_b2+1
        df['scale_b2'] = 1 / df.acc_clicks_b2
        df['a_b'] = df.acc_cost_b+1
        df['scale_b'] = 1 / df.acc_clicks_b

        return df

    def calc_performance(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """ calc. performance for all events

        Args:
            df (DataFrame): dataframe with all observed impression / clicks / conversions
            config (Dict): configuration parameters
        """
        n_samples = config['metrics']['n_samples']

        # A/A test - beta distribution
        if config['metrics']['ks']:
            print(f'- A/A - Beta - Kolmogorov-Smirnov...')
            df['P_A1A2_b_ks'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_a1'],x['beta_a1']), rv_b=beta(x['alpha_a2'],x['beta_a2']), metric='ks', n_samples=n_samples), axis=1)
            df['P_B1B2_b_ks'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_b1'],x['beta_b1']), rv_b=beta(x['alpha_b2'],x['beta_b2']), metric='ks', n_samples=n_samples), axis=1)
        if config['metrics']['ws']:
            print(f'- A/A - Beta - Wasserstein...')
            df['P_A1A2_b_ws'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_a1'],x['beta_a1']), rv_b=beta(x['alpha_a2'],x['beta_a2']), metric='ws', n_samples=n_samples), axis=1)
            df['P_B1B2_b_ws'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_b1'],x['beta_b1']), rv_b=beta(x['alpha_b2'],x['beta_b2']), metric='ws', n_samples=n_samples), axis=1)
        if config['metrics']['jsd']:
            print(f'- A/A - Beta - JSD...')
            df['P_A1A2_b_jsd'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_a1'],x['beta_a1']), rv_b=beta(x['alpha_a2'],x['beta_a2']), metric='jsd', n_samples=n_samples), axis=1)
            df['P_B1B2_b_jsd'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_b1'],x['beta_b1']), rv_b=beta(x['alpha_b2'],x['beta_b2']), metric='jsd', n_samples=n_samples), axis=1)

        # A/A test - gamma distribution
        if config['metrics']['ks']:
            print(f'- A/A - Gamma - Kolmogorov-Smirnov...')
            df['P_A1A2_g_ks'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x['a_a1'],scale=x['scale_a1']), rv_b=gamma(a=x['a_a2'],scale=x['scale_a2']), metric='ks', n_samples=n_samples), axis=1)
            df['P_B1B2_g_ks'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x['a_b1'],scale=x['scale_b1']), rv_b=gamma(a=x['a_b2'],scale=x['scale_b2']), metric='ks', n_samples=n_samples), axis=1)
        if config['metrics']['ws']:
            print(f'- A/A - Gamma - Wasserstein...')
            df['P_A1A2_g_ws'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x['a_a1'],scale=x['scale_a1']), rv_b=gamma(a=x['a_a2'],scale=x['scale_a2']), metric='ws', n_samples=n_samples), axis=1)
            df['P_B1B2_g_ws'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x['a_b1'],scale=x['scale_b1']), rv_b=gamma(a=x['a_b2'],scale=x['scale_b2']), metric='ws', n_samples=n_samples), axis=1)
        if config['metrics']['jsd']:
            print(f'- A/A - Gamma - JSD...')
            df['P_A1A2_g_jsd'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x['a_a1'],scale=x['scale_a1']), rv_b=gamma(a=x['a_a2'],scale=x['scale_a2']), metric='jsd', n_samples=n_samples), axis=1)
            df['P_B1B2_g_jsd'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x['a_b1'],scale=x['scale_b1']), rv_b=gamma(a=x['a_b2'],scale=x['scale_b2']), metric='jsd', n_samples=n_samples), axis=1)

        # A/B test - both distributions
        print(f'- calc. P(B>A)...')
        results = df.progress_apply(lambda x: self.p_ab_loss( [beta(x['alpha_a'],x['beta_a']), beta(x['alpha_b'],x['beta_b'])], thr=1, n_samples=n_samples), axis=1)
        df['P_AB_b'] = [_[0][0] for _ in results]
        df['P_BA_b'] = [_[0][1] for _ in results]
        df['loss_ctr_a'] = [_[1][0] for _ in results]
        df['loss_ctr_b'] = [_[1][1] for _ in results]
    
        # df['P_BA_g'] = df.progress_apply(lambda x: self.p_ba(rv_a=gamma(a=x['a_a'],scale=x['scale_a']), rv_b=gamma(a=x['a_b'],scale=x['scale_b']), n_samples=n_samples), axis=1)
        # df['P_AB_g'] = 1 - df.P_BA_g
        # P = df.progress_apply(lambda x: self.p_ab( [gamma(a=x['a_a'],scale=x['scale_a']), gamma(a=x['a_b'],scale=x['scale_b'])], thr=1, n_samples=n_samples), axis=1)
        results = df.progress_apply(lambda x: self.p_ab_loss( [gamma(a=x['a_a'],scale=x['scale_a']), gamma(a=x['a_b'],scale=x['scale_b'])], thr=1, n_samples=n_samples), axis=1)
        df['P_AB_g'] = [_[0][0] for _ in results]
        df['P_BA_g'] = [_[0][1] for _ in results]
        df['loss_cpc_a'] = [_[1][0] for _ in results]
        df['loss_cpc_b'] = [_[1][1] for _ in results]
    
        if config['metrics']['ks']:
            print(f'- calc. AB - Kolmogorov-Smirnov...')
            df['P_AB_b_ks'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_a'],x['beta_a']), rv_b=beta(x['alpha_b'],x['beta_b']), metric='ks', n_samples=n_samples), axis=1)
            df['P_AB_g_ks'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x['a_a'],scale=x['scale_a']), rv_b=gamma(a=x['a_b'],scale=x['scale_b']), metric='ks', n_samples=n_samples), axis=1)
        if config['metrics']['ws']:
            print(f'- calc. AB - Wasserstein...')
            df['P_AB_b_ws'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_a'],x['beta_a']), rv_b=beta(x['alpha_b'],x['beta_b']), metric='ws', n_samples=n_samples), axis=1)
            df['P_AB_g_ws'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x.a_a,scale=x.scale_a), rv_b=gamma(a=x.a_b,scale=x.scale_b), metric='ws', n_samples=n_samples), axis=1)

        return df

    # execute
    def transform(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Args:
            df (pd.DataFrame): dataframe with all observed impression / clicks / conversions
            config (Dict): configuration parameters
        """
        df = self.agg_stats(df)
        df = self.calc_performance(df, config)

        return df
