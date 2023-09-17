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

    def B(self, alpha, beta):
        """ mapper function for the beta distribution
        """
        return scipy.special.beta(alpha, beta)

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

    def p_ab(self, rvs: List, best: str = 'max', thr: float = 1, n_samples: int =10000):
        """ Calc. probability that all variant are better than the rest
            one by one.

        Args:
            rvs (List): list of scipy.stats objects
            thr (float, optional): threshold. Defaults to 1.
            n_samples (int, optional): number of samples. Defaults to 10000.

        Returns:
            P_ab_thr (List): list of probabilities that each variant is better than the rest
        """
        # Generate samples from all variants
        samples = np.array( [rv.rvs(size=n_samples) for rv in rvs] )

        # Calc. probability that a variant is better than the rest
        P_ab_thr = []
        for id_ref in range(len(rvs)):
            # Identify the rest of the variants
            id_rest = [i for i in range(len(rvs)) if i != id_ref]
            # Calc. probability ratio that ref is better than the rest
            P_ratio = samples[id_ref] / np.max(samples[id_rest], axis=0)
            # Calc. prob. mass above threshold and save for each variant
            if best == 'max':
                P_ab_thr += [ (P_ratio>thr).sum() / n_samples ]
            if best == 'min':
                P_ab_thr += [ (P_ratio<thr).sum() / n_samples ]
        return P_ab_thr

    def p_ba(self, rv_a, rv_b, n_samples: int=10000) -> float:
        """ probability of B having higher performance than A

        Args:
            rv_a (scipy.stats): random variable function for variant A
            rv_b (scipy.stats): random variable function for variant B
            n_samples (int): number of samples
        """
        # sampling our way out of it...
        P = np.sum( rv_a.rvs(size=n_samples) < rv_b.rvs(size=n_samples) ) / n_samples
        return P

    def p_ba_beta(self, alpha_a: float, beta_a: float, alpha_b: float, beta_b: float) -> float:
        """ probability of B having higher performance than A assuming the Beta distribution.
            ref: https://www.evanmiller.org/bayesian-ab-testing.html#implementation

        Args:
            alpha_a (float): alpha parameter for rv A
            beta_a (float): beta parameter for rv A
            alpha_b (float): alpha parameter for rv B
            beta_b (float): beta parameter for rv B
        """
        P = np.sum( [ np.exp( betaln(alpha_a+i, beta_b+beta_a) \
                    - np.log(beta_b+i) \
                    - betaln(1+i, beta_b) \
                    - betaln(alpha_a, beta_a) ) for i in range(alpha_b) ] )

        return P


    def loss(self, rv_a, rv_b, f_max: float=1, N: int=100) -> List:
        """ calc. the loss - ie. amount of performance lost if wrong variant is chosen
        
        Args:
            rv_a (scipy.stats): random variable function for variant A
            rv_b (scipy.stats): random variable function for variant B
            f_max (float): max. value for the pdf
            N (int): number of pdf divisions for integration
        """
        # util function to calc. loss
        def __loss(i, j):
            return max(j/N - i/N, 0)

        # util function
        def __joint_posterior_array(rv_a, rv_b, f_max, N=100):
            joint = np.array( [rv_a.pdf(ii) * rv_b.pdf(np.linspace(0,f_max,N)) for i,ii in enumerate(np.linspace(0,f_max,N))] ) + 1e-16
            return joint/joint.sum()

        loss_a, loss_b = 0, 0
        # calc. f_max based in std of gamma distributions
        # if isinstance(rv_a, gamma):
        #     f_max = 5 * rv_a.std()
        # if isinstance(rv_b, gamma):
        #     f_max = max(f_max, 5 * rv_b.std())
        if f_max == 0:
            f_max = max(5*rv_a.std(), 5*rv_b.std())

        # calc. loss
        joint = __joint_posterior_array(rv_a, rv_b, f_max=f_max, N=N)
        for i in range(N):
            loss_a += sum( [joint[i,j]*__loss(i,j) for j in range(N)] )
            loss_b += sum( [joint[i,j]*__loss(j,i) for j in range(N)] )

        return loss_a, loss_b

    def loss_parallel(self, x) -> List:
        """ wrapper function for calc. loss in parallel using multiprocessing
            assuming the gamma distribution!

        Args:
            x (pandas df row): probability distribution parameters
        """
        x = x[1]
        return self.loss(rv_a=gamma(a=x.a_a, scale=x.scale_a), rv_b=gamma(a=x.a_b, scale=x.scale_b))
        
    def loss_beta(self, alpha_a: float, beta_a: float, alpha_b: float, beta_b: float, n_samples: int=1000) -> List:
        """ https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html

        Args:
            alpha_a (float): alpha parameter for rv A
            beta_a (float): beta parameter for rv A
            alpha_b (float): alpha parameter for rv B
            beta_b (float): beta parameter for rv B
        """
        # analytically
        from scipy.special import beta as B
        a, b, c, d = int(alpha_a), int(beta_a), int(alpha_b), int(beta_b)

        # normal domain - not numerically stable
        # loss_a = B(a+1, b) / B(a, b) * (1-self.p_ba_anal(a+1, b, c, d)) \
        #        - B(c+1, d) / B(c, d) * (1-self.p_ba_anal(a, b, c+1, d))
        # loss_b = B(c+1, d) / B(c, d) * (1-self.p_ba_anal(c+1, d, a, b) \
        #        - B(a+1, b) / B(a, b) * (1-self.p_ba_anal(c, d, a+1, b)

        # log domain calc. - TODO: p_ab has two outputs...[1] correct?
        loss_a = np.exp( betaln(c+1, d) - betaln(c, d) + np.log(1-self.p_ab([beta(c+1,d), beta(a,b)], n_samples=n_samples)[1]) ) \
               - np.exp( betaln(a+1, b) - betaln(a, b) + np.log(1-self.p_ab([beta(c,d), beta(a+1,b)], n_samples=n_samples)[1]) )
        loss_b = np.exp( betaln(a+1, b) - betaln(a, b) + np.log(1-self.p_ab([beta(a+1,b), beta(c,d)], n_samples=n_samples)[1]) ) \
               - np.exp( betaln(c+1, d) - betaln(c, d) + np.log(1-self.p_ab([beta(a,b), beta(c+1,d)], n_samples=n_samples)[1]) )

        return loss_a, loss_b


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
        P = df.progress_apply(lambda x: self.p_ab( [beta(x['alpha_a'],x['beta_a']), beta(x['alpha_b'],x['beta_b'])], thr=1, n_samples=n_samples), axis=1)
        df['P_AB_b'] = [_[0] for _ in P]
        df['P_BA_b'] = [_[1] for _ in P]
    
        # df['P_BA_g'] = df.progress_apply(lambda x: self.p_ba(rv_a=gamma(a=x['a_a'],scale=x['scale_a']), rv_b=gamma(a=x['a_b'],scale=x['scale_b']), n_samples=n_samples), axis=1)
        # df['P_AB_g'] = 1 - df.P_BA_g
        P = df.progress_apply(lambda x: self.p_ab( [gamma(a=x['a_a'],scale=x['scale_a']), gamma(a=x['a_b'],scale=x['scale_b'])], thr=1, n_samples=n_samples), axis=1)
        df['P_AB_g'] = [_[0] for _ in P]
        df['P_BA_g'] = [_[1] for _ in P]
    
        if config['metrics']['ks']:
            print(f'- calc. AB - Kolmogorov-Smirnov...')
            df['P_AB_b_ks'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_a'],x['beta_a']), rv_b=beta(x['alpha_b'],x['beta_b']), metric='ks', n_samples=n_samples), axis=1)
            df['P_AB_g_ks'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x['a_a'],scale=x['scale_a']), rv_b=gamma(a=x['a_b'],scale=x['scale_b']), metric='ks', n_samples=n_samples), axis=1)
        if config['metrics']['ws']:
            print(f'- calc. AB - Wasserstein...')
            df['P_AB_b_ws'] = df.progress_apply(lambda x: self.p_overlap(rv_a=beta(x['alpha_a'],x['beta_a']), rv_b=beta(x['alpha_b'],x['beta_b']), metric='ws', n_samples=n_samples), axis=1)
            df['P_AB_g_ws'] = df.progress_apply(lambda x: self.p_overlap(rv_a=gamma(a=x.a_a,scale=x.scale_a), rv_b=gamma(a=x.a_b,scale=x.scale_b), metric='ws', n_samples=n_samples), axis=1)

        # calc. loss
        print(f'- calc. loss...')
        df['loss_ctr'] = df.progress_apply(lambda x: self.loss_beta(x.alpha_a, x.beta_a, x.alpha_b, x.beta_b), axis=1)
        df['loss_ctr_a'] = df.loss_ctr.apply(lambda x: x[0])
        df['loss_ctr_b'] = df.loss_ctr.apply(lambda x: x[1])
        df = df.drop(columns=['loss_ctr'])

        # calc. cpc loss in parallel
        with mp.Pool(16) as pp:
            tmp = pp.map(self.loss_parallel, df.iterrows())
        df['loss_cpc_a'] = [_[1] for _ in tmp]
        df['loss_cpc_b'] = [_[0] for _ in tmp]
        # serial execution - slow...
        # df['loss_cpc'] = df.progress_apply(lambda x: self.loss(rv_a=gamma(a=x.a_a,scale=x.scale_a), rv_b=gamma(a=x.a_b,scale=x.scale_b)), axis=1)
        # df['loss_cpc_a'] = df.loss_cpc.apply(lambda x: x[1]) # flip them around as lower CpC is better
        # df['loss_cpc_b'] = df.loss_cpc.apply(lambda x: x[0])
        # df = df.drop(columns=['loss_cpc'])

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
