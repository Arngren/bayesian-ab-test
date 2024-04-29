#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict
import numpy as np
import pandas as pd

__author__ = "Morten Arngren"

class Campaign:
    """ Campaign class for simulating a campaign with specified parameters
    """

    def __init__(self, params: Dict, random_seed: int = 42) -> None:
        """ Inits class - store params locally
            expecting exmaple:
            params = {'A': {'ctr': 0.10, 'cpm':11},
                      'B': {'ctr': 0.11, 'cpm':12},
                      'n_impr_per_period':5}

        Args:
            params (Dict): Dictionary holding all campaign parameters
            random_seed (int): random seed for reproducibility
        """
        self.ctr_a = params['A']['ctr']
        self.ctr_b = params['B']['ctr']
        self.cpm_mean_a = params['A']['cpm']
        self.cpm_mean_b = params['B']['cpm']
        self.n_impr_per_period = params['n_impr_per_period']

        # set random seed
        np.random.seed(random_seed)



    def cost(self, impr: int, cpm_mean: float) -> float:
        """ Calc. cost of choosing the wrong winner

        Args:
            impr (int): number of impressions
            cpm_mean (float): Cost-per-Mill avg. value
        """
        return impr * max(np.random.normal(loc=cpm_mean, scale=2),1) / 1000


    def sim_impr_clicks_cost(self, variant: str, ctr: float, cpm: float, n_periods: int) -> pd.DataFrame:
        """ Simulate campaign observation stastistics for specific period

        Args:
            variant (str): variant, eg. A1 / B2
            ctr (float): Click-Trough-Rate
            cpm (float): Cost-per-Mill - cost per 1000 (not million)
            n_periods (int): Number of periods to run simulations
        """
        # init empty dataframe
        df = pd.DataFrame()

        # simulate impressions and clicks for variant A1
        df[f'n_impr_{variant}'] = np.random.randint(0, self.n_impr_per_period, n_periods)
        df[f'n_clicks_{variant}'] = df.apply(lambda x: np.random.binomial(x[f'n_impr_{variant}'], ctr), axis=1)
        df[f'cost_{variant}'] = df.apply(lambda x: self.cost(x[f'n_impr_{variant}'], cpm), axis=1)

        # insert row with zeros at position 0
        df = pd.concat([pd.DataFrame([[0,0,0.0]], columns=[f'n_impr_{variant}', f'n_clicks_{variant}', f'cost_{variant}']), df], ignore_index=True)

        return df


    def create(self, n_periods: int=1000) -> pd.DataFrame:
        """ Create dataframe with simulated observation stastistics for specific period

        Args:
            n_periods (int): number of simulation periods
        """
        # simulate campaign events
        df = pd.DataFrame({'t':range(n_periods)})

        # simulate impressions, clicks and costs for variants A1 / A2 / B1 / B2
        df_a1 = self.sim_impr_clicks_cost('a1', self.ctr_a, self.cpm_mean_a, n_periods)
        df_a2 = self.sim_impr_clicks_cost('a2', self.ctr_a, self.cpm_mean_a, n_periods)
        df_b1 = self.sim_impr_clicks_cost('b1', self.ctr_b, self.cpm_mean_b, n_periods)
        df_b2 = self.sim_impr_clicks_cost('b2', self.ctr_b, self.cpm_mean_b, n_periods)

        # combine all simulations into single dataframe
        df = pd.concat([df, df_a1, df_a2, df_b1, df_b2], axis=1)

        # combine A1 and A2
        df['n_impr_a'] = df.n_impr_a1 + df.n_impr_a2
        df['n_clicks_a'] = df.n_clicks_a1 + df.n_clicks_a2
        df['cost_a'] = df.cost_a1 + df.cost_a2

        # combine B1 and B2
        df['n_impr_b'] = df.n_impr_b1 + df.n_impr_b2
        df['n_clicks_b'] = df.n_clicks_b1 + df.n_clicks_b2
        df['cost_b'] = df.cost_b1 + df.cost_b2

        return df


    def agg_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """ calc. aggregations for all events

        Args:
            df (DataFrame): dataframe with all observed impression / clicks / conversions
        """
        # accumulate over time
        df['acc_impr_a1'] = df.n_impr_a1.cumsum()
        df['acc_cost_a1'] = df.cost_a1.cumsum()
        df['acc_clicks_a1'] = df.n_clicks_a1.cumsum()

        df['acc_impr_a2'] = df.n_impr_a2.cumsum()
        df['acc_cost_a2'] = df.cost_a2.cumsum()
        df['acc_clicks_a2'] = df.n_clicks_a2.cumsum()
        
        df['acc_impr_a'] = df.n_impr_a.cumsum()
        df['acc_cost_a'] = df.cost_a.cumsum()
        df['acc_clicks_a'] = df.n_clicks_a.cumsum()

        df['acc_impr_b1'] = df.n_impr_b1.cumsum()
        df['acc_cost_b1'] = df.cost_b1.cumsum()
        df['acc_clicks_b1'] = df.n_clicks_b1.cumsum()
        
        df['acc_impr_b2'] = df.n_impr_b2.cumsum()
        df['acc_cost_b2'] = df.cost_b2.cumsum()
        df['acc_clicks_b2'] = df.n_clicks_b2.cumsum()
        
        df['acc_impr_b'] = df.n_impr_b.cumsum()
        df['acc_cost_b'] = df.cost_b.cumsum()
        df['acc_clicks_b'] = df.n_clicks_b.cumsum()

        return df
