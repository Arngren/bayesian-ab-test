#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Union
import numpy as np
import pandas as pd
import itertools

from scipy import stats
from scipy.stats import beta, gamma

from bayesian_test import Bayesian_AB_Test


__author__ = "Morten Arngren"

class Environment:
    """
    """
    def __init__(self, env_params: Dict, config: Dict) -> None:
        """ Inits class - store params locally

        Args:
            env_params (Dict): Dictionary holding all environment parameters
            config (Dict): Dictionary holding all config parameters
        """
        self.env_params = env_params
        
        self.n_impr_per_period = config['max_impr_before_update_param']


    def cost(self, impr: int, cpm_mean: float) -> float:
        """ Calc. cost of choosing the wrong winner

        Args:
            impr (int): number of impressions
            cpm_mean (float): Cost-per-Mill avg. value
        """
        return impr * max(np.random.normal(loc=cpm_mean, scale=2),1) / 1000


    def simulate_impr_one_period(self) -> pd.DataFrame:
        """ Generate random number of impression from environment

        Returns:
            n_impr (int): number of impressions
        """
        n_impr = np.random.randint(0, self.n_impr_per_period)
        return n_impr


    def simulate_clicks_one_period(self, variant: str, n_impr: int) -> pd.DataFrame:
        """ Generate random clicks from environment

        Args:
            variant (str): variant, eg. A1 / B2
            n_impr (int): number of impressions to serve

        Returns:
            n_clicks (int): number of clicks
        """
        n_clicks = np.random.binomial(n_impr, self.env_params[variant]['ctr'])
        return n_clicks


    def sim_clicks_cost_one_period(self, selected_variants: Dict) -> pd.DataFrame:
        """ Simulate campaign observation stastistics for one period

        Args:
            selected_variants (Dict): Dictionary holding all selected variants

        Returns:
            df (pd.DataFrame): dataframe with simulated impressions, clicks and cost for all variants
        """
        variants = list(selected_variants.keys())

        response = {}

        # Simulate impressions and clicks for all variants
        for variant in variants:
            ctr = self.env_params[variant]['ctr']
            cpm = self.env_params[variant]['cpm']

            # This should be modified to sample from the beta distribution for each variant
            n_impr = selected_variants[variant]

            response.update( {variant: {'clicks': np.random.binomial(n_impr, ctr),
                                        'cost': self.cost(n_impr, cpm) # TODO: check
                                       } } )

        return response

# =============================================================================
class Agent:
    """
    """
    def __init__(self, variants: List, config: Dict) -> None:
        """ Inits class - store params locally

        Args:   
            variants (List): List of variants
            config (Dict): Dictionary holding all config parameters
        """
        # Init name of variants
        self.variants = np.array(variants)

        # Init variants with zero impressions and clicks
        self.df_log = {}
        for variant in variants:
            self.create_log(variant, period=0)

        self.optimise_for = config['optimise_for']
        self.recency_param = config['recency_param']
        self.n_periods_per_day = config['n_periods_per_day']


    def create_log(self, variant: str, period: int = 0) -> None:
        """ Create log for agent

        Args:
            variant (str): name of the variant
            period (int): current period
        """
        self.df_log[variant] = pd.DataFrame( {  "period": period,
                                                "n_impr": 0,
                                                "n_impr_w_sum" : 0,
                                                "n_clicks": 0,
                                                "n_clicks_w_sum": 0,
                                                "cost": 0,
                                                "cost_sum": 0,
                                                "ctr": 0,
                                                "cpc": 0,
                                                "alpha": 1,
                                                "beta": 1,
                                                "a": 1,
                                                "scale": 1000
                                              }, index=[0] )

    def update_recency(self, variant: str, period: int) -> None:
        """ Update recency for each variant, both impressions and clicks
            As current timestanmpo can change, it updates all rows on only the following columns:
                - delta
                - recency
                - n_impr_w
                - n_clicks_w

        Args:
            variant (str): name of the variant
            period (int): current period

        """
        # Calc. recency
        self.df_log[variant]['delta'] = (period - self.df_log[variant]['period']) // self.n_periods_per_day
        self.df_log[variant]['recency'] = self.recency_param**self.df_log[variant]['delta']
        self.df_log[variant]['n_impr_w'] = self.df_log[variant]['n_impr'] * self.df_log[variant]['recency']
        self.df_log[variant]['n_clicks_w'] = self.df_log[variant]['n_clicks'] * self.df_log[variant]['recency']


    def choose(self, period: int, n_impr: int) -> Union[str, int]:
        """ Choose which variants to serve next

        Args:
            period (int): current period
            n_impr (int): number of impressions to serve

        Returns:
            selected_variants (str): names of the variant to serve
        """
        samples = {}

        # Sample from beta distr. from each variant
        for variant in self.variants:
            # self.update_recency(variant=variant, period=period)

            # Get alpha and beta from previous period
            df_prev = self.df_log[variant][self.df_log[variant]['period']==period-1]

            # Model a beta distribution for each variant using the observed impressions and clicks
            # and draw n_impr samples from the beta distribution
            if self.optimise_for == 'ctr':
                samples[variant] = stats.beta(a=df_prev.alpha, b=df_prev.beta).rvs(n_impr)
            if self.optimise_for == 'cpc':
                samples[variant] = stats.gamma(a=df_prev.a, scale=df_prev.scale).rvs(n_impr)

        # Thompson sampling - from samples for each variant, select the variant with the highest sample (highest ctr)
        if self.optimise_for == 'ctr':
            # Select variant with highest sample (highest ctr)
            selected_variants = np.array([samples[variant] for variant in self.variants]).argmax(axis=0)
        if self.optimise_for == 'cpc':
            # Select variant with lowest sample (cheapest)
            selected_variants = np.array([samples[variant] for variant in self.variants]).argmin(axis=0)
        
        # Convert to variant names and count occurences of each variant
        selected_variants = {variant: sum(self.variants[selected_variants]==variant) for variant in self.variants}

        return selected_variants

    def update_log(self, period: int, selected_variants: Dict, env_response: Dict) -> None:
        """ Update log for each variant

        Args:
            period (int): current period
            selected_variants (Dict): Dictionary holding all selected variants  
            env_response (Dict): Dictionary holding all environment responses

        Returns:
            log (Dict): Dictionary holding all log entries
        """
        log = {}
        # Update log for each variant
        for variant in self.variants:

            # Calc. recency
            delta = (period - self.df_log[variant]['period']) // self.n_periods_per_day
            recency_decay = self.recency_param**delta

            n_impr_w_sum = sum( self.df_log[variant]['n_impr'] * recency_decay ) + 1
            n_clicks_w_sum = sum( self.df_log[variant]['n_clicks'] * recency_decay ) + 1
            cost_w_sum = sum( self.df_log[variant]['cost'] * recency_decay ) + 1

            ctr = n_clicks_w_sum / n_impr_w_sum
            cpc = cost_w_sum / n_clicks_w_sum

            alpha = np.round(n_clicks_w_sum + 1).astype(int)
            beta = np.round(n_impr_w_sum - n_clicks_w_sum + 1).astype(int)

            a = cost_w_sum + 1
            scale = 1 / (n_clicks_w_sum + 1e-16)
            
            # Add newest raw entries to log
            tmp = { 'period': period,
                   
                    'n_impr': selected_variants[variant], # current impressions
                    'n_impr_w_sum': n_impr_w_sum, # accumulated weighted impressions from all previous periods

                    'n_clicks': env_response[variant]['clicks'],
                    'n_clicks_w_sum': n_clicks_w_sum,

                    'cost': env_response[variant]['cost'],
                    'cost_w_sum': cost_w_sum,

                    'ctr': ctr,
                    'cpc': cpc,

                    # For plotting the beta & gamma distributions
                    'alpha': alpha,
                    'beta': beta,
                    'a': a,
                    'scale': scale
            }
            # Append as additional row to df_log
            df_tmp = pd.DataFrame(tmp, index=[0])
            self.df_log[variant] = pd.concat([self.df_log[variant], df_tmp], ignore_index=True)

            # append to history of agents log
            log[variant] = self.df_log[variant][self.df_log[variant]['period']==period].to_dict(orient='records')[0]

        return log


# =============================================================================
class Bandit:
    """ Class running the Bandit in a simulated environment
    """

    def __init__(self, bandit_params: Dict, n_periods: int, config: Dict) -> None:
        """ Inits class - store params locally

        Args:
            bandit_params (Dict): Dictionary holding all bandit parameters
            n_periods (int): Number of periods to run simulations
            config (Dict): Dictionary holding all config parameters
        """
        self.params = bandit_params

        # Extract variants with period 0
        variants = [variant for variant in self.params if self.params[variant]['period'] == 0]

        # Set total number of period for the experiment
        self.n_periods = n_periods

        # Extract active variant with highest ctr
        self.optimal_variant = max(bandit_params, key=lambda x: bandit_params[x]['ctr'] if x in variants else 0)

        # Init environment and agent
        self.environment = Environment(env_params=self.params, config=config)
        self.agent = Agent(variants=variants, config=config)

        # Utility functions
        self.bayes = Bayesian_AB_Test()

    def add_variant(self, variant: str, period: int, env_param: Dict) -> None:
        """ Cold start - add another variant suddenly

        Args:
            variant (str): name of the variant
            period (int): current period    
            env_param (Dict): Dictionary holding all environment parameters
        """
        # Add new variant to agent and log
        self.agent.variants = np.append(self.agent.variants, variant)
        self.agent.create_log(variant, period=period)
        self.environment.env_params[variant] = env_param
        self.optimal_variant = max(self.params, key=lambda x: self.params[x]['ctr'] if x in self.agent.variants else 0)

    def run(self):
        """ Run the bandit
            Iterate over all periods and update agent and environment
            Calculate metrics
            Inject new variants at specific periods
        """
        self.df_metrics = pd.DataFrame()
        for period in range(1, self.n_periods+1):

            # ENVIRONMENT - GET IMPRESSIONS
            # random, how many impr. before updating agent
            n_impr = self.environment.simulate_impr_one_period()

            # AGENT - CHOOSE
            selected_variants = self.agent.choose(n_impr=n_impr, period=period)

            # ENVIRONMENT - RETURN CLICKS
            response = self.environment.sim_clicks_cost_one_period(selected_variants)

            # AGENT - UPDATE LOG
            log = self.agent.update_log(period=period, selected_variants=selected_variants, env_response=response)

            # METRICS
            # Regret
            regret = sum( [selected_variants[k] for k in selected_variants if k != self.optimal_variant] )

            # Calc. probabilities and losses
            P_ab_ctr = self.bayes.p_ab( [beta(log[variant]['alpha'], log[variant]['beta']) for variant in self.agent.variants] )
            P_ab_cpc = self.bayes.p_ab( [gamma(log[variant]['a'], log[variant]['scale']) for variant in self.agent.variants] )

            loss = {'ctr': {}, 'cpc': {}}
            p_overlap = {'ctr': {'ks':{}, 'ws':{}}, 'cpc': {'ks':{}, 'ks':{}}}
            for a, b in itertools.combinations(self.agent.variants, 2):    
                loss['ctr'][a, b] = self.bayes.loss( beta(log[a]['alpha'],log[a]['beta']),
                                                    beta(log[b]['alpha'],log[b]['beta']) )
                loss['cpc'][a, b] = self.bayes.loss( gamma(log[a]['a'],log[a]['scale']),
                                                    gamma(log[b]['a'],log[b]['scale']), f_max=0 )

                # add AA-test using kolmogorov-smirnof test
                p_overlap['ctr']['ks'][a,b] = self.bayes.p_overlap(rv_a=beta(log[a]['alpha'],log[a]['beta']),
                                                             rv_b=beta(log[b]['alpha'],log[b]['beta']),
                                                             metric='ks',
                                                             n_samples=1000)
                # add AA-test using kolmogorov-smirnof test
                p_overlap['ctr']['ks'][a,b] = self.bayes.p_overlap(rv_a=gamma(log[a]['a'],log[a]['scale']),
                                                             rv_b=gamma(log[b]['a'],log[b]['scale']),
                                                             metric='ks',
                                                             n_samples=1000)           

            # append df_tmp to df_metrics
            df_tmp = {'period': period, 'n_impr': n_impr,
                      'regret': regret,
                      'P_ab_ctr': [P_ab_ctr], 'P_ab_cpc': [P_ab_cpc],
                      'loss_ctr': [loss['ctr']], 'loss_cpc': [loss['cpc']],
                      'p_overlap_ctr': [p_overlap['ctr']], 'p_overlap_cpc': [p_overlap['cpc']]
                      }
            self.df_metrics = pd.concat([self.df_metrics, pd.DataFrame(df_tmp, index=[0])], ignore_index=True)

            self.df_metrics['n_impr_acc'] = self.df_metrics['n_impr'].cumsum()
            self.df_metrics['regret_acc'] = self.df_metrics['regret'].cumsum()
            self.df_metrics['regret_avg'] = self.df_metrics.regret_acc / self.df_metrics.n_impr_acc

            # AGENT - cold start - add another variant suddenly
            for variant in self.params:
                if self.params[variant]['period'] == period:
                    self.add_variant(variant=variant, period=period, env_param=self.params[variant])
