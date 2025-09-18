# import from ab_test in previuos directory
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('src')

from typing import List

import numpy as np
import streamlit as st

from utils.hypothesis_test import Hypothesis_AB_Test
from utils.bayesian_test import Bayesian_AB_Test
from utils.ui import ABTestUtils, show_hypothesis_input_fields, ShowButtons
from scipy.stats import beta, gamma, norm

import plotly
import plotly.graph_objects as go
from ab_test.graph import Visualisation

import streamlit as st

# Main function
class EvaluationAPP:
    def __init__(self) -> None:
        st.set_page_config(layout="wide")

        self.hypo = Hypothesis_AB_Test()
        self.bayes = Bayesian_AB_Test()

        self.thr_success = 0.7

        self.plot = Visualisation() # (renderer="vscode")
        plotly.io.json.config.default_engine = 'orjson'
        self.plot_width = 700

        self.color_default = "#EEEEDD"
        self.color_succes = "#88DD88"
        self.color_failure = "#FF8888"

        # pre-select test type, metric and best variant
        if 'test_type' not in st.session_state:
            st.session_state['test_type'] = 'Bayesian Test'
        if 'metric' not in st.session_state:
            st.session_state['metric'] = 'CTR'
        if 'best' not in st.session_state:
            st.session_state['best'] = 'max'

        # initialize variables
        self.utils = ABTestUtils()
        self.show_buttons = ShowButtons()


    def block_bayes_results(self, metric: str, P_ab_thr, loss, best='max'):
        """Create a block with two values side by side, colored according to the color parameter.
        
        Args:
            metric (str): The metric to display.
            P_ab_thr (List): List of probabilities that B > A.
            loss (List): List of expected losses.
            best (str, optional): The best variant. Defaults to 'max'.
        """
        col_1, col_2 = st.columns(2)
        with col_1:
            color_a = self.color_succes if P_ab_thr[0] > self.thr_success else self.color_failure
            color_b = self.color_succes if P_ab_thr[1] > self.thr_success else self.color_failure
            self.utils.show_dual_block(f"{P_ab_thr[0]*100:.1f}%", color_a,
                             f"{P_ab_thr[1]*100:.1f}%", color_b,
                             "Probability of winner (A | B)")
        with col_2:
            color_a = self.color_succes if loss[0] < loss[1] else self.color_failure
            color_b = self.color_succes if loss[1] < loss[0] else self.color_failure
            txt_a = f"${loss[0]:.2f}" if metric in ['CpC', 'CpA'] else f"{100*loss[0]:.2f}%"
            txt_b = f"${loss[1]:.2f}" if metric in ['CpC', 'CpA'] else f"{100*loss[1]:.2f}%"
            self.utils.show_dual_block(txt_a, color_a, txt_b, color_b, "Expected Loss (A | B)")

        # if P_ab_thr[0] > self.thr_success:
        #     # st.markdown(f"<br>", unsafe_allow_html=True)
        #     self.utils.show_headline('Variant A is the winner', 'h1', color=self.color_succes)
        # if P_ab_thr[1] > self.thr_success:
        #     # st.markdown(f"<br>", unsafe_allow_html=True)
        #     self.utils.show_headline('Variant B is the winner', 'h1', color=self.color_succes)
        # if P_ab_thr[0] <= self.thr_success and P_ab_thr[1] <= self.thr_success:
        #     # st.markdown(f"<br>", unsafe_allow_html=True)
        #     self.utils.show_headline('No clear winner', 'h1', color=self.color_failure)
        st.markdown(f"<br>", unsafe_allow_html=True)



    def show_metrics(self, metric):
        """Show the metrics in a block."""
        # calculate metrics
        col_1, col_2 = st.columns(2)
        with col_1:
            if metric == 'CTR':
                self.ctr_A = self.click_A / self.impr_A if self.impr_A > 0 else 0
                self.utils.show_value_block(f"{self.ctr_A*100:.1f}%", "CTR A", color=self.color_default)
            if metric == 'CVR':
                self.cvr_A = self.conv_A / self.click_A if self.click_A > 0 else 0
                self.utils.show_value_block(f"{self.cvr_A*100:.1f}%", "CVR A", color=self.color_default)
            if metric == 'CpC':
                self.cpc_A = self.cost_A / self.click_A if self.click_A > 0 else 0
                self.utils.show_value_block(f"${self.cpc_A:.2f}", "CpC A", color=self.color_default)
            if metric == 'CpA':
                self.cpa_A = self.cost_A / self.conv_A if self.conv_A > 0 else 0
                self.utils.show_value_block(f"${self.cpa_A:.2f}", "CpA A", color=self.color_default)
        with col_2:
            if metric == 'CTR':
                self.ctr_B = self.click_B / self.impr_B if self.impr_B > 0 else 0
                self.utils.show_value_block(f"{self.ctr_B*100:.1f}%", "CTR B", color=self.color_default)
            if metric == 'CVR':
                self.cvr_B = self.conv_B / self.click_B if self.click_B > 0 else 0
                self.utils.show_value_block(f"{self.cvr_B*100:.1f}%", "CVR B", color=self.color_default)
            if metric == 'CpC':
                self.cpc_B = self.cost_B / self.click_B if self.click_B > 0 else 0
                self.utils.show_value_block(f"${self.cpc_B:.2f}", "CpC B", color=self.color_default)
            if metric == 'CpA':
                self.cpa_B = self.cost_B / self.conv_B if self.conv_B > 0 else 0
                self.utils.show_value_block(f"${self.cpa_B:.2f}", "CpA B", color=self.color_default)

        st.markdown(f"<br>", unsafe_allow_html=True)


    def plot_insights(self, rv_a, rv_b, best='max'):
        """Plot the insights of the test.
        
        Args:
            rv_a (scipy.stats): Random variable of variant A.
            rv_b (scipy.stats): Random variable of variant B.
            best (str, optional): The best variant. Defaults to 'max'.        
        """
        thr = 1.0
        n_samples = 1_000_000

        samples_a = rv_a.rvs(size=n_samples)
        samples_b = rv_b.rvs(size=n_samples)
        # ratio
        ratio = samples_b / samples_a if best=='max' else samples_a / samples_b
        #95% quantile
        x_min, x_max = np.quantile(ratio, 1e-3), np.quantile(ratio, 1-1e-3)
        hist, bins = np.histogram(ratio, bins=np.linspace(x_min, x_max ,1001))
        id1, id2 = bins<=thr, bins>thr

        # ---------------------------------------------------------------------------------------------
        # calc. x_max from samples as 95% quantile
        x_min = min(rv_a.ppf(1e-3), rv_b.ppf(1e-3))
        x_max = max(rv_a.ppf(1-1e-3), rv_b.ppf(1-1e-3))

        x = np.linspace(x_min, x_max, 1001)
        p_data = [self.plot.plot(x=x, y=rv_a.pdf(x), color=0, opacity=0.7, name='A', showlegend=True),
                    self.plot.plot(x=x, y=rv_b.pdf(x), color=1, opacity=0.7, name='B', showlegend=True)]
        layout = self.plot.layout(title=f'Probability distributions of A & B', x_label=f"{st.session_state['metric']}", y_label='', theme='dark', width=self.plot_width, height=300)
        fig = go.Figure(data=p_data, layout=layout)
        st.plotly_chart(fig)

        # p_data = [self.plot.plot(x=bins[id1], y=hist[id1[:-1]]/sum(hist), color=0, opacity=0.7, name='A', showlegend=True),
        #             self.plot.plot(x=bins[id2], y=hist[id2[:-1]]/sum(hist), color=1, opacity=0.7, name='B', showlegend=True) ]
        # layout = self.plot.layout(title='Distribution of P(B>A)', x_label='lift ratio', y_label='', theme='dark', width=self.plot_width, height=300)
        # fig = go.Figure(data=p_data, layout=layout)
        # st.plotly_chart(fig)

        # if best == 'max':
        #     y1 = 1-hist.cumsum()[id1[:-1]]/sum(hist)
        #     y2 = 1-hist.cumsum()[id2[:-1]]/sum(hist)
        # else:
        #     y1 = hist.cumsum()[id1[:-1]]/sum(hist)
        #     y2 = hist.cumsum()[id2[:-1]]/sum(hist)
        # p_data = [self.plot.plot(x=bins[id1], y=y1, color=0, opacity=0.7, name='A', showlegend=True),
        #             self.plot.plot(x=bins[id2], y=y2, color=1, opacity=0.7, name='B', showlegend=True) ]
        # layout = self.plot.layout(title='Cumulative distribution of P(B>A)', x_label='lift ratio', y_label='', theme='dark', width=self.plot_width, height=300)
        # fig = go.Figure(data=p_data, layout=layout)
        # st.plotly_chart(fig)




    ############################################################################################
    def main(self):
        """Run this function to display the Streamlit app."""

        # Set Streamlit app title
        self.utils.show_headline('LAB', 'h1')
        self.utils.show_headline('Beta-distribution vs. Gaussian-distribution', 'h3')

        # create two html blocks side by side where data is entered for control and test group
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h4 style='text-align: center; margin-bottom: 0px; color: {self.color_default}'>Variant A</h4>", unsafe_allow_html=True)
            self.impr_A = st.number_input('# Impressions', min_value=0, max_value=None, value=100, step=1, format='%d', key='impr_A')
            self.click_A = st.number_input('# Clicks', min_value=0, max_value=None, value=10, step=1, format='%d')
            self.conv_A = st.number_input('# Conversions', min_value=0, max_value=None, value=2, step=1, format='%d')
            self.cpm_A = st.number_input('Cost-per-Mill (CPM)', min_value=0, max_value=None, value=11, step=1, format='%d')
        with col2:
            st.markdown(f"<h4 style='text-align: center; margin-bottom: 0px; color: {self.color_default}'>Variant B</h4>", unsafe_allow_html=True)
            self.impr_B = st.number_input('# Impressions', min_value=0, max_value=None, value=100, step=1, format='%d', key='impr_B')
            self.click_B = st.number_input('# Clicks', min_value=0, max_value=None, value=12, step=1, format='%d')
            self.conv_B = st.number_input('# Conversions', min_value=0, max_value=None, value=5, step=1, format='%d')
            self.cpm_B = st.number_input('Cost-per-Mill (CPM)', min_value=0, max_value=None, value=19, step=1, format='%d')

        # calculate actual costs
        self.cost_A = self.impr_A * self.cpm_A / 1000
        self.cost_B = self.impr_B * self.cpm_B / 1000

        # SHOW BUTTONS
        self.utils.show_headline("Metrics")
        self.show_buttons.metric()

        # setup the correct variables depending on metric
        if st.session_state['metric'] == 'CTR':
            n_success_a, n_trails_a = self.click_A, self.impr_A
            n_success_b, n_trails_b =self.click_B, self.impr_B
            perf_A, perf_B = self.click_A / self.impr_A, self.click_B / self.impr_B
            rv_a_beta, rv_b_beta = beta(self.click_A+1, self.impr_A-self.click_A), beta(self.click_B+1, self.impr_B-self.click_B)
            # Gaussian
            mean_a, std_a = perf_A, np.sqrt(perf_A * (1 - perf_A) / self.impr_A)
            mean_b, std_b = perf_B, np.sqrt(perf_B * (1 - perf_B) / self.impr_B)
            rv_a_gauss, rv_b_gauss = norm(mean_a, std_a), norm(mean_b, std_b)

        if st.session_state['metric'] == 'CVR':
            n_success_a, n_trails_a = self.conv_A, self.click_A
            n_success_b, n_trails_b =self.conv_B, self.click_B
            perf_A, perf_B = self.conv_A / self.click_A, self.conv_B / self.click_B
            rv_a_beta, rv_b_beta = beta(self.conv_A+1, self.click_A-self.conv_A), beta(self.conv_B+1, self.click_B-self.conv_B)
            # Gaussian
            mean_a, std_a = perf_A, np.sqrt(perf_A * (1 - perf_A) / self.click_A)
            mean_b, std_b = perf_B, np.sqrt(perf_B * (1 - perf_B) / self.click_B)
            rv_a_gauss, rv_b_gauss = norm(mean_a, std_a), norm(mean_b, std_b)

        if st.session_state['metric'] == 'CpC':
            n_success_a, n_trails_a = 0, 0
            n_success_b, n_trails_b =0, 0
            perf_A, perf_B = self.cost_A / self.click_A, self.cost_B / self.click_B
            rv_a_beta, rv_b_beta= gamma(a=self.cost_A+1, scale=1/self.click_A), gamma(a=self.cost_B+1, scale=1/self.click_B)
            # Gaussian
            mean_a, std_a = perf_A, np.sqrt(perf_A * (1 - perf_A) / self.click_A)
            mean_b, std_b = perf_B, np.sqrt(perf_B * (1 - perf_B) / self.click_B)
            rv_a_gauss, rv_b_gauss = norm(mean_a, std_a), norm(mean_b, std_b)

        if st.session_state['metric'] == 'CpA':
            n_success_a, n_trails_a = 0, 0
            n_success_b, n_trails_b =0,0
            perf_A, perf_B = self.cost_A / self.conv_A, self.cost_B / self.conv_B
            rv_a_beta, rv_b_beta = gamma(a=self.cost_A+1, scale=1/self.conv_A), gamma(a=self.cost_B+1, scale=1/self.conv_B)
            # Gaussian
            mean_a, std_a = perf_A, np.sqrt(perf_A * (1 - perf_A) / self.conv_A)
            mean_b, std_b = perf_B, np.sqrt(perf_B * (1 - perf_B) / self.conv_B)
            rv_a_gauss, rv_b_gauss = norm(mean_a, std_a), norm(mean_b, std_b)
            st.markdown(f"perf_A: {perf_A} - conv_A: {self.conv_A}", unsafe_allow_html=True)
            st.markdown(f"perf_B: {perf_B} - conv_B: {self.conv_B}", unsafe_allow_html=True)
            st.markdown(f"std_a: {std_a} - std_b: {std_b}", unsafe_allow_html=True)

        # SHOW METRICS
        self.show_metrics(metric=st.session_state['metric'])
        st.markdown(f"<br>", unsafe_allow_html=True)


        # = SHOW EVALUATION SECTION ==================================================

        # get the best variant
        best = st.session_state['best']

        # calculate the bayes test
        self.utils.show_headline(f'Bayesian Test - Beta', 'h2')
        P_ab_thr, loss = self.bayes.p_ab_loss(rvs=[rv_a_beta, rv_b_beta], best=best, thr=1, n_samples=1_000_000)
        self.thr_success = st.slider('Success Threshold [%]', min_value=0.0, max_value=100.0, value=90.0, step=1.0, format='%f')
        self.thr_success = self.thr_success / 100
        self.block_bayes_results(st.session_state['metric'], P_ab_thr, loss, best=best)
        self.utils.show_headline('Insights - Beta', 'h3', color=self.color_default)
        self.plot_insights(rv_a_beta, rv_b_beta, best=best)

        st.markdown(f"<br>", unsafe_allow_html=True)

        self.utils.show_headline(f'Bayesian Test - Gaussian', 'h2')
        P_ab_thr, loss = self.bayes.p_ab_loss(rvs=[rv_a_gauss, rv_b_gauss], best=best, thr=1, n_samples=1_000_000)
        self.block_bayes_results(st.session_state['metric'], P_ab_thr, loss, best=best)
        self.utils.show_headline('Insights - Gauss', 'h3', color=self.color_default)
        self.plot_insights(rv_a_gauss, rv_b_gauss, best=best)


if __name__ == "__main__":
    app = EvaluationAPP()
    app.main()
