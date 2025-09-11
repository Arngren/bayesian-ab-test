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
from src.utils.ui import ABTestUtils, show_hypothesis_input_fields, ShowButtons
from scipy.stats import beta, gamma

import plotly
import plotly.graph_objects as go
from ab_test.graph import Visualisation

# Main function
class EvaluationAPP:
    def __init__(self) -> None:
        st.set_page_config(layout="wide")

        self.hypo = Hypothesis_AB_Test()
        self.bayes = Bayesian_AB_Test()

        self.thr_success = 0.7

        self.plot = Visualisation() # (renderer="vscode")
        plotly.io.json.config.default_engine = 'orjson'
        self.plot_width = 1000

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

        if P_ab_thr[0] > self.thr_success:
            # st.markdown(f"<br>", unsafe_allow_html=True)
            self.utils.show_headline('Variant A is the winner', 'h1', color=self.color_succes)
        if P_ab_thr[1] > self.thr_success:
            # st.markdown(f"<br>", unsafe_allow_html=True)
            self.utils.show_headline('Variant B is the winner', 'h1', color=self.color_succes)
        if P_ab_thr[0] <= self.thr_success and P_ab_thr[1] <= self.thr_success:
            # st.markdown(f"<br>", unsafe_allow_html=True)
            self.utils.show_headline('No clear winner', 'h1', color=self.color_failure)
        st.markdown(f"<br>", unsafe_allow_html=True)



    def show_metrics(self, metric, hpdr_a, hpdr_b):
        """Show the metrics in a block
        
        Args:
            metric (str): The metric to display.
            hpdr_a (List): Highest Posterior Density Region for A.
            hpdr_b (List): Highest Posterior Density Region for B.
        """
        # calculate metrics
        col_1, col_2 = st.columns(2)
        with col_1:
            if metric == 'CTR':
                self.ctr_A = self.click_A / self.impr_A if self.impr_A > 0 else 0
                # self.utils.show_value_block(f"{self.ctr_A*100:.1f}%", "Avg. CTR A", color=self.color_default)
                txt = f"({hpdr_a[0]*100:.1f}% - {hpdr_a[1]*100:.1f}%)<br>Avg. CTR A"
                self.utils.show_value_block(f"{self.ctr_A*100:.1f}%", txt, color=self.color_default)
            if metric == 'CVR':
                self.cvr_A = self.conv_A / self.click_A if self.click_A > 0 else 0
                # self.utils.show_value_block(f"{self.cvr_A*100:.1f}%", "Avg. CVR A", color=self.color_default)
                txt = f"({hpdr_a[0]*100:.1f}% - {hpdr_a[1]*100:.1f}%)<br>Avg. CVR A"
                self.utils.show_value_block(f"{self.cvr_A*100:.1f}%", txt, color=self.color_default)                
            if metric == 'CpC':
                self.cpc_A = self.cost_A / self.click_A if self.click_A > 0 else 0
                self.utils.show_value_block(f"${self.cpc_A:.2f}", "Avg. CpC A", color=self.color_default)
            if metric == 'CpA':
                self.cpa_A = self.cost_A / self.conv_A if self.conv_A > 0 else 0
                self.utils.show_value_block(f"${self.cpa_A:.2f}", "Avg. CpA A", color=self.color_default)
        with col_2:
            if metric == 'CTR':
                self.ctr_B = self.click_B / self.impr_B if self.impr_B > 0 else 0
                # self.utils.show_value_block(f"{self.ctr_B*100:.1f}%", "Avg. CTR B", color=self.color_default)
                txt = f"({hpdr_b[0]*100:.1f}% - {hpdr_b[1]*100:.1f}%)<br>Avg. CTR B"
                self.utils.show_value_block(f"{self.ctr_B*100:.1f}%", txt, color=self.color_default)                
            if metric == 'CVR':
                self.cvr_B = self.conv_B / self.click_B if self.click_B > 0 else 0
                # self.utils.show_value_block(f"{self.cvr_B*100:.1f}%", "Avg. CVR B", color=self.color_default)
                txt = f"({hpdr_b[0]*100:.1f}% - {hpdr_b[1]*100:.1f}%)<br>Avg. CVR A"
                self.utils.show_value_block(f"{self.cvr_B*100:.1f}%", txt, color=self.color_default)  
            if metric == 'CpC':
                self.cpc_B = self.cost_B / self.click_B if self.click_B > 0 else 0
                self.utils.show_value_block(f"${self.cpc_B:.2f}", "Avg. CpC B", color=self.color_default)
            if metric == 'CpA':
                self.cpa_B = self.cost_B / self.conv_B if self.conv_B > 0 else 0
                self.utils.show_value_block(f"${self.cpa_B:.2f}", "Avg. CpA B", color=self.color_default)

        st.markdown(f"<br>", unsafe_allow_html=True)


    ############################################################################################
    def main(self):
        """Run this function to display the Streamlit app."""

        # Set Streamlit app title
        self.utils.show_headline('A/B TEST EVALUATION (BETA)', 'h1')

        # create two html blocks side by side where data is entered for control and test group
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h4 style='text-align: center; margin-bottom: 0px; color: {self.color_default}'>Variant A</h4>", unsafe_allow_html=True)
            self.impr_A = st.number_input('# Impressions', min_value=0, max_value=None, value=1000, step=1, format='%d', key='impr_A')
            self.click_A = st.number_input('# Clicks', min_value=0, max_value=None, value=100, step=1, format='%d')
            self.conv_A = st.number_input('# Conversions', min_value=0, max_value=None, value=2, step=1, format='%d')
            self.cpm_A = st.number_input('Cost-per-Mill (CPM)', min_value=0, max_value=None, value=11, step=1, format='%d')
        with col2:
            st.markdown(f"<h4 style='text-align: center; margin-bottom: 0px; color: {self.color_default}'>Variant B</h4>", unsafe_allow_html=True)
            self.impr_B = st.number_input('# Impressions', min_value=0, max_value=None, value=1000, step=1, format='%d', key='impr_B')
            self.click_B = st.number_input('# Clicks', min_value=0, max_value=None, value=120, step=1, format='%d')
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
            rv_a, rv_b = beta(self.click_A+1, self.impr_A-self.click_A), beta(self.click_B+1, self.impr_B-self.click_B)
        if st.session_state['metric'] == 'CVR':
            n_success_a, n_trails_a = self.conv_A, self.click_A
            n_success_b, n_trails_b =self.conv_B, self.click_B
            perf_A, perf_B = self.conv_A / self.click_A, self.conv_B / self.click_B
            rv_a, rv_b = beta(self.conv_A+1, self.click_A-self.conv_A), beta(self.conv_B+1, self.click_B-self.conv_B)
        if st.session_state['metric'] == 'CpC':
            n_success_a, n_trails_a = 0, 0
            n_success_b, n_trails_b =0, 0
            perf_A, perf_B = self.cost_A / self.click_A, self.cost_B / self.click_B
            rv_a, rv_b = gamma(a=self.cost_A+1, scale=1/self.click_A), gamma(a=self.cost_B+1, scale=1/self.click_B)
        if st.session_state['metric'] == 'CpA':
            n_success_a, n_trails_a = 0, 0
            n_success_b, n_trails_b =0,0
            perf_A, perf_B = self.cost_A / self.conv_A, self.cost_B / self.conv_B
            rv_a, rv_b = gamma(a=self.cost_A+1, scale=1/self.conv_A), gamma(a=self.cost_B+1, scale=1/self.conv_B)

        # = METRICS =====================================================================
        # calcualte Highest Posterior Density Region (HPDR) for both variants
        hpdr_a = self.bayes.hpdr(rv=rv_a, thr=0.95)
        hpdr_b = self.bayes.hpdr(rv=rv_b, thr=0.95)
        self.show_metrics(metric=st.session_state['metric'], hpdr_a=hpdr_a, hpdr_b=hpdr_b)
        st.markdown(f"<br>", unsafe_allow_html=True)


        # = TEST ANALYSIS ================================================================
        self.show_buttons.test_type()
        test_type = st.session_state['test_type']

        self.utils.show_headline(f'{test_type}', 'h2')
        if test_type == 'Hypothesis Test':
            hypo_test_type, significance_level, power, alpha_sided_test, beta_sided_test = show_hypothesis_input_fields()
        if test_type == 'Bayesian Test':
            if st.session_state['best'] == 'max':
                self.thr_lift = st.slider('Required Lift', min_value=1.0, max_value=2.0, value=1.0, step=0.01, format='%f')
            else:
                self.thr_lift = st.slider('Required Lift', min_value=0.0, max_value=1.0, value=1.0, step=0.01, format='%f')
            self.thr_success = st.number_input('Success Threshold [%]', min_value=0.0, max_value=100.0, value=90.0, step=1.0, format='%f')
            self.thr_success = self.thr_success / 100
            # options = ['Low', 'High']
            # bayesian_precision = st.selectbox('Precision', options)


        # = SHOW EVALUATION SECTION ==================================================
        if test_type == 'Hypothesis Test':

            if hypo_test_type == 'Chi-square Test':
                self.utils.show_headline(f'{hypo_test_type}', 'h2')

                # calculate the sample size
                n_samples_required, Z_a, Z_b = self.hypo.calc_sample_size(test_type=hypo_test_type,
                                                                          p_a=perf_A, p_b=perf_B,
                                                                          alpha=significance_level/100, beta=1-power/100,
                                                                          alpha_side=alpha_sided_test)

                # calculate the p-values for test
                chi2, p, dof, ex = self.hypo.chi2_test(n_success_a=n_success_a, n_trails_a=n_trails_a,
                                                        n_success_b=n_success_b, n_trails_b=n_trails_b)

                col_1, col_2 = st.columns(2)
                with col_1:
                    color=self.color_succes if (n_samples_required<self.impr_A) & (n_samples_required<self.impr_B) else self.color_failure
                    self.utils.show_value_block(f"{n_samples_required:,}", "required samples", color=color)
                with col_2:
                    color=self.color_succes if p <= (significance_level/100) else self.color_failure
                    self.utils.show_value_block(f"{100*p:.2f}%", "p-value", color=color)

                # Debug - display equation and Z-values used...
                st.markdown(f"<br><br>", unsafe_allow_html=True)
                self.utils.show_headline('Sample Size Calculation', 'h5')
                # st.markdown(f"<div style='text-align: center; margin-bottom: 0px; color: {self.color_default}'>Equation:</dic>", unsafe_allow_html=True)
                st.latex(r"n = \frac{{(Z_a + Z_b)^2 \cdot (p_a \cdot (1 - p_a) + p_b \cdot (1 - p_b))}}{{(p_b - p_a)^2}}, \quad where")
                st.latex(f"Z_a = {Z_a:.3f} \quad \land \quad  Z_b = {Z_b:.3f}")
                # col1, col2 = st.columns(2)
                # with col1:
                #     # st.markdown(f"<div style='text-align: center; margin-bottom: 0px; color: {self.color_default}'>", unsafe_allow_html=True)
                #     # st.markdown(f"$$n = \\frac{{(Z_a + Z_b)^2 \\cdot (p_a \\cdot (1 - p_a) + p_b \\cdot (1 - p_b))}}{{(p_b - p_a)^2}}$$", unsafe_allow_html=True)
                #     # st.markdown(f"</div>", unsafe_allow_html=True)
                #     st.latex(r"n = \frac{{(Z_a + Z_b)^2 \cdot (p_a \cdot (1 - p_a) + p_b \cdot (1 - p_b))}}{{(p_b - p_a)^2}}")

                # with col2:
                #     st.markdown(f"$$Z_a = {Z_a:.3f}$$<br>$$Z_b = {Z_b:.3f}$$", unsafe_allow_html=True)

                st.markdown(f"<br>", unsafe_allow_html=True)

            if hypo_test_type == 't-test':
                self.utils.show_headline(f'{hypo_test_type}', 'h2')
                self.utils.show_headline('...to be implemented', 'h4')


        if test_type == 'Bayesian Test':

            # get the best variant
            best = st.session_state['best']

            # calculate the bayes test
            P_ab_thr, loss = self.bayes.p_ab_loss(rvs=[rv_a, rv_b], best=best, thr=self.thr_lift, n_samples=1_000_000)
            self.block_bayes_results(st.session_state['metric'], P_ab_thr, loss, best=best)

            st.markdown(f"<br>", unsafe_allow_html=True)
            self.utils.show_headline('Insights', 'h1', color=self.color_default)

            # PLOTS
            n_samples = 1_000_000

            samples_a = rv_a.rvs(size=n_samples)
            samples_b = rv_b.rvs(size=n_samples)
            # ratio
            ratio = samples_b / samples_a if best=='max' else samples_a / samples_b
            #95% quantile
            x_min, x_max = np.quantile(ratio, 1e-3), np.quantile(ratio, 1-1e-3)
            hist, bins = np.histogram(ratio, bins=np.linspace(x_min, x_max ,1001))
            id1, id2 = bins<=self.thr_lift, bins>self.thr_lift

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

            p_data = [self.plot.plot(x=bins[id1], y=hist[id1[:-1]]/sum(hist), color=0, opacity=0.7, name='A', showlegend=True),
                        self.plot.plot(x=bins[id2], y=hist[id2[:-1]]/sum(hist), color=1, opacity=0.7, name='B', showlegend=True) ]
            layout = self.plot.layout(title='Distribution of P(B>A)', x_label='lift ratio', y_label='', theme='dark', width=self.plot_width, height=300)
            fig = go.Figure(data=p_data, layout=layout)
            st.plotly_chart(fig)

            if best == 'max':
                y1 = 1-hist.cumsum()[id1[:-1]]/sum(hist)
                y2 = 1-hist.cumsum()[id2[:-1]]/sum(hist)
            else:
                y1 = hist.cumsum()[id1[:-1]]/sum(hist)
                y2 = hist.cumsum()[id2[:-1]]/sum(hist)
            p_data = [self.plot.plot(x=bins[id1], y=y1, color=0, opacity=0.7, name='A', showlegend=True),
                        self.plot.plot(x=bins[id2], y=y2, color=1, opacity=0.7, name='B', showlegend=True) ]
            layout = self.plot.layout(title='Cumulative distribution of P(B>A)', x_label='lift ratio', y_label='', theme='dark', width=self.plot_width, height=300)
            fig = go.Figure(data=p_data, layout=layout)
            st.plotly_chart(fig)


if __name__ == "__main__":
    app = EvaluationAPP()
    app.main()
