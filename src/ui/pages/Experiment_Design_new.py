# How-to Run
# streamlit run streamlit_app.py
# docker build -t ab-test-app:latest .
# docker run -p 8080:8080 ab-test-app:latest

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
from scipy.stats import beta, gamma

import plotly
import plotly.graph_objects as go
from ab_test.graph import Visualisation

# Main function
class ExperimentDesignAPP:
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
        if 'campaign_type' not in st.session_state:
            st.session_state['campaign_type'] = 'E-Mails'

        if 'test_type' not in st.session_state:
            st.session_state['test_type'] = 'Bayesian Test'
        if 'metric' not in st.session_state:
            st.session_state['metric'] = 'CTR'

        self.utils = ABTestUtils()
        self.show_buttons = ShowButtons()


    def show_metrics(self, Perf_A, Perf_B, effect_size):
        """ Show results block
        
        Args:
            Perf_A (float): performance of variant A
            Perf_B (float): performance of variant B
            effect_size (float): effect size
            n_samples_required_txt (str): number of samples required
        """
        # Show metrics and results...
        col1, col2 = st.columns(2)
        with col1:
            self.utils.show_value_block(f'{Perf_A*100:.2f}% | {Perf_B*100:.2f}%', 'Perf. A | B', color=self.color_default, size_value="h3", size_txt='h6')
        with col2:
            self.utils.show_value_block(f'{effect_size*100:.2f}%', 'Effect size', color=self.color_default, size_value="h3", size_txt='h6')

        st.markdown("<br>", unsafe_allow_html=True)


    def show_results(self, campaign_type, n_samples_bayes, n_samples_hypo, n_impr_per_week=0, txt_n_samples: str=''):
        """ Show results block
        
        Args:
            n_samples_bayes (int): number of samples required for Bayesian approach
            n_samples_hypo (int): number of samples required for Hypothesis approach
            n_impr_per_week (int): number of impressions per week, if applicable
        """
        n_samples_bayes_txt = f'~{n_samples_bayes:,}' if n_samples_bayes > 0 else '> 1 Million'
        n_samples_hypo_txt = f'{n_samples_hypo:,}' if n_samples_hypo > 0 else '> 1 Million'
        n_samples_diff = n_samples_hypo - n_samples_bayes
        n_samples_diff_txt = f'~{n_samples_diff:,}' #  if n_samples_diff > 1e6 else '> 1 Million'
        n_samples_diff_pct = (n_samples_diff / n_samples_hypo) * 100
        n_samples_diff_pct_txt = f'~{n_samples_diff_pct:.1f}%' #  if n_samples_diff > 1e6 else '> 1 Million'

        # Show how many samples is saved from using Bayesian approach
        if campaign_type == 'Paid Media':
            n_weeks = n_samples_bayes / n_impr_per_week
            n_weeks_txt = f'~{int(np.ceil(n_weeks))} weeks'

            col1, col2 = st.columns(2)
            with col1:    
                self.utils.show_value_block(n_samples_bayes_txt, txt_n_samples, color=self.color_succes)
            with col2:
                self.utils.show_value_block(n_weeks_txt, 'Test duration', color=self.color_succes)
        else:
            self.utils.show_value_block(n_samples_bayes_txt, txt_n_samples, color=self.color_succes)
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:    
            self.utils.show_value_block(n_samples_hypo_txt, 'Hypothesis approach<br>(single sided)', color=self.color_failure)
        with col2:
            self.utils.show_value_block(n_samples_diff_txt, 'Saved samples from using Bayesian approach', color=self.color_failure)
        with col3:
            self.utils.show_value_block(n_samples_diff_pct_txt, 'Percentage of saved samples', color=self.color_failure)

        st.markdown("<br>", unsafe_allow_html=True)


    # ==================================================================================
    def hypo_test(self):
        """ Hypothesis Test"""
        
        self.utils.show_headline('Hypothesis Test', 'h2')
        st.markdown("""<hr style='margin-top: 0px; margin-bottom: 0px;'>""", unsafe_allow_html=True)

        # = INPUT ========================================
        hypo_test_type, significance_level, power, alpha_sided_test, beta_sided_test = show_hypothesis_input_fields()

        col1, col2 = st.columns(2)
        with col1:
            Perf_A = st.number_input('Reference Performance [%]', min_value=0.0, max_value=100.0, value=2.0, step=0.1, format='%.1f')
        with col2:
            lift = st.number_input('Lift [%]', min_value=0.0, max_value=None, value=3.0, step=0.1, format='%.1f')

        # convert to percentage
        Perf_A = Perf_A / 100
        significance_level = significance_level/100
        power = power/100

        # calc. performance of B
        Perf_B = Perf_A * (1+lift/100)
        effect_size = (Perf_B - Perf_A)
        # CPC_A = CPM_MEAN_A  / 100 / Perf_A
        # CPC_B = CPM_MEAN_A*(1-lift/100) / 1000 / CTR_A
    #     CPC_B = CPM_MEAN_A / 1000 / CTR_B
        # CPC_B = CPC_A * (1-lift/100)

        # var_a, var_b = 10, 10 # ???
        # n_samples_cpc = ( ( Z_a*np.sqrt(2*var_a) + Z_b*np.sqrt(var_a+var_b))/(CPC_A-CPC_B) ) ** 2
        # # print(f"CPC_A/B: {CPC_A:.3f}/{CPC_B:.3f} - lift: {lift} | {1-lift/100:.2f} - {CPC_B / CPC_A:.4f} | n_samples: {n_samples}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # hypothesis testing - number of samples required
        n_samples_required, Z_a, Z_b = self.hypo.calc_sample_size(test_type=hypo_test_type, p_a=Perf_A, p_b=Perf_B, alpha=significance_level, beta=1-power, alpha_side=alpha_sided_test, beta_side=beta_sided_test)
        n_samples_required_txt = f'~{n_samples_required:,}' if n_samples_required > 0 else '> 1 Million'

        if hypo_test_type == 'Chi-square Test':
            # DISPLAY
            self.utils.show_headline('Chi-square Test', 'h2')
            st.markdown("<br>", unsafe_allow_html=True)
            self.show_results_block(Perf_A, Perf_B, effect_size, n_samples_required_txt)

            # Debug - display equation and Z-values used...
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"$$n = \\frac{{(Z_a + Z_b)^2 \\cdot (p_a \\cdot (1 - p_a) + p_b \\cdot (1 - p_b))}}{{(p_b - p_a)^2}}$$", unsafe_allow_html=True)
            with col2:
                st.markdown(f"$$Z_a = {Z_a:.3f}$$<br>$$Z_b = {Z_b:.3f}$$", unsafe_allow_html=True)

            # = CHI-SQUARE TEST - DECISION ANALYSIS ========================================
            steps = int(n_samples_required*1.1 / 1000)
            impr_list = np.arange(1, n_samples_required*1.1, steps)
            perf_list = []
            for impr in impr_list:
                # clicks
                clicks_a = int(impr * Perf_A)
                clicks_b = int(impr * Perf_B)

                # cost
                # cost_a, scale_a = np.round(impr*CPM_MEAN_A/1000)+1e-6, 1/(clicks_a + 1e-6)
                # cost_b, scale_b = np.round(impr*CPM_MEAN_A/1000)+1e-6, 1/(clicks_b + 1e-6)
                # cost_b, scale_b = np.round(impr*CPM_MEAN_A*(1-lift/100)/1000)+1e-6, 1/(clicks_a + 1e-6)
                # cost_b, scale_b = impr*CPM_MEAN_A/1000, 1/(clicks_b + 1e-6)

                # hypothesis testing
                perf_list += [ 1 - self.hypo.chi2_test(clicks_a, impr, clicks_b, impr)[1] ]
                # PA['hypo']['cpc'] += [ 1 - self.hypo.chi2_test(cost_a, clicks_a, cost_b, clicks_b)[1] ]

            self.plot_decision_analysis(impr_list, perf_list, Perf_A, lift, n_samples_required, 1-significance_level, y_label='1 - p-value', y_range=[0,1])

        if hypo_test_type == 't-test':
            self.utils.show_headline('t-test', 'h4')
            self.utils.show_headline('...to be implemented', 'h4')


    # ==============================================================================================================
    def bayesian_test(self):
        """ Bayesian Test """

        # self.utils.show_headline('Bayesian Test', 'h2')
        # st.markdown("""<hr style='margin-top: 0px; margin-bottom: 0px;'>""", unsafe_allow_html=True)

        # = INPUT ========================================
        self.show_buttons.campaign_type()
        campaign_type = st.session_state['campaign_type']
        self.show_buttons.metric()
        metric = st.session_state['metric']

        # options = ['E-Mails', 'Paid Media']
        # campaign_type = st.selectbox('Select Campaing Type', options)

        if campaign_type == 'E-Mails':
            # options = ['Open-Rate (OR)', 'Click-Through-Rate (CTR)']
            # metric = st.selectbox('Select Metrics family', options)
            # n_impr_per_week = 0
            txt_n_samples = '#Send outs per variant'

        if campaign_type == 'Paid Media':
            # options = ['Click-Through-Rate (CTR)', 'Conversion-Rate (CVR)', 'Cost-per-Click (CpC)', 'Cost-per-Acquisition (CpA)']
            # metric = st.selectbox('Select Metric', options)
            txt_n_samples = '#Unique visitors per variant'

        # create two html blocks side by side where data is entered for control and test group
        col1, col2 = st.columns(2)
        with col1:
            Perf_A = st.number_input('Reference Performance [%]', min_value=0.0, max_value=100.0, value=2.0, step=0.1, format='%.1f')
            threshold = st.number_input('Acceptance Threshold [%]', min_value=0.0, max_value=99.9, value=95.0, step=1.0, format='%.1f')
        with col2:
            lift = st.number_input('Expected Lift [%]', min_value=0.0, max_value=None, value=15.0, step=0.1, format='%.1f')
            if campaign_type == 'Paid Media':
                n_impr_per_week = st.number_input('Expected impressions per week', min_value=0, max_value=None, value=10_000, step=10, format='%d')
            else:
                n_impr_per_week = 0

        options = ['Low', 'Medium', 'High']
        bayesian_precision = st.selectbox('Simulation Precision', options)

        # round lift to 1 decimmal
        lift = np.round(lift, 1)

        # convert threshold for alpha
        alpha = 1 - threshold/100

        # convert to percentage
        Perf_A = Perf_A / 100
        threshold = threshold/100
        # calc. performacne of B
        Perf_B = Perf_A * (1+lift/100)
        effect_size = (Perf_B - Perf_A)

        # = CALCULATIONS ========================================
        # calc. sample size for both Bayes and Hypothesis testing (to illustate the difference)
        # if metric in  ['Open-Rate (OR)', 'Click-Through-Rate (CTR)', 'Conversion-Rate (CVR)']:
        if metric in  ['OR', 'CTR', 'CVR']:
            # Number of required samples - bayesian
            n_samples_required_bayes, impr_list, perf_list = self.bayes.calc_sample_size(Perf_A, lift, threshold=threshold, metric=metric, precision=bayesian_precision)

            # Number of required samples - hypothesis testing - as reference
            n_samples_required_hypo, Z_a, Z_b = self.hypo.calc_sample_size(test_type='Chi-square Test', p_a=Perf_A, p_b=Perf_B, alpha=alpha, beta=1-0.8, alpha_side='single', beta_side='single')

            st.markdown("<br>", unsafe_allow_html=True)
            
            # SHOW METRICS BLOCK
            self.show_metrics(Perf_A, Perf_B, effect_size)

            # SHOW RESULTS BLOCK
            self.show_results(campaign_type, n_samples_required_bayes, n_samples_required_hypo, n_impr_per_week, txt_n_samples)

            # SHOW INSIGHTS BLOCK
            self.plot_decision_analysis(impr_list, perf_list, Perf_A, lift, n_samples_required_bayes, threshold, y_label='P(B>A)', y_range=[0.5,1])

        if metric in  ['Cost-per-Click (CpC)', 'Cost-per-Acquisition (CpA)']:
            st.markdown("Not implemented yet...", unsafe_allow_html=True)



    def plot_decision_analysis(self, impr_list: List[int], perf_list: List[float],
                               Perf_A: float, lift: float,
                               n_samples_required: int, threshold: float,
                               y_label: str, y_range: List[float]=[0,1]):
        """ Plot decision analysis
        
        Args:
            impr_list (List): list of impressions
            perf_list (List): list of performances
            Perf_A (float): performance of variant A
            lift (float): lift
            n_samples_required (int): number of samples required
            threshold (float): threshold
            y_label (str): y-axis label
            y_range (List): y-axis range
        """
        self.utils.show_headline('Insights', 'h1', color=self.color_default)

        n_impr = len(impr_list)
        p_data =  [self.plot.plot(x=impr_list, y=perf_list, color=0, opacity=0.7, name=f'Lift: {lift}%', showlegend=True),
                   self.plot.plot(x=impr_list, y=n_impr*[threshold], color='#FF88FF', opacity=1, fill=None, linewidth=3, name=f'>{threshold}', showlegend=True),
                   self.plot.plot(x=100*[n_samples_required], y=list(range(0,100,1)), color='#DDDDDD', opacity=1, fill=None, name='Required')
                   ]
        layout = self.plot.layout(title=f'Decision Analysis for Perf. = {Perf_A*100:.2f}%', x_label='impressions [#]', y_label=y_label,
                                  theme='dark', width=self.plot_width, height=400)
        layout['yaxis']['range'] = y_range
        fig = go.Figure(data=p_data, layout=layout)
        st.plotly_chart(fig)


    ############################################################################################
    def main(self):
        """Run this function to display the Streamlit app."""
        
        self.utils.show_headline('A/B TEST EXPERIMENT DESIGN (BETA)', 'h1')

        # make list of options for the dropdown
        # options = ['Bayesian Test', 'Hypothesis Test']
        # test_type = st.selectbox('Select Test', options)

        # self.show_buttons.test_type()

        st.markdown("<br>", unsafe_allow_html=True)

        # if st.session_state['test_type'] == 'Bayesian Test':
        #     self.bayesian_test()

        # if st.session_state['test_type'] == 'Hypothesis Test':
        #     self.hypo_test()

        self.bayesian_test()


if __name__ == "__main__":
    app = ExperimentDesignAPP()
    app.main()
