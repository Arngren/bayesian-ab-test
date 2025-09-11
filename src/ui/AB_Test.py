# How-to Run
# streamlit run streamlit_app.py
# docker build -t ab-test-app:latest .
# docker run -p 8080:8080 ab-test-app:latest

# import from ab_test in previous directory
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('src')

import streamlit as st

from src.utils.ui import ABTestUtils
from utils.graph import Visualisation

# Main function
class ABTestAPP:
    def __init__(self) -> None:

        self.plot = Visualisation() # (renderer="vscode")
        self.plot_width = 700

        self.color_default = "#EEEEDD"
        self.color_succes = "#88DD88"
        self.color_failure = "#FF8888"

        self.utils = ABTestUtils()

        # st.set_page_config(layout="wide")

        # Custom CSS to inject for changing the background color
        st.markdown("""
            <style>
            .stApp { background-color: #111122; }
            .main {
                max-width: 900px;
                margin: 0 auto;
            }
            input {
                color: #EEEEDD;
                background-color: #111122;
                # font-size: 10rem !important;
            }
            button {
                display: block;
                margin: 0 auto;
            }
            </style>
            """, unsafe_allow_html=True)

    ############################################################################################
    def main(self):
        """Run this function to display the Streamlit app."""

        self.utils.show_headline('BAYESIAN A/B TEST (BETA)', 'h1')
        self.utils.show_headline("RSSS'25 Vienna", 'h4')
        
        st.markdown("""<br><br>""", unsafe_allow_html=True)

        # add simple description as text field
        st.markdown("""
            <div style='text-align: left;'>            
            This app is Bayesian A/B test calculator for VML MAP team. It has two parts:
            <ul>
                <li><b>Experiment Design</b>: Calculate the sample size needed for a test</li>
                <li><b>Evaluation</b>: Evalaute the test results of a test</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""<hr>""", unsafe_allow_html=True)

        # streamlit link to email adress centered in page
        st.markdown("""
            <div style='text-align: center; margin-bottom: 0px;'>
            This app is a work-in-progress and is intended to be used for internal purposes only.
            For questions or improvements, reach out to
            <a href="mailto:morten.arngren@wundermanthompson.com">Morten Arngren</a>
            </div>
            """, unsafe_allow_html=True)




if __name__ == "__main__":
    abtest_app = ABTestAPP()
    abtest_app.main()
    