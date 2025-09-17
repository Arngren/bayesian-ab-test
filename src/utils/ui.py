import streamlit as st

# Main function
class ABTestUtils:
    def __init__(self) -> None:

        self.color_default = "#EEEEDD"
        self.color_succes = "#88DD88"
        self.color_failure = "#FF8888"

        # st.set_page_config(layout="wide")

        # Custom CSS to inject for changing the background color
        st.markdown("""
            <style>
            .stApp { background-color: #111122; }
            input {
                color: #EEEEDD;
                background-color: #111122;
            }
            </style>
            """, unsafe_allow_html=True)

    def show_headline(self, headline: str, type: str = 'h2', color: str=""):
        color = self.color_default if color == "" else color
        st.markdown(f"<{type} style='text-align: center; margin-bottom: 0px; color: {color}'>{headline}</{type}>", unsafe_allow_html=True)


    def show_value_block(self, top_text: str, bottom_text: str, color: str="", size_value: str="h2", size_txt: str="h5"):
        color = self.color_default if color == "" else color
        st.markdown(f"""
            <{size_value} style='text-align: center; margin-bottom: 0px; color: {color}'>{top_text}</{size_value}>
            <hr style='margin-top: 0px; margin-bottom: 3px;'>
            <{size_txt} style='text-align: center; margin-top: 0px;'>{bottom_text}</{size_txt}>""",
            unsafe_allow_html=True)
        
    def show_value_conf_block(self, top_text: str, middle_txt: str, bottom_text: str, color: str="", size_value: str="h2", size_txt: str="h5"):
        color = self.color_default if color == "" else color
        st.markdown(f"""
            <{size_value} style='text-align: center; margin-bottom: 0px; color: {color}'>{top_text}</{size_value}>
            <{size_txt} style='text-align: center; margin-top: 0px; margin-bottom: 0px; color: {color}'>{middle_txt}</{size_txt}>
            <hr style='margin-top: 0px; margin-bottom: 3px;'>
            <{size_txt} style='text-align: center; margin-top: 0px;'>{bottom_text}</{size_txt}>""",
            unsafe_allow_html=True)


    def show_dual_block(self, top_text_1: str="", color1: str="", top_text_2: str="", color2: str="", bottom_text: str=""):
        # Use default colors if none are specified
        color1 = self.color_default if color1 == "" else color1
        color2 = self.color_default if color2 == "" else color2
        st.markdown(f"""
            <h1 style='text-align: center; margin-bottom: 0px;'>
                <span style='color: {color1};'>{top_text_1}</span> |  
                <span style='color: {color2};'>{top_text_2}</span>
            </h1>
            <hr style='margin-top: 0px; margin-bottom: 0px;'>
            <h4 style='text-align: center; margin-top: 0px;'>{bottom_text}</h4>
            """, unsafe_allow_html=True)


class ShowButtons():
    """ show buttons for selecting the type of test"""

    def __init__(self):
        self.utils = ABTestUtils()

    # Define a function to create a centered button
    def create_centered_button(self, label):
        """Create a centered button.
        
        Args:
            label (str): The label to display on the button.

        Returns:
            bool: True if the button is clicked, False otherwise.
        """
        button_style = """
        <style>
            div.stButton > button:first-child {
                display: block;
                margin: 0 auto;
            }
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)
        return st.button(label)

    def campaign_type(self):
        """Show the buttons for selecting the type of campaign"""

        self.utils.show_headline("Select Campaign Type")

        col_1, col_2 = st.columns(2)
        with col_1:
            if self.create_centered_button('E-Mails'):
                st.session_state['campaign_type'] = 'E-Mails'
        with col_2:
            if self.create_centered_button('Paid Media'):
                st.session_state['campaign_type'] = 'Paid Media'

        col_1, col_2= st.columns(2)
        with col_1:
            if st.session_state['campaign_type'] == 'E-Mails':
                st.markdown(f"<hr style='height: 3px; background-color:red; margin-top: 0px; margin-bottom: 0px;'>", unsafe_allow_html=True)
        with col_2:
            if st.session_state['campaign_type'] == 'Paid Media':
                st.markdown(f"<hr style='height: 3px; background-color:red; margin-top: 0px; margin-bottom: 0px;'>", unsafe_allow_html=True)


    def test_type(self):
        """Show the buttons for selecting the type of test to perform."""

        self.utils.show_headline("Choose test")

        col_1, col_2 = st.columns(2)
        with col_1:
            if self.create_centered_button('Hypothesis Test'):
                st.session_state['test_type'] = 'Hypothesis Test'
        with col_2:
            if self.create_centered_button('Bayesian Test'):
                st.session_state['test_type'] = 'Bayesian Test'


        col_1, col_2= st.columns(2)
        with col_1:
            if st.session_state['test_type'] == 'Hypothesis Test':
                st.markdown(f"<hr style='height: 3px; background-color:red; margin-top: 0px; margin-bottom: 0px;'>", unsafe_allow_html=True)
        with col_2:
            if st.session_state['test_type'] == 'Bayesian Test':
                st.markdown(f"<hr style='height: 3px; background-color:red; margin-top: 0px; margin-bottom: 0px;'>", unsafe_allow_html=True)


    def metric(self):
        """Show the buttons for selecting the metric to evaluate."""

        col_1, col_2, col_3, col_4 = st.columns(4)
        with col_1:
            if self.create_centered_button('Click - Through - Rate (CTR)'):
                st.session_state['metric'] = 'CTR'
                st.session_state['best'] = 'max'
        with col_2:
            if self.create_centered_button('Conversion - Rate (CVR)'):
                st.session_state['metric'] = 'CVR'
                st.session_state['best'] = 'max'
        with col_3:
            if self.create_centered_button('Cost - per - Click (CpC)'):
                st.session_state['metric'] = 'CpC'
                st.session_state['best'] = 'min'
        with col_4:
            if self.create_centered_button('Cost - per - Acquisition (CpA)'):
                st.session_state['metric'] = 'CpA'
                st.session_state['best'] = 'min'

        col_1, col_2, col_3, col_4 = st.columns(4)
        with col_1:
            if st.session_state['metric'] == 'CTR':       
                st.markdown(f"<hr style='height: 3px; background-color:red; margin-top: 0px; margin-bottom: 0px;'>", unsafe_allow_html=True)
        with col_2:
            if st.session_state['metric'] == 'CVR':
                st.markdown(f"<hr style='height: 3px; background-color:red; margin-top: 0px; margin-bottom: 0px;'>", unsafe_allow_html=True)
        with col_3:
            if st.session_state['metric'] == 'CpC':
                st.markdown(f"<hr style='height: 3px; background-color:red; margin-top: 0px; margin-bottom: 0px;'>", unsafe_allow_html=True)
        with col_4:
            if st.session_state['metric'] == 'CpA':
                st.markdown(f"<hr style='height: 3px; background-color:red; margin-top: 0px; margin-bottom: 0px;'>", unsafe_allow_html=True)



def show_hypothesis_input_fields():
    # dropdown to select specific test
    options = ['Chi-square Test', 't-test']
    hypo_test_type = st.selectbox('Select type of test', options)        
    # create two html blocks side by side where data is entered for control and test group
    side_options = ['Single Sided', 'Double Sided']
    col1, col2, col3 = st.columns(3)
    with col1:
        significance_level = st.number_input('Significance level - alpha [%]', min_value=0.0, max_value=None, value=5.0, step=1.0, format='%.1f')
    with col2:
        power = st.number_input('Power - beta [%]', min_value=0.0, max_value=None, value=80.0, step=1.0, format='%.1f')
    with col3:
        alpha_sided_test = st.selectbox('Select single- or double sided', side_options, key='alpha_sided_test')
        alpha_sided_test = 'single' if alpha_sided_test == 'Single Sided' else 'double'
        # beta_sided_test = st.selectbox('Select single- or double sided', side_options, key='beta_sided_test')
        # beta_sided_test = 'single' if beta_sided_test == 'Single Sided' else 'double'
    beta_sided_test = 'single'

    return hypo_test_type, significance_level, power, alpha_sided_test, beta_sided_test


if __name__ == "__main__":
    pass
