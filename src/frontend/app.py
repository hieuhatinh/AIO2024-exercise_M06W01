import streamlit as st

from components import digit_recognition, cassava_leaf_disease_classification, sentiment_analysis


def app():
    page_names_to_funcs = {
        "MNIST dataset": digit_recognition.run,
        "Cassava Leaf Disease datase": cassava_leaf_disease_classification.run,
        "NTC_SCV dataset": sentiment_analysis.run
    }
    demo_name = st.sidebar.selectbox(
        'Choose a demo', page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()


if __name__ == '__main__':
    app()
