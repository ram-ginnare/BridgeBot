import streamlit as st


def login(user):

    st.session_state.logged_in = True

    st.session_state.user = user["username"]

    st.session_state.role = user["role"]


def logout():

    st.session_state.clear()

def is_logged_in():

    return st.session_state.get("logged_in", False)