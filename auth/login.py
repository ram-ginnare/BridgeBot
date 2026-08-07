import streamlit as st
from auth.auth import authenticate
from auth.session import login

def login_page():

    st.title("BridgeBot Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate(username, password)

        if user:

            login(user)

            st.success("Login Successful")

            st.session_state.user = user["username"]
            st.session_state.role = user["role"]
            st.session_state.department = user["department"]
            st.session_state.team = user["team"]

            st.rerun()

        else:

            st.error("Invalid Username or Password")