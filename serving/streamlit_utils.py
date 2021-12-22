import streamlit as st

@st.cache(allow_output_mutation=True)
def button_states():
    return {"pressed": None}

press_button = st.button("Press it Now!")
is_pressed = button_states()  # gets our cached dictionary

if press_button:
    # any changes need to be performed in place
    is_pressed.update({"pressed": True})

if is_pressed["pressed"]:  # saved between sessions
    th = st.number_input("Please enter the values from 0 - 10")