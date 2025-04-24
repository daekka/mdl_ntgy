"""MENÚ"""

import streamlit as st

st.set_page_config(
    page_title="Utilidades MDL-NTGY",
    page_icon="📊",
    layout="wide",
)

# CSS personalizado para cambiar ancho del sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 275px !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 275px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Bienvenido a Utilidades MDL-NTGY 🧮")

