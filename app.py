import streamlit as st

st.set_page_config(
    page_title="equity_platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("📊 equity_platform")
st.sidebar.markdown("---")
st.sidebar.markdown("**Navegação**")

st.title("📊 Capital Flow Intelligence Platform")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Tema Dominante", "AI_Compute", "LEADING")

with col2:
    st.metric("Tema Acelerando", "Metals", "+↑")

with col3:
    st.metric("Dispersão", "CONCENTRADO", "⚠️")

st.markdown("---")
st.info("🚧 Selecione uma página na barra lateral para começar.")
