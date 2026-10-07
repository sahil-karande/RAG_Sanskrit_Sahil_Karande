import streamlit as st

print("Testing st.container...")
with st.container(border=True):
    st.markdown('<div class="directory-banner">Test Banner</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.button('btn 1')
    with c2:
        st.button('btn 2')
print("st.container syntax verified!")
