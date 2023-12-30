import streamlit as st
import os


def logging(secret1, secret2) :
    if "wait" not in st.session_state :
        st.session_state.wait=False

    if "tries" not in st.session_state :
        st.session_state.tries=0

    if "password" not in st.session_state :
        col1,col2,col3 = st.columns(3)
        password1 = col2.text_input("Insert access password1", type="password")
        password2 = col2.text_input("Insert access password2", type="password")
        st.write(password1)
        st.write(secret1)
        st.write(password1 != secret1)
        st.write(password2 != secret2)
        connect = col2.button("Connect", type="primary", disabled = False if st.session_state.wait==False else True)
        if connect :
            if (password1 != secret1) | (password2 != secret2) :
                st.session_state.tries+=1
                col2.error(f"Try number : {st.session_state.tries}")
                if st.session_state.tries > 3 :
                    st.session_state.wait=True
                    st.rerun()
            else :
                st.session_state.password=True
                st.rerun()
        st.stop()

# logging(secret1,secret2)
