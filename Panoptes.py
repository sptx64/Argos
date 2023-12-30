import streamlit as st
import os

st.set_page_config(layout = 'wide')

from app.logging import logging
logging(st.secrets["secret1"], st.secrets["secret2"])


"# Panoptès"
'### 📊 A technical analysis dashboard'
st.image("img/panoptes.jpg")

st.caption('For educational only, not financial advise')

''
notepad_path = "temp/NOTEPAD.txt"
notepad = open(notepad_path, "r")

if "notepad" not in st.session_state :
    text = notepad.read()
    notepad.close()
    st.session_state.notepad = text



input_notepad = st.text_area('Your last notes', st.session_state.notepad, height=500)

save_notes = st.button("Save notes")
if save_notes :
    with open(notepad_path, 'w') as f:
        f.write(input_notepad)
    del st.session_state.notepad
    st.success("Notepad saved")
    st.balloons()



""
"---"
""

with st.expander("Theming") :
    "# Theming"
    theme = st.text_area("Paste the streamlit theme here")
    go = st.button("Actualiser le thème")
    if go :
        if not os.path.exists(".streamlit"):
            os.mkdir(".streamlit")
        if os.path.exists(".streamlit/config.toml") :
            os.remove(".streamlit/config.toml")
        with open(".streamlit/config.toml", "w") as f:
            f.write(theme)
