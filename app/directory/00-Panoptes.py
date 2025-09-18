import streamlit as st
import os
import time
import functions.pathfunc as pf

from functions.logging import logging

st.set_page_config(layout = 'wide')

logging(st.secrets["secret1"], st.secrets["secret2"])


"# Panoptes"
'### :material/candlestick_chart: A technical analysis dashboard'

img_path = os.path.join(pf.get_path_app(), "img/panoptes.jpg")
st.write(os.listdir("/app"))
st.image(img_path, width="stretch")
st.caption('For educational only, not financial advise')

''
notepad_path = os.path.join( pf.get_path_data(), "NOTEPAD.txt" )
notepad = open(notepad_path, "r")

if "notepad" not in st.session_state :
    text = notepad.read()
    notepad.close()
    st.session_state.notepad = text



input_notepad = st.text_area(':material/note_add: Your last notes', st.session_state.notepad, height=500).replace("/","").replace(":","").replace("'","").replace("FROM","")

if st.button("Save notes", type="primary") :
    with open(notepad_path, 'w') as f:
        f.write(input_notepad).replace("/","").replace(":","").replace("'","").replace("FROM","")
    del st.session_state.notepad
    st.success("Notepad saved! The app will rerun in 3 sec.")
    st.toast("Notepad draft saved.", icon=":material/check_small:")
    time.sleep(3)
