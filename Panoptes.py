import streamlit as st
import os

st.set_page_config(layout = 'wide')

# st.markdown(
#     """
#     <style>
#     .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
#     .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
#     .viewerBadge_text__1JaDK {
#         display: none;
#     } header { visibility: hidden; }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


from app.logging import logging
logging(st.secrets["secret1"], st.secrets["secret2"])


"# Panoptès"
'### 📊 A technical analysis dashboard'
tab1, tab2 = st.tabs(["Picture", "Preview video"])
with tab1 :
    st.image("img/panoptes.jpg")
with tab2 :
    st.video("img/streamlit-Panoptes-2024-07-06-22-07-47.webm")
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
