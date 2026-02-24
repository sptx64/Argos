import streamlit as st


# if st.context.theme["type"] == "light" :
#     st.logo("img/Zero dark writing.png", icon_image="img/Zero dark.png", size="large")
# else :
#     st.logo("img/Zero white.png", icon_image="img/Zero white.png", size="large")

pages_dict = {"HOME":[st.Page("directory/00-Panoptes.py", title="Home", icon=":material/home:", default=True)],
                "UPDATE":[ st.Page("directory/01_Hermes.py", title="Hermes", icon=":material/cloud:"),],
                "EXPLORE":[st.Page("directory/02_️Argos.py", title="Argos", icon=":material/ophthalmology:"),
                            st.Page("directory/03_Zeus.py", title="Zeus", icon=":material/bolt:"),
                            st.Page("directory/04_Panel.py", title="Panel", icon=":material/bolt:"),
                            st.Page("directory/05_Corr.py", title="Correlations", icon=":material/bolt:")] }
                            # st.Page("directory/03_Zeus_echarts.py", title="Zeus_ec", icon=":material/bolt:")] }

pg = st.navigation(pages_dict, position="top")
pg.run()


from functions.logging import logging
logging(st.secrets["secret1"], st.secrets["secret2"])
