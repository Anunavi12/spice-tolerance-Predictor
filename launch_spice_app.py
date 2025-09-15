# launch_spice_app.py
import webbrowser
import subprocess
import time

# Open default browser to the Streamlit app URL
webbrowser.open("http://localhost:8501")

# Launch the Streamlit app
subprocess.run(["streamlit", "run", "spice_predictor_streamlit.py"])
