from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    # Run Streamlit inside Flask
    cmd = "streamlit run main.py --server.port 8501 --server.headless true"
    subprocess.Popen(cmd, shell=True)
    return "Streamlit app is running on port 8501"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
