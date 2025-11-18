import webbrowser
import time

# Wait a moment for Streamlit to fully start
time.sleep(2)

# Open the Streamlit app in the default browser
url = "http://localhost:8501"
webbrowser.open(url)

print(f"Opening Streamlit dashboard at: {url}")
print("If the browser doesn't open automatically, copy and paste the URL above.")
