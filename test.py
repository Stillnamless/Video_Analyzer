# contents of this file were used for testing a particular part, it has nothing to do with the main code
# # # # import speech_recognition as sr

# # # # recognizer = sr.Recognizer()
# # # # mic = sr.Microphone()

# # # # print("Say something...")

# # # # with mic as source:
# # # #     recognizer.adjust_for_ambient_noise(source)
# # # #     audio = recognizer.listen(source)

# # # # print("Got audio! Trying to recognize...")

# # # # try:
# # # #     text = recognizer.recognize_google(audio)
# # # #     print("You said:", text)
# # # # except sr.UnknownValueError:
# # # #     print("Could not understand the audio.")
# # # # except sr.RequestError as e:
# # # #     print("Could not request results; {0}".format(e))

# # # # import speech_recognition as sr

# # # # print(sr.Microphone.list_microphone_names())

# # # # import speech_recognition as sr

# # # # print(sr.Microphone.list_microphone_names())

# # # # import speech_recognition as sr

# # # # for index, name in enumerate(sr.Microphone.list_microphone_names()):
# # # #     print(f"{index}: {name}")


# # # # import speech_recognition as sr

# # # # recognizer = sr.Recognizer()

# # # # with sr.Microphone() as source:
# # # #     print("Say something...")
# # # #     audio = recognizer.listen(source)

# # # # try:
# # # #     print("You said: " + recognizer.recognize_google(audio))
# # # # except sr.UnknownValueError:
# # # #     print("Could not understand audio")
# # # # except sr.RequestError as e:
# # # #     print(f"Could not request results; {e}")


# # # import streamlit as st
# # # import requests
# # # import subprocess
# # # import os
# # # import re

# # # # 1. Map software names to trusted download URLs and installer filenames
# # # SOFTWARE_CATALOG = {
# # #     "vlc media player": {
# # #         "url": "https://get.videolan.org/vlc/3.0.18/win64/vlc-3.0.18-win64.exe",
# # #         "installer": "vlc_installer.exe",
# # #         "silent_flag": "/S"
# # #     },
# # #         "r":{
# # #             "url":"https://cran.icts.res.in/",
# # #             "installer":"r.exe",
            
# # #         }
# # #     }
# # #     # Add more software here as needed


# # # def parse_software_name(command):
# # #     # Simple regex to extract software name after 'install'
# # #     match = re.search(r'install (.+)', command, re.IGNORECASE)
# # #     if match:
# # #         return match.group(1).strip().lower()
# # #     return None

# # # def download_and_install_software(software_key):
# # #     info = get_software_info(software_key)
# # #     if not info:
# # #         st.error("Software not found in catalog.")
# # #         return
    
# # #     url = info["url"]
# # #     installer = info["installer"]
# # #     silent_flag = info.get("silent_flag", "")

# # #     # Download
# # #     st.info(f"Downloading {software_key.title()}...")
# # #     try:
# # #         with requests.get(url, stream=True) as r:
# # #             r.raise_for_status()
# # #             with open(installer, 'wb') as f:
# # #                 for chunk in r.iter_content(chunk_size=8192):
# # #                     f.write(chunk)
# # #         st.success("Download complete.")
# # #     except Exception as e:
# # #         st.error(f"Download failed: {e}")
# # #         return

# # #     # Install
# # #     st.info("Installing... (this may take a moment)")
# # #     try:
# # #         subprocess.run([installer, silent_flag], check=True)
# # #         st.success(f"{software_key.title()} installed successfully!")
# # #     except Exception as e:
# # #         st.error(f"Installation failed: {e}")
# # #     finally:
# # #         # Clean up installer file
# # #         if os.path.exists(installer):
# # #             os.remove(installer)
# # #             st.info("Installer removed.")

# # # # Streamlit UI
# # # st.title("Software Installer Chatbot")

# # # user_input = st.text_input("Type your command (e.g., install vlc media player):")

# # # if user_input:
# # #     software_name = parse_software_name(user_input)
# # #     if software_name and software_name in SOFTWARE_CATALOG:
# # #         download_and_install_software(software_name)
# # #     elif user_input.lower().startswith("install "):
# # #         st.warning("Sorry, I don't recognize that software yet.")
# # #     else:
# # #         st.write("You said:", user_input)

# # # import streamlit as st

# # # # Add vertical space (adjust the range for more/less space)
# # # for _ in range(10):
# # #     st.write("")

# # # # Center horizontally with columns
# # # col1, col2, col3 = st.columns([1,2,1])
# # # with col2:
# # #     user_input = st.text_input("Type your search here")


# # import streamlit as st
# # import subprocess
# # import os
# # import json

# # def get_appx_packages():
# #     result = subprocess.run([
# #         "powershell", "-Command",
# #         "Get-AppxPackage -AllUsers | Select Name, InstallLocation | ConvertTo-Json"
# #     ], capture_output=True, text=True)
# #     try:
# #         return json.loads(result.stdout)
# #     except Exception:
# #         return []

# # def get_win32_apps():
# #     result = subprocess.run([
# #         "powershell", "-Command",
# #         "Get-WmiObject -Class Win32_Product | Select Name, InstallLocation | ConvertTo-Json"
# #     ], capture_output=True, text=True)
# #     try:
# #         return json.loads(result.stdout)
# #     except Exception:
# #         return []

# # def get_folder_size(path):
# #     total_size = 0
# #     for dirpath, dirnames, filenames in os.walk(path):
# #         for f in filenames:
# #             fp = os.path.join(dirpath, f)
# #             try:
# #                 if os.path.isfile(fp):
# #                     total_size += os.path.getsize(fp)
# #             except Exception:
# #                 pass
# #     return total_size

# # st.title("System vs Downloaded Apps and Storage Usage")

# # if st.button("Scan My Apps"):
# #     with st.spinner("Scanning system apps..."):
# #         system_apps = get_appx_packages()
# #     with st.spinner("Scanning downloaded apps..."):
# #         downloaded_apps = get_win32_apps()

# #     system_total = 0
# #     downloaded_total = 0
# #     system_list = []
# #     downloaded_list = []

# #     with st.spinner("Calculating storage usage..."):
# #         for app in system_apps:
# #             path = app.get("InstallLocation")
# #             name = app.get("Name")
# #             if path and os.path.exists(path):
# #                 size = get_folder_size(path)
# #                 system_total += size
# #                 system_list.append({"App": name, "Size (MB)": f"{size/1e6:.2f}"})

# #         for app in downloaded_apps:
# #             path = app.get("InstallLocation")
# #             name = app.get("Name")
# #             if path and os.path.exists(path):
# #                 size = get_folder_size(path)
# #                 downloaded_total += size
# #                 downloaded_list.append({"App": name, "Size (MB)": f"{size/1e6:.2f}"})

# #     st.subheader("Summary")
# #     st.markdown(f"**System Apps:** {len(system_list)} apps, **{system_total/1e9:.2f} GB**")
# #     st.markdown(f"**Downloaded Apps:** {len(downloaded_list)} apps, **{downloaded_total/1e9:.2f} GB**")

# #     with st.expander("See details for System Apps"):
# #         st.table(system_list)
# #     with st.expander("See details for Downloaded Apps"):
# #         st.table(downloaded_list)

# import streamlit as st
# import tempfile
# import PyPDF2
# import speech_recognition as sr
# import pyttsx3
# import subprocess

# # Initialize TTS engine once
# engine = pyttsx3.init()

# def call_gemini_api(prompt):
#     return f"Gemini says: {prompt}"

# def extract_text_from_pdf(pdf_file):
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text() + "\n"
#     return text

# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Please speak now.")
#         audio = recognizer.listen(source, phrase_time_limit=5)
#     try:
#         text = recognizer.recognize_google(audio)
#         return text
#     except:
#         return None

# def install_software(exe_path):
#     try:
#         subprocess.Popen(exe_path, shell=True)
#         return "Installation started."
#     except Exception as e:
#         return f"Error: {str(e)}"

# st.title("Chatbot with Auto Listen & Auto Speak")

# # State to store input text
# if "user_input" not in st.session_state:
#     st.session_state.user_input = ""

# # Auto listen on first load
# if st.session_state.user_input == "":
#     spoken_text = recognize_speech()
#     if spoken_text:
#         st.session_state.user_input = spoken_text
#         st.success(f"You said: {spoken_text}")
#     else:
#         st.warning("Couldn't understand speech or microphone not available.")

# # Text input to modify or enter manually
# user_input = st.text_input("Your message:", st.session_state.user_input)

# # PDF upload & extraction
# uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
# if uploaded_pdf:
#     with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
#         temp_pdf.write(uploaded_pdf.read())
#         extracted_text = extract_text_from_pdf(temp_pdf.name)
#     st.text_area("Extracted PDF Text", extracted_text, height=200)

# # Software installation input
# exe_file_path = st.text_input("Path to .exe for installation:")

# if st.button("Install Software"):
#     if exe_file_path:
#         install_message = install_software(exe_file_path)
#         st.info(install_message)
#     else:
#         st.error("Please enter .exe path.")

# # If user input available, get bot response and auto speak it
# if user_input:
#     response = call_gemini_api(user_input)
#     st.markdown(f"**Bot:** {response}")
    
#     # TTS auto speak
#     engine.say(response)
#     engine.runAndWait()
    
#     # Clear user input after response so next time auto listen triggers again
#     st.session_state.user_input = ""

import streamlit as st
import speech_recognition as sr
import pyttsx3

# === Text-to-Speech ===
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # You can customize this
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"TTS Error: {e}")

# === Voice Input from Microphone ===
def listen_from_mic():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.success("🎧 Voice captured. Processing...")
            return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("Listening timed out.")
    except sr.UnknownValueError:
        st.warning("Could not understand your voice.")
    except sr.RequestError as e:
        st.error(f"Speech Recognition API error: {e}")
    except Exception as e:
        st.error(f"Error capturing mic input: {e}")
    return None

# === Dummy Gemini-style Response Generator ===
def generate_response(prompt):
    prompt = prompt.lower()
    if "hello" in prompt:
        return "Hello! How can I help you today?"
    elif "your name" in prompt:
        return "I'm your AI assistant bot."
    elif "time" in prompt:
        from datetime import datetime
        return "The current time is " + datetime.now().strftime("%I:%M %p")
    else:
        return "Sorry, I didn't understand that. Try asking something else!"

# === Streamlit App ===
st.title("🧠 Voice Command Chatbot")
st.markdown("Speak and get a response!")

if st.button("🎙️ Speak Now"):
    user_text = listen_from_mic()
    if user_text:
        st.markdown(f"**You said:** {user_text}")
        response = generate_response(user_text)
        st.markdown(f"**Bot:** {response}")
        speak(response)

