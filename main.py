import time
import speech_recognition as sr
from groq import Groq
from PIL import ImageGrab, Image
# import cv2
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts
import asyncio
import os
from faster_whisper import WhisperModel
import re
import time
import streamlit as st
import base64

load_dotenv()


ngroq_api_key = os.getenv("GROQ_API_KEY")
genai_api_key = os.getenv("GENAI_API_KEY")

groq_client = Groq(api_key=ngroq_api_key)
genai.configure(api_key=genai_api_key)

num_cores = os.cpu_count()
whisper_size = 'base'
wake_word = "Prometheus"
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

r = sr.Recognizer()
source = sr.Microphone()

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
]
model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config, safety_settings=safety_settings)

sys_msg = (
    'You are a multi-modal AI voice assistant named Prometheus. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity. also call me Abraar while replying'
)

convo = [
    {'role': 'system', 'content': sys_msg}
]

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

# "capture webcam"
def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model named AVA. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["take screenshot", "None"]. '
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )
    convo = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content


def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)


# def web_cam_capture():
#     web_cam = cv2.VideoCapture(0)
#     if not web_cam.isOpened():
#         print('Error: Camera did not open successfully')
#         exit()
#     path = 'webcam.jpg'
#     ret, frame = web_cam.read()
#     cv2.imwrite(path, frame)



def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI '
        'assistant to the user. Instead take the user prompt input and try to extract all meaning '
        'from the photo relevant to the user prompt. Then generate as much objective data about '
        'the image for the AI assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text


def speak(response):
    async def _main():
        #en-US-JennyNeural- professional/coool
        #en-US-MichelleNeural- pleasant/comfort
        communicate = edge_tts.Communicate(response, 'en-US-MichelleNeural')
        await communicate.save("temp_output.wav")

    asyncio.run(_main())
    os.system("afplay temp_output.wav")
    # os.remove("temp_output.mp3")


def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    print(text)
    return text


def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    prompt_text = wav_to_text(prompt_audio_path)
    with open('report.txt', 'a') as file:  # 'a' mode to append to the file
            file.write(f"Prompt: {prompt_text}\n")
    clean_prompt = extract_prompt(prompt_text, wake_word)
    if clean_prompt!="None":
        with open('report.txt', 'a') as file:  # 'a' mode to append to the file
            file.write(f"User: {clean_prompt}\n")

    if clean_prompt:
        # Print the prompt on the Streamlit interface
        st.write(f"**User Prompt:** {clean_prompt}")

        call = function_call(clean_prompt)
        if 'take screenshot' in call:
            print('Taking screenshot')
            take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print('Capturing webcam')
            #web_cam_capture()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam.jpg')
        else:
            visual_context = None

        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        
        # Print the response on the Streamlit interface
        st.write(f"**Assistant Response:** {response}")
        with open('report.txt', 'a') as file:  # 'a' mode to append to the file
            file.write(f"AI: {response}\n")
            file.write("-" * 40 + "\n")  # Separator for readability

        speak(response)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay', wake_word, 'followed with your prompt.\n')
    r.listen_in_background(source, callback)
    print("end")

    while True:
        time.sleep(0.5)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)
    print(match)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

#--------------------------------UI------------------------------------------------
#st.set_page_config(page_title="Centered Circular Button Interface", layout="wide")
    
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 70vh;
    }

    .stButton button {
        width: 150px;
        height: 150px;
        font-size: 20px;
        margin-top: 200px;
        margin-left: auto;
        margin-right: auto;
        display: block;
        border-radius: 50%;  /* Makes the button circular */
    }

    .loader {
        border: 16px solid #f3f3f3;
        border-radius: 50%;
        border-top: 16px solid #3498db;
        width: 120px;
        height: 120px;
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .listening-text {
        margin-top: 10px;
        font-size: 24px;
        color: #3498db;
        margin-left:20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

image_path = "prometheus.png"  # replace with your image path
image_base64 = get_base64_of_bin_file(image_path)
st.sidebar.markdown(
    f'<div style="display: flex; justify-content: space-between; align-items: center;"><h2 style="font-family: Montserrat; font-size: 30px;">Prometheus.ai</h2><img src="data:image/png;base64,{image_base64}" style="width: 100px; margin-left: 2px;"></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("Multi Model Voice assistant")

# Define a callback function to update the session state
def start_button_clicked():
    st.session_state.button_clicked = True

# Check if the button has been clicked
if st.session_state.button_clicked:
    st.markdown(
        '''
        <div class="centered">
            <div class="loader"></div>
            <div class="listening-text">Listening...</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    

    # Replace this with the actual function that starts listening
    start_listening()
else:
    st.button("START", on_click=start_button_clicked)

