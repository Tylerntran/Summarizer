import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
import tempfile
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from youtube_transcript_api.formatters import TextFormatter
# from openai import OpenAI
from datetime import time
from datetime import timedelta
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import openai
import torch
import torch.nn.functional as F

#Speech to text imports - Shoaib
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

st.set_page_config(layout="wide")

# API key input
with st.sidebar:
   openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

#client = openai.OpenAI(api_key=openai_api_key)
openai.api_key = openai_api_key

# Dependencies:
# pip3 install ffmpeg-python
# pip3 install whisper
# pip3 install moviepy==2.0.dev2
# pip3 install youtube-transcript-api
# pip3 install streamlit
# pip3 install openai==0.28
# pip3 install git+https://github.com/openai/whisper.git
# pip3 install transformers

# How to run:
# 1. Paste file into vscode / save as .py
# 2. Install dependencies using pip
# 3. In terminal, run: streamlit run {path/to/file}


# st.session_state:
# ["raw_transcript"] -> raw srt extracted from video to be processed
# ["srt_cleaned"] -> clean and trimmed transcript with timestamps for transcript tag
# ["trimmed_transcript"] -> clean and trimmed raw text, used for summarization
# ["video_path"] -> temp file path to video for the preview
# ["start_time"] -> start time for trimming
# ["end_time"] -> end time for trimming
# ["video_length"] -> used for trimming
# ["messages"] -> chat messages
# ["summary"] -> summarized text


# TODO:
# Error handling -> new videos, more than one, API Key etc
# Inputs for topic selection and summary length
# Chat bot integration with RAG/transcript
# Speech to text input for chatbot
# UI enhancements -> loading messages, buttons, layout sizes, etc

# Helper functions
def get_video_path(video_file):
    """
    Get the video path from file

    :param video_file: input file from st
    :return: file path
    :return: temp directory
    """
    # Open temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, video_file.name)
    with open(temp_file_path, "wb") as temp_file:
          temp_file.write(video_file.getvalue())

    return temp_file_path, temp_dir

def extract_audio_from_file(video_file_path, temp_dir):
  """
  Extract audio from the video
  and return an audio file path

  :param video_file_path: original video file path
  :param temp_dir: temp directory to put audio file
  :return: string -> audio file path
  """
  video = VideoFileClip(video_file_path)
  audio = video.audio

  temp_file_path_audio = os.path.join(temp_dir, "audio.mp3")
  audio.write_audiofile(temp_file_path_audio)

  return temp_file_path_audio

def transcribe_audio(audio_file_path):
  """
  Take an audio file and transcribe with whisper.

  :param audio_file_path: path name of audio file
  :return: transcript as string of text
  """
  model = whisper.load_model("base")

  result = model.transcribe(audio_file_path)

  return result["text"]

def transcribe_openAI(audio_file_path):
  """
  Take audio file and transcribe with OpenAI whisper API

  :param audio_file_path: path to the audio file
  :client: openAI api client to use whisper
  """
  audio_file = open(audio_file_path, "rb")

  transcript = openai.Audio.transcribe(
    file=audio_file,
    model="whisper-1",
    response_format="srt"
  )

  return transcript

def is_valid_video(video):
  """
  Check video validity -> error handling
  """

  return True

def format_time(seconds):
    """
    Format time into HH:MM:SS from seconds

    :param seconds: number of seconds
    :return: formatted string in HH:MM:SS
    """
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def time_to_timedelta(time_str):
        hours, minutes, seconds = map(int, time_str.split(":"))
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def extract_ending_time(srt_content):
    """
    Extracts the ending timestamp of the last subtitle from the SRT content.

    :param srt_content: The content of the SRT file as a string.
    :return: The ending timestamp as a string (HH:MM:SS,mmm).
    """
    # Regular expression to match SRT timestamps
    pattern = r"\d+\s(\d{2}:\d{2}:\d{2},\d{3})\s-->\s(\d{2}:\d{2}:\d{2},\d{3})"
    matches = re.findall(pattern, srt_content)

    # Extract the last ending timestamp
    if matches:
        _, last_end_time = matches[-1]
        return last_end_time
    return None

def convert_to_seconds(timestamp):
    """
    Converts a timestamp in HH:MM:SS,mmm format to total seconds.

    :param timestamp: The timestamp as a string (HH:MM:SS,mmm).
    :return: The total time in seconds as an int.
    """
    # Split the timestamp into hours, minutes, seconds, and milliseconds
    hours, minutes, rest = timestamp.split(':')
    seconds, milliseconds = rest.split(',')

    # Convert to integers
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)

    # Convert to total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return int(total_seconds)

def ending_time_seconds(srt_content):
   time_stamp = extract_ending_time(srt_content)
   seconds = convert_to_seconds(time_stamp)
   return seconds


# Transcript interaction Functions
def get_youtube_transcript(link):
  """
  Get youtube transcript srt with youtube-transcript-api

  :param link: string -> youtube video link
  :return: srt formatted transcript
  """
  transcript = {}
  try:
    video_id = link.split('v=')[1].split('&')[0]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
  except Exception as e:
    st.error(f"Error fetching transcript: {e}")

  formatter = SRTFormatter()

  srt_formatted = formatter.format_transcript(transcript=transcript)

  return srt_formatted

def get_file_transcript(video_file):
  """
  Transcribe video by extracting audio then
  turning audio to script.

  :param video_file: valid video file
  :return: String containing the transcript
  """
  video_path, temp_dir = get_video_path(video_file)
  audio_path = extract_audio_from_file(video_path, temp_dir)
  transcript = transcribe_openAI(audio_path)

  return transcript

def trim_transcript(transcript, start_seconds, end_seconds):
  """
  Trims and preprocesses an srt transcript

  :param transcript: .srt -> srt formatted transcript
  :param start: start time in seconds
  :param end: end time in seconds
  :return: string -> cleaned transcript
  """

  start_timedelta = timedelta(seconds=start_seconds)
  end_timedelta = timedelta(seconds=end_seconds)

  # Regular expression to parse the SRT content
  pattern = r"\d+\s(\d{2}:\d{2}:\d{2},\d{3})\s-->\s(\d{2}:\d{2}:\d{2},\d{3})\s(.+)"
  matches = re.findall(pattern, transcript)

  # Filter and clean subtitles within the specified time range
  cleaned_transcript = []
  filtered_subtitles = []
  for start, end, text in matches:
      # Convert timestamps to timedelta
      start_time = timedelta(hours=int(start[:2]), minutes=int(start[3:5]), seconds=int(start[6:8]))
      end_time = timedelta(hours=int(end[:2]), minutes=int(end[3:5]), seconds=int(end[6:8]))

      # Check if the subtitle is within the time range
      if start_time >= start_timedelta and end_time <= end_timedelta:
          cleaned_transcript.append(text)
          filtered_subtitles.append((start[:8], text))

  combined_with_timestamps = []
  for i in range(0, len(filtered_subtitles), 2):
      timestamp1, text1 = filtered_subtitles[i]
      if i + 1 < len(filtered_subtitles):  # If there is a second entry
          _, text2 = filtered_subtitles[i + 1]
          combined_with_timestamps.append(f"[{timestamp1}] {text1} {text2}")
      else:  # If it's the last single entry
          combined_with_timestamps.append(f"[{timestamp1}] {text1}")

  return "\n".join(cleaned_transcript), "\n\n".join(combined_with_timestamps)



# Summarization -> From Iliyan
def summarize_transcript(transcript, word_count, topic=None):
    """
    Summarize a transcript using OpenAI GPT-4 API (32k context version).

    Args:
        transcript (str): The transcript text to summarize.
        word_count (int): Desired word count for the summary.
        topic (str, optional): Specific topic to focus on. Defaults to None.

    Returns:
        str: The generated summary.
    """
    # Construct the base prompt
    base_prompt = f"Summarize the following transcript in approximately {word_count} words."

    # Add topic focus if specified
    if topic:
        base_prompt += f" Focus on the topic: '{topic}'."

    # Append the transcript to the prompt
    full_prompt = f"{base_prompt}\n\nTranscript:\n{transcript}\n"


    # Call the OpenAI API
    #response = client.chat.completions.create(
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Ensure you're using the 32k version for long inputs
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.7,  # Control creativity
        max_tokens=word_count * 5,  # Approximate 5 tokens per word
        top_p=1,  # Keep outputs focused
        frequency_penalty=0,  # No penalty for repetitive phrases
        presence_penalty=0  # No penalty for novel content
    )

    # Extract the summary from the response
    summary = response.choices[0].message.content
    return summary

# getting system response
def get_completion(messages, model="gpt-4o"):

      response = openai.ChatCompletion.create(
          model=model,
          messages=messages,
          temperature=0
      )
      return response['choices'][0]['message']['content'].strip()

# cerating embeddings
def embed_text(text, tokenizer, model):
    # Ensure the input is valid
    if not isinstance(text, str):
        raise ValueError(f"Invalid input type for text: {type(text)}. Expected string.")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Pass the tokens through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the correct attribute for embeddings
    if hasattr(outputs, 'pooler_output'):
        vectors = outputs.pooler_output
    else:
        # Fallback to mean pooling over `last_hidden_state`
        vectors = outputs.last_hidden_state.mean(dim=1)

    return vectors


def find_most_relevant_document(query, document_embeddings, sentences, tokenizer, model, summary, threshold=0.1):
    # Embed the query
    query_embedding = embed_text(str(query), tokenizer, model)

    # Calculate cosine similarity
    similarities = F.cosine_similarity(query_embedding, document_embeddings, dim=1)

    # Find relevant sentences
    important = ""
    for i in range(len(similarities)):  # Include the last similarity score
        if similarities[i] >= threshold:
            important += sentences[i] + " "  # Add a space between sentences

    return important.strip() if important else summary

def parse_script(script):
    # Regex to match timestamp and associated text
    pattern = r"\[(\d{2}:\d{2}:\d{2})\](.*?)((?=\[\d{2}:\d{2}:\d{2}\])|$)"
    matches = re.findall(pattern, script, re.DOTALL)

    parsed_data = []
    for match in matches:
        timestamp, text, _ = match
        parsed_data.append({"timestamp": timestamp, "text": text.strip()})

    return parsed_data


# cerating embeddings
def chatBotInteraction(query, summary, tokenizer, model):
    # Parse script and extract sentences
    parsed_data = parse_script(summary)
    sentences = [entry["text"] for entry in parsed_data if entry["text"].strip()]

    # Embed documents
    document_embeddings = torch.vstack([embed_text(sent, tokenizer, model) for sent in sentences])

    # Find the most relevant document
    doc = find_most_relevant_document(query, document_embeddings, sentences, tokenizer, model, summary)

    # Construct the prompt for the model (internal use only)
    prompt = f"answer the query: {query}, and use the following information to guide your answer: {doc}"

    # Call GPT model with the constructed prompt
    user_q = {"role": "user", "content": query}  # User's original query
    system_message = {"role": "system", "content": prompt}  # Full prompt for GPT
    response = get_completion([system_message])

    gpt_response = {"role": "assistant", "content": response}

    return user_q, gpt_response


# ################ Shoaib Speech to text ####################
# Function to record audio
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile

def record_audio(duration=5, samplerate=44100, device=None):
    """Records audio using sounddevice and saves to a temporary file."""

    # Query available devices if device is not provided
    if device is None:
        device = sd.default.device[0]  # Use the default input device

    # Get the number of input channels for the selected device
    device_info = sd.query_devices(device)
    input_channels = device_info['max_input_channels']

    # Ensure there is at least one input channel
    if input_channels < 1:
        raise ValueError("Selected device does not support any input channels.")

    # Record audio with the detected number of input channels
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=input_channels, dtype=np.int16, device=device)
    sd.wait()  # Wait until recording is finished

    # Save audio to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, samplerate, audio)
    return temp_file.name


# Function to transcribe and update the query box
def transcribe_and_update_query(file_path):
    """Transcribes audio using Whisper and updates the query box."""
    try:
        transcription = transcribe_audio(file_path)  # Transcribe audio
        st.session_state.transcription_status = transcription  # Update transcription status
    finally:
        os.remove(file_path)  # Clean up temporary file

# ####################### Shoaib Speech to text ######################

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def main():

# Initialize session state variables  -- Shoaib
  if "summary" not in st.session_state:
    st.session_state["summary"] = ""
  if "transcription_status" not in st.session_state:
    st.session_state["transcription_status"] = ""
  if "messages" not in st.session_state:
      st.session_state["messages"] = [{"role": "assistant", "content": "Have any questions about the video?"}]
  if "is_listening" not in st.session_state:
      st.session_state["is_listening"] = False
  if "query" not in st.session_state:
      st.session_state["query"] = ""

  # Streamlit UI headers
  st.title("🎥 Video Summarizer")
  st.header("Upload a video or enter a youtube link to summarize!")

  # Accept new videos
  uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])

  # Accept a link
  link = st.text_input("Link to Youtube video")

  # Update states
  if st.button("Select Video"):
    if len(link) > 3:
      srt = get_youtube_transcript(link)
      st.session_state['raw_transcript'] = srt
      st.session_state['video_length'] = ending_time_seconds(srt)
      video_path = link
      st.session_state['video_path'] = video_path

    elif uploaded_video is not None:
      if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
      # Display video and transcript underneath
      srt = get_file_transcript(uploaded_video)
      st.session_state['raw_transcript'] = srt
      st.session_state['video_length'] = ending_time_seconds(srt)
      video_path, _ = get_video_path(uploaded_video)
      st.session_state['video_path'] = video_path


  # Video processed and we have transcript -> show summary and bot
  if 'raw_transcript' in st.session_state:
    video_col, summary_col = st.columns([2, 3])

    # Video preview and trim settings
    with video_col:
      st.subheader("Video Preview")
      st.video(st.session_state['video_path'])
      # Select times with slider
      start_time, end_time = st.slider(
          "Select a start and end time for summary",
          value=(0, st.session_state['video_length']),
          format="%.0f"
      )
      # Update states with new trimmed content
      st.session_state['start_time'] = start_time
      st.session_state['end_time'] = end_time
      st.session_state['trimmed_transcript'], st.session_state['cleaned_srt'] = trim_transcript(
         st.session_state['raw_transcript'],
         start_time,
         end_time
      )

      # Show start and end times
      col1, col2 = st.columns(2)
      with col1:
        st.subheader("Start time")
        st.write(format_time(start_time))
        st.session_state['start_time'] = start_time

      with col2:
        st.subheader("End time")
        st.write(format_time(end_time))
        st.session_state['end_time'] = end_time



    # Summary and Chatbot tabs
    with summary_col:
      transcript_tab, summary_tab, chat_tab = st.tabs(["Transcript", "Summary", "QA Chat"])

      # Display original transcript
      with transcript_tab:
        with st.container(height=600):
          st.write(st.session_state['cleaned_srt'])

      # Check for API key
      if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

      # Display generated summary
      with summary_tab:
        st.subheader("Summary")

        # Inputs for topic and word count
        topic = st.text_input("Enter a specific topic for the summary (optional):")
        word_count = st.selectbox(
        "Select the number of words for the summary:",
        options=[50, 100, 150, 250, 500],
        index=2  # Default to 150 words
        )

        transcript = st.session_state['trimmed_transcript']

        # Button to generate the summary
        if st.button("Generate Summary"):
          summary = summarize_transcript(transcript, word_count, topic)
          st.session_state['summary'] = summary

        # Show the summary only if it exists
        if "summary" in st.session_state:
            st.write(st.session_state['summary'])

      # Chat bot
      with chat_tab:
        if "messages" not in st.session_state:
          st.session_state['messages'] = [{"role": "assistant", "content": "Have any questions about the video?"}]

        # Display the messages here
        with st.container(height=550):
          for msg in st.session_state.messages:
              st.chat_message(msg["role"]).write(msg["content"])

        # # Chat input -- Haydee's original work
        # if prompt := st.chat_input():
        #     if not openai_api_key:
        #       st.info("Please add your OpenAI API key to continue.")
        #       st.stop()

        #     user_q, gpt_response = chatBotInteraction(prompt, transcript, tokenizer, model)
        #     st.session_state.messages.append(user_q)
        #     st.session_state.messages.append(gpt_response)
        #     st.rerun()
        #     #st.session_state.messages.append({"role": "user", "content": prompt})
        #     # TODO: use RAG complete function
        #     #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        #     #msg = response.choices[0].message.content
        #     #st.session_state.messages.append({"role": "assistant", "content": msg})
        #     #st.rerun()
        # Input row: mic button and text input

        # Input row: mic button and text input
    col1, col2 = st.columns([0.1, 0.9])  # Adjust column widths as needed

    with col1:
        # Microphone button (starts recording)
        if st.button("🎤", key="mic_button"):
            st.session_state.is_listening = True
            st.session_state.query_placeholder = "Recording... Please speak into the microphone."

    with col2:
        # Text input box for queries
        query = st.chat_input(placeholder="Type your query here...", key="query")

    # Microphone recording logic
    if st.session_state.is_listening:
        # Use Streamlit's audio input for recording
        transcription = record_audio_with_streamlit()  # Get transcription of the recorded audio

        if transcription:
            st.session_state.query_placeholder = "✅ Recording complete. Processing transcription..."
            st.session_state.temp_query = transcription  # Store the transcription as the query
            st.session_state.is_listening = False  # Reset listening state

    # Handle query submission (either typed or transcribed from speech)
    if query or 'temp_query' in st.session_state:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # Get the correct query (either from text input or transcription)
        final_query = query if query else st.session_state.temp_query

        # Ensure the query is not empty
        if final_query.strip() != "":
            user_q, gpt_response = chatBotInteraction(final_query, str(st.session_state['cleaned_srt']), tokenizer, model)
            st.session_state.messages.append(user_q)
            st.session_state.messages.append(gpt_response)

        # Clear temp query after submission
        st.session_state.temp_query = ""  # Reset temp_query after sending
        st.session_state.query_placeholder = "Type your query here..."

        # Ensure to rerun only when new query is added
        if final_query.strip() != "":
            st.rerun()


if __name__ == "__main__":
    #initialized these above
    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # main(tokenizer, model)
    main()
