import random
import time
import streamlit as st
import os
import json
import uuid 

from rag.agent import ask_chatbot, ask_chatbot_with_files

st.set_page_config(layout="wide")

SAVE_DIR = "tmp"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def save_upload_file_to_local(session_id, uploaded_file):
    new_filename = f"{str(uuid.uuid4())}_{uploaded_file.name}"
    save_dir = os.path.join(SAVE_DIR, session_id)
    os.makedirs(save_dir, exist_ok=True)  
    save_path = os.path.join(save_dir, new_filename)
    # Write file to local disk
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.session_uploaded_files.append(save_path)

def update_total_stats(old, new):
    flag = False
    for key, value in new.items():
        if old.get(key) is None:
            old[key] = 0
        else:
            flag = True
            old[key] = old.get(key, 0) + value
    if flag:
        old["number_of_requests"] += 1
    return old

def reset_total_stats():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0,
        "number_of_requests": 0,
    }

def init_app():
    st.title("WE-Telecom Chatbot Interface")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_session" not in st.session_state:
        st.session_state.current_session = str(uuid.uuid4())

    if "stats" not in st.session_state:
        st.session_state.stats = reset_total_stats()
    
    if "session_uploaded_files" not in st.session_state:
        st.session_state.session_uploaded_files = []

def chat_history_section():
    st.title(f"Current Session: {st.session_state.current_session}")
    new_chat = st.button("New Chat", type="primary", use_container_width=True)
    
    # new chat button 
    if new_chat:
        st.session_state.current_session = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.stats = reset_total_stats()
        st.rerun()

def stats_section(stats_dict):             
    total_stats_list = [
        "**Total Requests Stats**",
        f"**Total Requests**: {stats_dict.get("number_of_requests")}",
        f"**Total Prompt Tokens**: {stats_dict.get("input_tokens")}",
        f"**Total Completion Tokens**: {stats_dict.get("output_tokens")}",
        f"**Total Cost**: {round(stats_dict.get("total_cost"),5)} $",
        ]    
    
    with st.container(border=True):         
        st.write("\n\n".join(total_stats_list))

def sidebar_section():
    with st.sidebar:
        chat_history_section()         
        stats_section(st.session_state.stats)

def display_history_messages():
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_input_message(prompt):
    if prompt:
        st.session_state.session_uploaded_files = []
        # 1. Access the text message
        if prompt.text:
            st.markdown(prompt.text)
    
        # 2. Access the uploaded files (as a list)
        if prompt.files:
            for uploaded_file in prompt.files:
                st.write(f"Uploaded: {uploaded_file.name}")
                save_upload_file_to_local(st.session_state.current_session, uploaded_file)

def display_agent_actions(actions):
    with st.expander("See Actions"):
        pass

def display_sources(sources):
    with st.expander("See Sources"):
        st.write(sources)

def display_request_stats(stats):
    with st.expander("See Stats"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Input Tokens", stats.get('input_tokens', 0))
        col2.metric("Output Tokens", stats.get('output_tokens', 0))
        col3.metric("Cost", round(stats.get('total_cost', 0), 5))
        col4.metric("Exec Time", round(stats.get('exec_time', 0), 5))

def main():          
    display_history_messages()

    # Main chat interface
    if prompt := st.chat_input("How can I help?", accept_file=True, file_type=["PDF", "DOCX", "TXT", "HTML"], accept_audio=False):
        with st.chat_message("human"):
            # st.markdown(prompt.text)
            display_input_message(prompt)

        with st.chat_message("AI"):
            with st.spinner("thinking"):
                response, stats = ask_chatbot_with_files(st.session_state.current_session, prompt.text, st.session_state.session_uploaded_files)
                final_answer = response.get("answer", "Something Went down, please try again")
                sources = response.get("chunks", [])

            st.session_state.stats = update_total_stats(st.session_state.stats, stats)
            st.markdown(final_answer)
            if sources:
                display_sources(sources)
            display_request_stats(stats)
              
        if final_answer: 
            st.session_state.messages.append({"role": "human", "content": prompt.text})
            st.session_state.messages.append({"role": "AI", "content": final_answer}) 

    sidebar_section()

if __name__ == "__main__":
    init_app()
    main()