# Adapted from: https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps#introduction
import streamlit as st

from sports_chatbot.chatbot import SportsChatbot

st.title("Sports News ChatBot")
chatbot_name = "Sporty"
# TODO: change this hardcoded data reference
chatbot = SportsChatbot("./data/sports_chatbot/Day_1.csv")

response = (
    "Hello! My name is Sporty, how can I assist you in the world of sports today?"
)
# Display assistant response in chat message container
with st.chat_message(chatbot_name):
    st.markdown(response)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt:
    response = chatbot.query(prompt)["answer"]
    # Display assistant response in chat message container
    with st.chat_message(chatbot_name):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": chatbot_name, "content": response})
