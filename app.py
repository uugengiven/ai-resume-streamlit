import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index.memory import ChatMemoryBuffer
import logging
from opencensus.ext.azure.log_exporter import AzureEventHandler

logger = logging.getLogger("Streamlit app")
logger.setLevel(logging.INFO)


openai.api_key = st.secrets.openai_key
st.header("John Lange's Resume as a Service (RaaS)")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "What would you like to know about John Lange? Some suggestions to start: What is his work history? What are some projects he worked on? What are his skills?"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing John's history - give me a moment"):
        logger.addHandler(AzureEventHandler(connection_string=st.secrets.azure_connection_string))
        logger.info('startup starting')
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an enthusastic assistant and an expert on John Lange. Assume that all questions are related to John Lange. Keep your answers based on facts - do not hallucinate facts."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        logger.info('startup complete')
        return index

index = load_data()

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory, 
    verbose=True
    )

if prompt := st.chat_input("Your question"):
    logger.info('question ' + prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)