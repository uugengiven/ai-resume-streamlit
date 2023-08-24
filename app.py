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
logger.addHandler(AzureEventHandler(connection_string=st.secrets.azure_connection_string))

openai.api_key = st.secrets.openai_key
st.header("John Lange's Personal Historian")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "What would you like to know about John Lange?"}
    ]
try:
    logger.info("checking prompt state" + st.session_state.prompt)
except:
    pass
if 'prompt' not in st.session_state:
    st.session_state.prompt = None
    logger.info("prompt set to none")
try:
    logger.info("checking prompt state after check" + st.session_state.prompt)
except:
    pass

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing John's history - give me a moment"):
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

if st.session_state.prompt != prompt:
    if st.session_state.prompt is None:
        logger.info("st prompt is none")
    else:
        logger.info(st.session_state.prompt)
    try:
        logger.info('question ' + prompt)
    except:
        pass
    st.session_state.prompt = prompt

logger.info(prompt)
