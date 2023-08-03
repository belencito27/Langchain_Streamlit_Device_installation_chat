# Import os to set API key
import os

# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
#Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store
from langchain.vectorstores import Chroma

# Import vector store
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

#Set API Key for OpenAI service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = "Your_OpenAI_Key"

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose= True)
embeddings = OpenAIEmbeddings()

# Create and Load PDF Loader
loader = PyPDFLoader('DCH3416_User_Guide.pdf')
# Split pages from pdf
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name= 'DCH3416_User_Guide')

# Create vectorstore info object - metadata repo
vectorstore_info = VectorStoreInfo(
    name= "DCH3416_User_Guide",
    description="Operation manual TV Box DCH3416 User guide Motorola",
    vectorstore=store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end to end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# set tittle of app
st.title('🦜️🔗 GPT AI - Your Network / Devices Intallation Guide')

# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    #response = llm(prompt)
    # Swapt out the raw llm for a document agent
    response = agent_executor.run(prompt)
    # .. and write it out to the screen
    st.write(response)

    # With a streamlit expander
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt)
        # Write out the first
        st.write(search[0][0].page_content)