import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
import openai
import time

# Layout as wide and adding custom title
st.set_page_config(page_title="EduGPT", layout="wide")

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

os.environ["OPENAI_API_KEY"] = user_api_key
# Initialize the selected model
openai.api_key = user_api_key  # Replace with your OpenAI API key

@st.cache_data
def read_pdf(file_path):
    # load the document as before
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

# Mapping of PDFs
pdf_mapping = {
    'English Book': 'English.pdf',
    'Tax Regime': 'New-vs-Old-Tax.pdf',
    'Reinforcement Learning': 'SuttonBartoIPRLBook2ndEd.pdf',
    'GPT4 All Training': '2023_GPT4All_Technical_Report.pdf',
    # Add more mappings as needed
}

# Main Streamlit app
def main():
    with st.sidebar:
        st.title('EduBOT')
        st.markdown('''
        ## About
        Choose the desired PDF, select a model and retrieval method, and then perform a query.
        ''')

    custom_names = list(pdf_mapping.keys())
    selected_custom_name = st.sidebar.selectbox('Choose your PDF', ['', *custom_names])
    selected_actual_name = pdf_mapping.get(selected_custom_name)

    # Create a sidebar for model selection
    selected_model = st.sidebar.selectbox("Select a model", ["gpt-3.5-turbo", "gpt-4"])

    # Retrieval method dropdown
    selected_retrieval = st.sidebar.selectbox("Select a retrieval method", ["Conversational Retrieval", "Retrieval QA", "Multi Query", "Compression Retriever"])

     # Temperature slider
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    if selected_actual_name:
        pdf_folder = "pdfs"
        file_path = os.path.join(pdf_folder, selected_actual_name)

        try:
            text = read_pdf(file_path)
            # st.info("The content of the PDF is hidden. Type your query in the chat window.")
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return
        except Exception as e:
            st.error(f"Error occurred while reading the PDF: {e}")
            return

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(text)

        # Vectorize the documents and create vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embedding=embeddings)

        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=False)

        # Load the selected model
        llm = ChatOpenAI(temperature=temperature, max_tokens=1000, model_name=selected_model)

        if selected_retrieval == "Conversational Retrieval":
            qa = ConversationalRetrievalChain.from_llm(llm, 
                                                        vectorstore.as_retriever(),
                                                        get_chat_history=lambda o:o,
                                                        return_generated_question=True,
                                                        verbose=False,
                                                        memory=memory,
                                                        return_source_documents=True)

        elif selected_retrieval == "Retrieval QA":
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True
            )

        elif selected_retrieval == "Multi Query":
            multi_query_retriever_from_llm = MultiQueryRetriever.from_llm(
                                            retriever=vectorstore.as_retriever(search_type = "mmr"), llm=llm
            )
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=multi_query_retriever_from_llm, verbose=False, return_source_documents=True)

        elif selected_retrieval == "Compression Retriever":
            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(
                                                                    base_compressor=compressor,
                                                                    base_retriever=vectorstore.as_retriever(search_type = "mmr")
            )
            qa = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type='stuff', 
                                            retriever=compression_retriever, 
                                            verbose=False, 
                                            return_source_documents=True)

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"Ask your questions from PDF '{selected_custom_name}' using {selected_model} model?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if selected_retrieval == "Conversational Retrieval":
                result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
            elif selected_retrieval == "Retrieval QA":
                result = qa({"query": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
            elif selected_retrieval == "Multi Query":
                result = qa({"query": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
            elif selected_retrieval == "Compression Retriever":
                result = qa({"query": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

            with st.expander("Sources"):
                sources = result["source_documents"][0].page_content
                st.write(sources)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
        
                if selected_retrieval == "Retrieval QA":
                    response = result["result"]
                elif selected_retrieval == "Multi Query":
                    response = result["result"]
                elif selected_retrieval == "Compression Retriever":
                    response = result["result"]
                else:
                    response = result["answer"]
    
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        

if __name__ == "__main__":
    main()
