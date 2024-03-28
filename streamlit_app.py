import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings  import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import  OpenAI
#from langchain.chains.question_answering import load_qa_chain
#from langchain.callbacks import get_openai_callback
os.environ["OPENAI_API_KEY"] = "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"
#side bar contents
with st.sidebar:
    st.title('🤗💬 PDF LLM Chat App')
    st.markdown("""
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchian.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [Github](https://github.com/praj2408/Langchain-PDF-App-GUI) Repository
                
    """)
    add_vertical_space(5)
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    st.write("Made by Nyongesa.")
@st.cache_data
def preprocess_pdf(query):
    # upload a PDF file
    #st.write(pdf)
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # it will divide the text into 800 chunk size each (800 tokens)
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        #st.write(chunks)
        
        
        ## embeddings
        
        store_name = pdf.name[:-4]
        
        
        vector_store_path = os.path.join('vector_store', store_name)
        
        
        if os.path.exists(f"{vector_store_path}.pkl"):
            pass
            #with open(f"{vector_store_path}.pkl", 'rb') as f:
            #    VectorStore = pickle.load(f)
            # st.write("Embeddings loaded from the Disk")
                
        else:
            pass
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(chunks, embedding=embeddings)
        retriever = db.as_retriever()
        llm = OpenAI()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return qa({"query": f"{query}"})['result']
        # Accept user questions/query
def  main():
    if  pdf !=None:
        query = st.text_input("Ask questions about your PDF file")
        if query:
            response = preprocess_pdf(query)
            st.write(response)
    else:
        st.title('🤗💬 PDF LLM Chat App')
        st.error("Please  upload a  pdf  file  to continue.")
     

            
            

    
    
    
if __name__ == "__main__":
    main()
    
