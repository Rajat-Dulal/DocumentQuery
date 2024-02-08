import streamlit as st
import warnings
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore")

st.title("Document Chatbot")
st.markdown("Copyright © Rajat Dulal - All rights reserved 2024")
file = st.file_uploader("Upload your pdf document", type = ['pdf'])
if file is not None:
    pdf_file = f"./{file.name}"
    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

    #Define the model and prompt template
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """

    prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )

    stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    #loop for main chatbot
    question = st.text_input("Enter your query")
    if st.button("Inquire"):
        #question = "What is the moral of the story two frogs"
        docs = vector_index.get_relevant_documents(question)
        st.write(docs)
        print(docs)
        stuff_answer = stuff_chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
        )
        st.write(stuff_answer)
