from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import pdfplumber
import streamlit as st
import os


# =========================================================
# 1. OPENAI API KEY
# =========================================================

MY_OPENAI_KEY = st.secrets["OPENAI_API_KEY"]


# =========================================================
# 2. STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="Betting Rules Assistant",
    page_icon="📚"
)

st.header("Betting Rules Assistant")


# =========================================================
# 3. DOCUMENTS FOLDER
# =========================================================

DOCUMENTS_FOLDER = "documents"


# =========================================================
# 4. READ ALL PDF FILES
# =========================================================

@st.cache_data
def load_documents():

    if not os.path.exists(DOCUMENTS_FOLDER):
        return None, "The documents folder was not found."

    pdf_files = [
        file_name
        for file_name in os.listdir(DOCUMENTS_FOLDER)
        if file_name.lower().endswith(".pdf")
    ]

    if not pdf_files:
        return None, "No PDF documents were found in the documents folder."

    documents_text = ""

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            DOCUMENTS_FOLDER,
            pdf_file
        )

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                page_text = page.extract_text() or ""

                documents_text += (
                    f"\n\n"
                    f"--- SOURCE: {pdf_file} | PAGE: {page_number} ---\n\n"
                    f"{page_text}"
                )

    return documents_text, None


documents_text, error_message = load_documents()


if error_message:
    st.error(error_message)
    st.stop()


# =========================================================
# 5. SPLIT TEXT INTO CHUNKS
# =========================================================

@st.cache_data
def create_chunks(documents_text):

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ",", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(documents_text)


chunks = create_chunks(documents_text)


# =========================================================
# 6. GENERATE EMBEDDINGS
# =========================================================

@st.cache_resource
def create_embeddings():

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=MY_OPENAI_KEY
    )


embeddings = create_embeddings()


# =========================================================
# 7. CREATE FAISS DATABASE
# =========================================================

@st.cache_resource
def create_vector_store(chunks):

    return FAISS.from_texts(
        chunks,
        embeddings
    )


vector_store = create_vector_store(chunks)


# =========================================================
# 8. RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)


# =========================================================
# 9. FORMAT DOCUMENTS
# =========================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# =========================================================
# 10. OPENAI MODEL
# =========================================================

@st.cache_resource
def create_llm():

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=1000,
        openai_api_key=MY_OPENAI_KEY
    )


llm = create_llm()


# =========================================================
# 11. PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",

        "You are an internal Betting Rules Assistant.\n\n"

        "Your job is to answer questions strictly based "
        "on the betting rules contained in the provided documents.\n\n"

        "IMPORTANT RULES:\n"

        "1. Use ONLY information contained in the provided documents.\n"

        "2. Do NOT use outside knowledge.\n"

        "3. Do NOT make assumptions.\n"

        "4. Do NOT invent or guess a rule.\n"

        "5. If the documents do not contain enough information "
        "to answer the question, clearly state that there is "
        "not enough information in the available rules.\n"

        "6. When answering a question about voiding, settlement "
        "or grading a bet, explain the rule that supports your answer.\n"

        "7. Always mention the source document and page when "
        "this information is available.\n\n"

        "Give the answer clearly and professionally.\n\n"

        "CONTEXT:\n"
        "{context}"
    ),

    (
        "human",
        "{question}"
    )

])


# =========================================================
# 12. CREATE RAG CHAIN
# =========================================================

chain = (

    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }

    | prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# 13. USER QUESTION
# =========================================================

user_question = st.text_input(
    "Ask a question about the betting rules:"
)


# =========================================================
# 14. GENERATE ANSWER
# =========================================================

if user_question:

    with st.spinner("Searching the betting rules..."):

        response = chain.invoke(user_question)

    st.subheader("Answer")

    st.write(response)
