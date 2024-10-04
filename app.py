import streamlit as st
import base64
import uuid
import os
from llama_cpp import Llama
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from PIL import Image as PILImage
from io import BytesIO
import streamlit.components.v1 as components

# Streamlit Sidebar UI
st.sidebar.title("NotebookLM-Like System")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])
query = st.sidebar.text_input("Enter your query:")
submit_button = st.sidebar.button("Run Query")

# Enhanced User Interaction with Tabs
tab1, tab2, tab3 = st.tabs(["📄 Document Summarization", "🔍 Querying", "📊 Visualize & Collaborate"])

# Store document metadata and content globally
documents = []

# Helper Functions for Document Processing

# Function to load and split PDF document
def load_document(doc_file):
    if not os.path.exists(doc_file):
        st.error("File not found.")
        return []

    loader = UnstructuredPDFLoader(file_path=doc_file, strategy='hi_res')
    return loader.load()

# Summarization Function for Text, Images, and Tables
def summarize_content(text_docs, llama_model):
    prompt_text = "Summarize the following text or table for semantic retrieval."
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    summarize_chain = (
        {"element": RunnablePassthrough()}
        | prompt
        | RunnableLambda(lambda x: llama_model(x['element']))
        | StrOutputParser()
    )
    
    text_summaries = summarize_chain.batch(text_docs, {"max_concurrency": 5})
    return text_summaries

# Image Processing Functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def summarize_images(image_paths, llama_model):
    image_summaries = []
    for img_path in image_paths:
        base64_image = encode_image(img_path)
        summary = llama_model(f"Summarize this image: [IMAGE:{base64_image}]")
        image_summaries.append(summary)
    return image_summaries

# Setup Database Functions
def setup_databases():
    client = get_client('redis://localhost:6379')
    redis_store = RedisStore(client=client)
    llama_embed_model = LlamaCppEmbeddings(model_path="./llama-3b.bin")
    chroma_db = Chroma(collection_name="mm_rag", embedding_function=llama_embed_model)
    return redis_store, chroma_db

# Retrieve documents from vector database
def create_retriever(redis_store, chroma_db, text_summaries, text_docs, image_summaries, images):
    retriever = MultiVectorRetriever(vectorstore=chroma_db, docstore=redis_store)
    
    # Function to add documents to the retriever
    def add_documents(summaries, docs):
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        summary_docs = [Document(page_content=s, metadata={"doc_id": doc_ids[i]}) for i, s in enumerate(summaries)]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))

    if text_summaries:
        add_documents(text_summaries, text_docs)
    if image_summaries:
        add_documents(image_summaries, images)
    
    return retriever

# Build Multimodal Retrieval Pipeline
def build_multimodal_pipeline(retriever, llama_model):
    multimodal_rag = (
        {"context": itemgetter('context'), "question": itemgetter('input')}
        | RunnableLambda(lambda x: f"User question: {x['question']}, Context: {x['context']}")
        | RunnableLambda(lambda x: llama_model(x))
        | StrOutputParser()
    )

    retrieve_docs = itemgetter('input') | retriever
    return RunnablePassthrough.assign(context=retrieve_docs).assign(answer=multimodal_rag)

# Running the Query
def run_query(pipeline, query):
    response = pipeline.invoke({"input": query})
    st.markdown("### Answer:")
    st.markdown(response['answer'])

# Main Execution Section
if __name__ == "__main__":
    if uploaded_file:
        # Save the uploaded file
        temp_dir = "uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display document information in tab1
        with tab1:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            documents = load_document(file_path)

            # Splitting the document into text and table components
            text_docs = [doc.page_content for doc in documents]
            try:
                llama_model = Llama(model_path="./llama-3b.bin")
            except Exception as e:
                st.error(f"Error loading Llama model: {e}")
                llama_model = None

            # Summarize content
            if llama_model:
                text_summaries = summarize_content(text_docs, llama_model)
                st.write("Text Summaries Generated")

    if submit_button and query:
        with tab2:
            redis_store, chroma_db = setup_databases()
            retriever = create_retriever(redis_store, chroma_db, text_summaries, text_docs, [], [])
            multimodal_pipeline = build_multimodal_pipeline(retriever, llama_model)
            run_query(multimodal_pipeline, query)

    with tab3:
        st.markdown("Collaborative Visualization Coming Soon...")
        # You can add more features such as real-time collaboration here.
