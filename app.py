import streamlit as st
import tempfile
import os
import faiss

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq

# ---------------------------------------------------
# Secrets / Environment Variables
# ---------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable not set!")
    st.stop()

# ---------------------------------------------------
# App Config
# ---------------------------------------------------
st.set_page_config(page_title="RAG Document Chatbot", layout="wide")
st.title("📄 Document-Based Chatbot")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

embedding_model = load_embeddings()
reranker = load_reranker()

# LLM
llm = ChatGroq(
    model="qwen/qwen-7b",  # smaller model to avoid timeout
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

# ---------------------------------------------------
# Sidebar: Upload PDFs
# ---------------------------------------------------
with st.sidebar:
    st.header("📂 Knowledge Base")
    pdfs = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("⚙️ Process Documents"):
        if not pdfs:
            st.warning("Please upload at least one PDF")
        else:
            with st.spinner("Processing documents..."):
                docs = []
                for pdf in pdfs:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf.read())
                        loader = PyPDFLoader(tmp.name)
                        docs.extend(loader.load())

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ".", " "]
                )
                chunks = splitter.split_documents(docs)

                # Check if FAISS index already exists
                if os.path.exists("faiss_index"):
                    vector_store = FAISS.load_local("faiss_index", embedding_function=embedding_model)
                else:
                    dim = len(embedding_model.embed_query("test"))
                    index = faiss.IndexFlatIP(dim)
                    vector_store = FAISS(
                        embedding_function=embedding_model,
                        index=index,
                        docstore=InMemoryDocstore({}),
                        index_to_docstore_id={}
                    )
                    vector_store.add_documents(chunks)
                    vector_store.save_local("faiss_index")  # save prebuilt index

                bm25 = BM25Retriever.from_documents(chunks, bm25_variant="plus")
                bm25.k = 10

                st.session_state.vector_store = vector_store
                st.session_state.bm25_retriever = bm25
                st.session_state.chunks = chunks
                st.session_state.chat_history = []

                st.success(f"Processed {len(chunks)} chunks")

    # Clear Chat Button
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared")

# ---------------------------------------------------
# Retrieval Function
# ---------------------------------------------------
def retrieve(query, top_n=3):
    if not st.session_state.vector_store:
        return []

    faiss_retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )
    faiss_docs = faiss_retriever.invoke(query)
    bm25_docs = st.session_state.bm25_retriever.invoke(query)

    # Deduplicate
    all_candidates = []
    seen = set()
    for doc in faiss_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            all_candidates.append(doc)

    if not all_candidates:
        return []

    # Rerank with CrossEncoder
    pairs = [[query, doc.page_content] for doc in all_candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(all_candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_n]]

# ---------------------------------------------------
# LLM Call
# ---------------------------------------------------
def ask_llm(query):
    docs = retrieve(query)
    context = "\n\n".join(
        f"[source={d.metadata.get('source')} page={d.metadata.get('page')}] {d.page_content}"
        for d in docs
    )

    prompt = f"""
SYSTEM INSTRUCTIONS:
- You are an AI chatbot for the clinic "functiomed".
- Respond ONLY in German.
- Answer only using document context, do not invent answers.

CHAT HISTORY:
{st.session_state.chat_history}

USER QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

ANSWER (GERMAN ONLY):
""".strip()

    ai_msg = llm.invoke(prompt)
    return ai_msg.content

# ---------------------------------------------------
# Chat UI
# ---------------------------------------------------
st.subheader("💬 Chat With Your Documents")

if not st.session_state.vector_store:
    st.info("Upload PDFs and click Process to start chatting.")
else:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    query = st.chat_input("Ask a question...")
    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Thinking..."):
            answer = ask_llm(query)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", answer))
