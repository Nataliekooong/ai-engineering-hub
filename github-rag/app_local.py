import gc
import re
import uuid

from gitingest import ingest

from llama_index.core import Document, Settings, VectorStoreIndex, PromptTemplate
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

import streamlit as st

FILE_PATTERN = re.compile(r"^={20,}\nFILE: (.+?)\n={20,}\n", re.MULTILINE)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None


@st.cache_resource
def load_llm():
    llm = Ollama(model="llama3.2", request_timeout=120.0)
    return llm


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def process_with_gitingets(github_url):
    # or from URL
    summary, tree, content = ingest(github_url)
    return summary, tree, content


def split_into_files(content):
    """Split the gitingest text dump into (file_path, file_text) pairs."""
    matches = list(FILE_PATTERN.finditer(content))
    files = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if text:
            files.append((m.group(1).strip(), text))
    if not files:  # fallback if the format isn't what we expect
        files = [("repository", content)]
    return files


def build_nodes(docs):
    """Chunk each file separately. Markdown files are split by heading first."""
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    md_docs, other_docs = [], []
    for d in docs:
        if d.metadata["file_path"].lower().endswith((".md", ".markdown")):
            md_docs.append(d)
        else:
            other_docs.append(d)

    nodes = []
    if md_docs:
        md_nodes = MarkdownNodeParser().get_nodes_from_documents(md_docs)
        nodes += splitter(md_nodes)  # keep heading chunks within the size limit
    if other_docs:
        nodes += splitter.get_nodes_from_documents(other_docs)
    return nodes


with st.sidebar:
    st.header(f"Add your GitHub repository!")

    github_url = st.text_input("Enter GitHub repository URL", placeholder="GitHub URL")
    load_repo = st.button("Load Repository")

    if github_url and load_repo:
        try:
            st.write("Processing your repository...")
            repo_name = github_url.split('/')[-1]
            file_key = f"{session_id}-{repo_name}"

            if file_key not in st.session_state.get('file_cache', {}):

                summary, tree, content = process_with_gitingets(github_url)

                # NEW: one Document per file, with the file path as metadata
                files = split_into_files(content)
                docs = [
                    Document(text=text, metadata={"file_path": path})
                    for path, text in files
                ]
                nodes = build_nodes(docs)

                # setup llm & embedding model
                llm = load_llm()
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                Settings.embed_model = embed_model

                # Creating an index over the per-file chunks
                index = VectorStoreIndex(nodes, show_progress=True)

                Settings.llm = llm
                query_engine = index.as_query_engine(streaming=True)

                # ====== Customise prompt template ======
                qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                )
                qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                )

                st.session_state.file_cache[file_key] = query_engine
                st.success(f"Ready to Chat! Indexed {len(files)} files ({len(nodes)} chunks).")
            else:
                query_engine = st.session_state.file_cache[file_key]
                st.success("Ready to Chat!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with GitHub using RAG </>")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Get the repo name from the GitHub URL
            repo_name = github_url.split('/')[-1]
            file_key = f"{session_id}-{repo_name}"

            # Get query engine from session state
            query_engine = st.session_state.file_cache.get(file_key)

            if query_engine is None:
                st.error("Please load a repository first!")
                st.stop()

            # Use the query engine
            response = query_engine.query(prompt)

            # Handle streaming response
            if hasattr(response, 'response_gen'):
                for chunk in response.response_gen:
                    if isinstance(chunk, str):  # Only process string chunks
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
            else:
                # Handle non-streaming response
                full_response = str(response)
                message_placeholder.markdown(full_response)

            message_placeholder.markdown(full_response)
            if hasattr(response, "source_nodes"):
                with st.expander("Sources"):
                    for n in response.source_nodes:
                        score = f" (similarity {n.score:.2f})" if n.score is not None else ""
                        st.markdown(f"**`{n.node.metadata.get('file_path', 'unknown')}`**{score}")
                        st.code(n.node.get_content()[:500])
                        
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
            full_response = "Sorry, I encountered an error while processing your request."
            message_placeholder.markdown(full_response)



    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})