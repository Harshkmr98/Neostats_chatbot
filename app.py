from __future__ import annotations
from typing import Dict, List
import streamlit as st
from config.config import DEFAULT_SYSTEM_PROMPT
from models.llm import generate_chat_completion
from utils.rag import SimpleVectorStore, chunk_text
from utils.web_search import search_web
from utils.pdf_utils import extract_text_from_pdf

def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()
    if "use_web_search" not in st.session_state:
        st.session_state.use_web_search = False
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Concise" 

def _ingest_files(uploader_files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> int:
    added = 0
    vs = st.session_state.vector_store
    for uf in uploader_files:
        data = uf.read()
        text = ""
        if uf.name.lower().endswith(".pdf"):
            try:
                text = extract_text_from_pdf(data)
            except Exception as e:
                st.warning(f"Failed to parse {uf.name}: {e}")
                continue
        else:
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception as e:
                st.warning(f"Failed to decode {uf.name}: {e}")
                continue
        chunks = chunk_text(text)
        vs.add_texts(chunks, source=uf.name)
        added += len(chunks)
    return added

def _style_instruction() -> str:
    mode = st.session_state.response_mode
    if mode == "Concise":
        return "Answer briefly in 3-5 sentences. Use bullet points when helpful. Do not include internal reasoning steps."
    else:
        return "Provide an in-depth, well-structured answer with sections, examples, and caveats. Present final reasoning only, not hidden internal steps."

def instructions_page() -> None:
    st.markdown(
        """

        ### How this works
        1. Upload PDFs or text files in the sidebar. They are chunked and embedded to a local in-memory store.

        2. Ask questions in the chat. The app retrieves top-k chunks and passes them to the model.

        3. Toggle **Response Mode** between Concise and Detailed.

        4. Optional web search augmentation.

        """
    )

def chat_page() -> None:
    # st.header("AI Use Case Demo with PDF Upload")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask something...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare retrieval context
        retrieved = st.session_state.vector_store.similarity_search(prompt)
        context_blocks = []
        for doc, score in retrieved:
            source = f"\n\n[SOURCE: {doc.source}]" if doc.source else ""
            context_blocks.append(f"<doc score={score:.3f}>{doc.text}</doc>{source}")
        web_ctx = search_web(prompt) if st.session_state.use_web_search else ""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": st.session_state.system_prompt},
            {"role": "system", "content": _style_instruction()},
            {"role": "user", "content": f"Context from documents:\n{chr(10).join(context_blocks)}\n\nWeb:\n{web_ctx}\n\nQuestion: {prompt}"},
        ]

        # Show "thinking…" until the response arrives (no internal reasoning revealed)
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            with st.spinner("Thinking…"):
                answer = generate_chat_completion(messages)
            thinking_placeholder.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

def main() -> None:
    _init_session_state()

    with st.sidebar:
        st.subheader("Knowledge Base")
        uploaded = st.file_uploader("Upload PDFs or .txt", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded:
            added = _ingest_files(uploaded)
            st.success(f"Added {added} chunks from {len(uploaded)} file(s)")

        

        st.divider()
        st.subheader("Response Mode")
        st.radio("Choose style", ["Concise", "Detailed"], key="response_mode", horizontal=True)

        st.divider()
        st.checkbox("Enable web search", key="use_web_search")

        st.divider()
        st.subheader("System Prompt")
        st.text_area("System prompt", key="system_prompt", height=120)

        st.divider()

        if st.button("Reset chat"):
            st.session_state.messages.clear()

        st.caption("About: Streamlit + OpenAI + simple RAG with PDF ingest.")

    st.title("AI for the Future")
    page = st.radio("Go to", ["Instructions", "Chat"], index=1, horizontal=True)
    if page == "Instructions":
        instructions_page()
    else:
        chat_page()

    st.divider()    

if __name__ == "__main__":
    main()
