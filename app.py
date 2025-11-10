from typing import Dict, List

import streamlit as st

from config.config import DEFAULT_SYSTEM_PROMPT
from models.llm import generate_chat_completion
from utils.rag import SimpleVectorStore, chunk_text
from utils.web_search import search_web


def _init_session_state() -> None:
    """Initialise all Streamlit session_state keys used by the app."""
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Concise"
    if "use_web_search" not in st.session_state:
        st.session_state.use_web_search = False
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""   

def get_chat_response(
    messages: List[Dict[str, str]],
    system_prompt: str,
    response_mode: str,
    use_web_search: bool,
) -> str:
    """Build context (RAG + web search) and get a response from the LLM.

    Args:
        messages: Conversation history with roles "user"/"assistant".
        system_prompt: Base system instructions.
        response_mode: "Concise" or "Detailed".
        use_web_search: Whether to enrich context with web search results.

    Returns:
        Model-generated reply text, or an error message if the call fails.
    """
    try:
        if not messages:
            return "I did not receive any user message."

        last_user_message = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            messages[-1]["content"],
        )

        context_sections: List[str] = []

        # RAG: query in-session knowledge base if available
        vector_store: SimpleVectorStore = st.session_state.vector_store
        kb_hits = vector_store.similarity_search(last_user_message)
        if kb_hits:
            kb_text = "\n\n".join(f"- {doc.text}" for (doc, _score) in kb_hits)
            context_sections.append("Knowledge base context:\n" + kb_text)

        # Web search: optional, controlled from UI
        if use_web_search:
            web_context = search_web(last_user_message)
            if web_context:
                context_sections.append("Web search results:\n" + web_context)

        # Practical behaviour differences between modes
        if response_mode.lower() == "concise":
            mode_instruction = (
                "Give a short answer of at most 3–5 sentences. "
                "Focus only on what directly answers the user question. "
                "If you list items, use a very small bullet list. "
                "Do not add background theory or long explanations."
            )
            max_tokens = 256
            temperature = 0.2
        else:
            mode_instruction = (
                "Give an in-depth explanation. "
                "Start with a brief 2–3 sentence summary, then add details in clearly separated "
                "sections with headings and bullet points where useful. "
                "Keep reasoning high-level and user-facing; do not describe your internal steps."
            )
            max_tokens = 1024
            temperature = 0.4

        # Compose final system message
        base_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        if context_sections:
            base_prompt += (
                "\n\nYou have additional context below. "
                "Prefer it over prior knowledge when they conflict.\n\n"
                + "\n\n".join(context_sections)
            )
        base_prompt += "\n\n" + mode_instruction

        # Convert stored messages into OpenAI chat format
        openai_messages: List[Dict[str, str]] = [
            {"role": "system", "content": base_prompt}
        ]
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            openai_messages.append({"role": role, "content": msg["content"]})

        reply = generate_chat_completion(
            openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return reply

    except Exception as exc:  # noqa: BLE001
        return f"Error while generating response: {exc}"


def instructions_page() -> None:
    """Instructions and setup page."""
    st.title("The Chatbot Blueprint")

    st.markdown(
        "This app demonstrates an end-to-end LLM chatbot with RAG and OpenAI web search."
    )

    st.markdown(
        """## Setup

1. **Create API key**

   - OpenAI: create a key from your OpenAI dashboard with access to both normal
     chat models (e.g. `gpt-4o`) and the search preview model
     (e.g. `gpt-4o-search-preview` or `gpt-4o-search-preview`).

2. **Set environment variables** before running `streamlit run app.py`:

   ```bash
   export OPENAI_API_KEY="sk-..."
   # optional overrides:
   export OPENAI_MODEL="gpt-4o"
   export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
   export OPENAI_SEARCH_MODEL="gpt-4o-search-preview"
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:

   ```bash
   streamlit run app.py
   ```

## Features

- Multi-turn chat using OpenAI Chat Completions.
- Retrieval-Augmented Generation (RAG) over files you upload.
- Web search using OpenAI GPT-4o Search Preview models.
- Two answer styles: **Concise** and **Detailed**.
"""
    )


def chat_page() -> None:
    """Main chat interface page."""
    _init_session_state()

    st.title("AI Assistant")


    # One-time clear KB logic (runs before widgets are created)
    if "clear_kb_flag" not in st.session_state:
        st.session_state.clear_kb_flag = False

    if st.session_state.clear_kb_flag:
        st.session_state.vector_store = SimpleVectorStore()
        if "kb_files" in st.session_state:
            del st.session_state["kb_files"]
        st.session_state.clear_kb_flag = False





    # Sidebar configuration for the chat behaviour
    with st.sidebar:
        st.subheader("Chat settings")

        st.session_state.response_mode = st.radio(
            "Response mode",
            options=["Concise", "Detailed"],
            index=0 if st.session_state.response_mode == "Concise" else 1,
        )


        st.session_state.use_web_search = st.checkbox(
            "Enable web search",
            value=st.session_state.use_web_search,
        )

        st.session_state.system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            help="High-level instructions for the assistant.",
        )

        st.markdown("---")
        st.subheader("Knowledge base (RAG)")
        uploaded_files = st.file_uploader(
            "Upload text/markdown files",
            type=["txt", "md"],
            accept_multiple_files=True,
            key="kb_files",
        )

        vector_store: SimpleVectorStore = st.session_state.vector_store

        if uploaded_files:
            new_chunks: List[str] = []
            for uploaded in uploaded_files:
                try:
                    content = uploaded.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue
                new_chunks.extend(chunk_text(content))

            if new_chunks:
                with st.spinner("Indexing uploaded content..."):
                    vector_store.add_texts(new_chunks)
                st.success(
                    f"Indexed {len(new_chunks)} text chunks into the knowledge base."
                )

        st.caption(
            "The knowledge base is stored only in your current session memory and will reset on reload."  # noqa: E501
        )

        st.markdown("---")
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear knowledge base", use_container_width=True):
            st.session_state.clear_kb_flag = True
            st.rerun()

    # Chat history display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Store and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_chat_response(
                    messages=st.session_state.messages,
                    system_prompt=st.session_state.system_prompt,
                    response_mode=st.session_state.response_mode,
                    use_web_search=st.session_state.use_web_search,
                )
                st.markdown(reply)

        # Persist assistant message
        st.session_state.messages.append({"role": "assistant", "content": reply})


def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="AI Use Case Chatbot",
        page_icon="🤖",
        layout="wide",
    )

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Instructions", "Chat"], index=1)

    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()


if __name__ == "__main__":  # pragma: no cover
    main()
