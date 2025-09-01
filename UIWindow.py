import streamlit as st

st.set_page_config(page_title="Nested Layout", layout="wide")

# ---------- Session State ----------
if "list_items" not in st.session_state:
    st.session_state.list_items = []
if "chat_item" not in st.session_state:
    st.session_state.chat_item = []
if "Input_value" not in st.session_state:
    st.session_state.Input_value = ""
if "ask_value" not in st.session_state:
    st.session_state.ask_value = ""

# ---------- Functions ----------
def add_item(prefix, value):
    if value:
        st.session_state.list_items.append(f"{prefix}: {value}")

def add_chat(prefix, value):
    if value:
        st.session_state.chat_item.append(f"{prefix}: {value}")

# ---------- LAYOUT ----------
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.session_state.Input_value = st.text_input("Enter Here:", value=st.session_state.Input_value)
    if st.button(".txt file"):
        add_item("txt_file", st.session_state.Input_value)
        st.session_state.Input_value = ""
    if st.button(".txt folder"):
        add_item("txt_folder", st.session_state.Input_value)
        st.session_state.Input_value = ""
    if st.button("url upload"):
        add_item("url_upload", st.session_state.Input_value)
        st.session_state.Input_value = ""

    st.subheader("Uploads")
    if st.session_state.list_items:
        st.write(st.session_state.list_items)
    else:
        st.caption("No data uploaded yet...")

with col2:
    top_area = st.container(height="stretch", gap="large")
    bottom_area = st.container()
    

    with top_area:
        # Middle chat window
        st.subheader("Chat Area")
        chat_window = st.container(gap="large")
        with chat_window:
            if st.session_state.chat_item:
                for msg in st.session_state.chat_item:
                    st.write(msg)
            else:
                st.caption("No messages yet. Type below to start chatting.")

    with bottom_area:
        # Bottom bar
        leftcol, rightcol = st.columns([6,1],vertical_alignment="bottom")
        
        with leftcol:
            st.subheader("ASK")
            st.session_state.ask_value = st.text_input("Enter question here", value=st.session_state.ask_value, key="ask")
        with rightcol:
            if st.button("Send"):
                add_chat("User", st.session_state.ask_value)
                st.session_state.ask_value = ""
