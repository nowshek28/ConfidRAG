import streamlit as st

st.set_page_config(page_title="Basic Layout", layout="wide")

# ---------- Session State ----------
if "list_items" not in st.session_state:
    st.session_state.list_items = []
if "text5_value" not in st.session_state:
    st.session_state.text5_value = ""

# Functions
def add_item(prefix, value):
    if value:
        st.session_state.list_items.append(f"{prefix}: {value}")

# ---------- LAYOUT ----------
col1, col2 = st.columns([1, 3])

with col1:
    # Buttons TEXT1–TEXT3
    if st.button("TEXT1"):
        val = st.text_input("Enter value for TEXT1", key="text1")
        if val:
            add_item("TEXT1", val)

    if st.button("TEXT2"):
        val = st.text_input("Enter value for TEXT2", key="text2")
        if val:
            add_item("TEXT2", val)

    if st.button("TEXT3"):
        val = st.text_input("Enter value for TEXT3", key="text3")
        if val:
            add_item("TEXT3", val)

    # TEXT4 — list window
    st.subheader("TEXT4")
    st.write(st.session_state.list_items)

with col2:
    # Placeholder for the big black area
    st.subheader("Main Area")
    st.empty()

# ---------- Bottom Bar ----------
st.subheader("TEXT5")
st.session_state.text5_value = st.text_input("Enter text", value=st.session_state.text5_value, key="text5")

if st.button("TEXT6"):
    add_item("TEXT5", st.session_state.text5_value)
    st.session_state.text5_value = ""
