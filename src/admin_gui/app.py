import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Weaviate Admin (Mini)", layout="wide")
st.title("Weaviate Admin (Mini)")

st.caption(f"Backend: {BACKEND_URL}")

col1, col2 = st.columns(2)

with col1:
    if st.button("Refresh count"):
        r = requests.get(f"{BACKEND_URL}/debug/count", timeout=30)
        st.json(r.json())

with col2:
    limit = st.number_input("Sample size", min_value=1, max_value=200, value=20, step=5)

if st.button("Fetch sample objects"):
    r = requests.get(f"{BACKEND_URL}/debug/sample", params={"limit": limit}, timeout=30)
    data = r.json()
    st.write(f"Collection: {data.get('collection')}")
    items = data.get("items", [])
    st.dataframe(items, width='stretch')

st.divider()
st.subheader("Search (via your backend /search endpoint)")

q = st.text_input("Question", value="qualité de l air paris")
k = st.slider("k", 1, 20, 5)

if st.button("Search"):
    r = requests.post(f"{BACKEND_URL}/search", json={"question": q, "k": k}, timeout=60)
    st.json(r.json())