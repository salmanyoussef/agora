import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Weaviate Admin", layout="wide")
st.title("Weaviate Admin")

st.caption(f"Backend: {BACKEND_URL}")

# --- Dataset count ---
st.subheader("Datasets in Weaviate")
if st.button("Refresh count"):
    try:
        r = requests.get(f"{BACKEND_URL}/debug/count", timeout=30)
        r.raise_for_status()
        data = r.json()
        st.metric("Number of datasets", data.get("count", 0))
        st.caption(f"Collection: {data.get('collection', '—')}")
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

st.divider()

# --- Sample ---
st.subheader("Sample objects")
limit = st.number_input("Sample size", min_value=1, max_value=200, value=20, step=5)

if st.button("Fetch sample objects"):
    try:
        r = requests.get(f"{BACKEND_URL}/debug/sample", params={"limit": limit}, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.caption(f"Collection: {data.get('collection', '—')}")
        items = data.get("items", [])
        st.dataframe(items, use_container_width=True)
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
