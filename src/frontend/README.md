# Agora frontend

Sample UI for the Agora Q&A pipeline over French open data.

## Run

1. Start the backend from `src/backend`:
   ```bash
   uvicorn app.main:app --reload
   ```
2. Open **http://localhost:8000/** in a browser. The backend serves this frontend at `/`.

## General vs Technical mode

Before submitting a question, you must choose a mode:

- **General** — The pipeline uses **only RAG**: dataset resources are downloaded, text is extracted, chunked, and retrieved by semantic similarity; the LLM answers from that context. No technical analysis, no RLM, no computation over structured data. Use this when you want a document-style answer from dataset contents.
- **Technical** — The pipeline uses **both** general and technical analysis **per resource**: for each relevant dataset, the backend’s dataset selector decides whether to run **RAG** (general) or **technical** extraction. Technical resources (e.g. CSV, JSON, XLSX) are parsed into structured form and explored by an RLM (reasoning language model) in a sandboxed REPL; non-structured resources (e.g. PDF) still get text extraction and are included in the technical context. So choosing Technical gives you the best of both: RAG where a text sample is enough, and structured/computational analysis where the data is tabular or record-based.

The chosen mode is sent to the streaming API as `use_only_general_agent: true` (General) or `false` (Technical).

## API base

The page uses the current origin as the API base. To point to another host/port, add in `<head>`:

```html
<meta name="api-base" content="http://localhost:8000" />
```
