
---

## 🔹 `src/` — Application Layer

This directory contains the core application:

- Backend services
- API logic
- Database integration
- Embedding pipelines
- Retrieval systems
- Infrastructure configuration

A detailed guide explaining how to run and configure the application is available here:

👉 **`src/README.md`**

---

## 🔹 `notebooks/` — Research & Prototyping

This directory contains:

- Embedding experiments
- Retrieval benchmarking
- Data exploration
- Early-stage prototypes
- Model validation notebooks

Notebooks are used to experiment and validate ideas before integrating them into the production system inside `src/`.

---

## 🧠 Design Philosophy

AGORA follows a structured development workflow:

1. Prototype ideas in `notebooks/`
2. Validate experimentally
3. Refactor into production-quality modules in `src/`
4. Maintain clean separation between experimentation and deployment

This ensures stability in production while preserving flexibility for research and iteration.

---

## 🎯 Core Capabilities

- Semantic search via vector embeddings
- Hybrid retrieval (vector + keyword)
- Dataset metadata indexing
- Modular backend architecture
- Extensible embedding pipeline

---

## 📌 Repository Purpose

This repository contains both:

- A deployable semantic search application (`src/`)
- A research environment for continuous improvement (`notebooks/`)

For execution instructions, environment setup, and deployment details, please refer to:

👉 **`src/README.md`**

---

## 📄 License

N/A