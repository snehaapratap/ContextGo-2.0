# ContextGo 2.0

## 📚 AmbedkarGPT: SemRAG-Based Q&A System

A fully functional RAG (Retrieval-Augmented Generation) system based on the **SEMRAG research paper**, designed to answer questions about Dr. B.R. Ambedkar's works.


---

## 🎯 Overview

This project implements the complete **SemRAG (Semantic Knowledge-Augmented RAG)** architecture as described in the research paper:

> **SEMRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering**
> 
> SemRAG enhances conventional RAG by integrating semantic chunking and knowledge graphs for improved contextual understanding and retrieval accuracy.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Semantic Chunking** | Algorithm 1 - Groups sentences by cosine similarity |
| 🔗 **Knowledge Graph** | Entity extraction with relationship mapping |
| 🏘️ **Community Detection** | Leiden/Louvain algorithm for entity grouping |
| 🔍 **Hybrid Search** | Local (Eq.4) + Global (Eq.5) retrieval |
| 🤖 **Local LLM** | Powered by Llama 3.2 via Ollama |
| 🎨 **Beautiful UI** | Modern Streamlit interface |

---

## 📸 Demo Screenshots

### Main Interface

<img width="1470" height="956" alt="Screenshot 2025-12-07 at 20 26 56" src="https://github.com/user-attachments/assets/f703a557-2dee-4fba-a432-5496fe6a3dcd" />

### Question Answering

<img width="1470" height="956" alt="Screenshot 2025-12-07 at 20 27 59" src="https://github.com/user-attachments/assets/4b72eb59-b3a0-4952-8a86-afb572f05242" />

### Query Details

<img width="1470" height="956" alt="Screenshot 2025-12-07 at 20 28 09" src="https://github.com/user-attachments/assets/9e7a82ab-69d7-40f5-8f5d-cd4b522001ac" />

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AmbedkarGPT                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │   Semantic   │   │  Knowledge  │   │     Community       │   │
│  │   Chunking   │──▶│    Graph    │──▶│     Detection       │   │
│  │ (Algorithm 1)│   │   Builder   │   │  (Leiden/Louvain)   │   │
│  └─────────────┘   └─────────────┘   └─────────────────────┘   │
│         │                │                     │                │
│         ▼                ▼                     ▼                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Retrieval Layer                        │   │
│  │  ┌─────────────────┐     ┌─────────────────────────┐    │   │
│  │  │  Local Search   │     │     Global Search       │    │   │
│  │  │  (Equation 4)   │     │     (Equation 5)        │    │   │
│  │  └─────────────────┘     └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                        │                                        │
│                        ▼                                        │
│              ┌─────────────────┐                               │
│              │    LLM Client   │                               │
│              │ (Llama 3.2)     │                               │
│              └─────────────────┘                               │
│                        │                                        │
│                        ▼                                        │
│              ┌─────────────────┐                               │
│              │     Answer      │                               │
│              └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- **Python 3.9+**
- **[Ollama](https://ollama.ai/)** installed and running
- At least **8GB RAM** (16GB recommended)

### Step 1: Navigate to Project

```bash
cd ambedkargpt
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ambedkargpt python=3.10
conda activate ambedkargpt
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 4: Setup Ollama

```bash
# Start Ollama server (in a separate terminal)
ollama serve

# Pull the LLM model
ollama pull llama3.2:latest
```

---

## 🚀 Running the Application

### Option 1: Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Option 2: Command Line Interface

```bash
python run.py
```

### Option 3: Demo Script

```bash
python demo.py --pdf data/Ambedkar_book.pdf --interactive
```

---

## 🎮 How to Use

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Index the Document
- The app will automatically find `Ambedkar_book.pdf`
- Click **"🚀 Start Indexing"** button
- Wait for indexing to complete (~1-2 minutes)

### 3. Ask Questions
Use the sample questions or type your own:
- "What is caste according to Ambedkar?"
- "What does Ambedkar say about endogamy?"
- "What solution does Ambedkar propose for the caste problem?"

### 4. Explore Results
- View the answer in the chat interface
- Expand **"Last Query Details"** to see:
  - Entities found
  - Search type used
  - Source citations

---

## 📁 Project Structure

```
ambedkargpt/
├── app.py                      # 🎨 Streamlit Web UI
├── config.yaml                 # ⚙️ Configuration
├── requirements.txt            # 📦 Dependencies
├── data/
│   ├── Ambedkar_book.pdf       # 📄 Input document
│   └── processed/              # 💾 Saved index
├── src/
│   ├── chunking/
│   │   ├── semantic_chunker.py # ✂️ Algorithm 1
│   │   └── buffer_merger.py    # 🔄 Buffer merging
│   ├── graph/
│   │   ├── entity_extractor.py # 🏷️ NER extraction
│   │   ├── graph_builder.py    # 🔗 Graph construction
│   │   ├── community_detector.py # 🏘️ Leiden/Louvain
│   │   └── summarizer.py       # 📝 LLM summarization
│   ├── retrieval/
│   │   ├── local_search.py     # 🔍 Equation 4
│   │   ├── global_search.py    # 🌐 Equation 5
│   │   └── ranker.py           # 📊 Result ranking
│   ├── llm/
│   │   ├── llm_client.py       # 🤖 Ollama client
│   │   ├── prompt_templates.py # 💬 Prompts
│   │   └── answer_generator.py # ✍️ Answer generation
│   └── pipeline/
│       └── ambedkargpt.py      # 🔧 Main pipeline
├── tests/                      # 🧪 Unit tests
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
models:
  embedding_model: "all-MiniLM-L6-v2"
  llm_model: "llama3.2:latest"
  ner_model: "en_core_web_sm"

chunking:
  buffer_size: 5              # Context window
  similarity_threshold: 0.5   # Cosine distance threshold
  max_tokens: 1024            # Max tokens per chunk
  overlap_tokens: 128         # Overlap for sub-chunks

retrieval:
  local:
    entity_threshold: 0.3     # τ_e (Equation 4)
    document_threshold: 0.3   # τ_d (Equation 4)
    top_k: 5
  global:
    top_k_communities: 3      # Top-K communities (Equation 5)
    top_k_points: 5
```

---

## 📐 Algorithm Details

### Algorithm 1: Semantic Chunking

```
Input: Document D; threshold θ; buffer size b; token limit Tmax
Output: Chunk set C

1. S ← Split(D)                    # Split into sentences
2. Ŝ ← BufferMerge(S, b)           # Contextual merging
3. Z ← {zi = Embed(ŝi)}            # LLM embeddings
4. for i = 1 to |Ŝ| - 1 do
5.   di ← 1 - cos(zi, zi+1)        # Cosine distance
6. Group sentences where di < θ
7. Split large chunks with 128 token overlap
8. return C
```

### Equation 4: Local Graph RAG Search

```
D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
```

### Equation 5: Global Graph RAG Search

```
D_retrieved = Top_k(⋃_{r∈R_Top-K(Q)} ⋃_{c_i∈C_r} (⋃_{p_j∈c_i} (p_j, score(p_j, Q))))
```

---

## 📊 Sample Results

| Metric | Value |
|--------|-------|
| Semantic Chunks | 66 |
| Entities Extracted | 566 |
| Graph Nodes | 566 |
| Graph Edges | 794 |
| Communities Detected | 209 |
| Community Summaries | 72 |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_chunking.py -v
pytest tests/test_retrieval.py -v
```

---

## 💡 Sample Questions

Try these questions in the demo:

1. **"What is caste according to Dr. Ambedkar?"**
2. **"What does Ambedkar say about endogamy?"**
3. **"How does the caste system maintain itself?"**
4. **"What is Ambedkar's view on inter-caste marriage?"**
5. **"What solution does Ambedkar propose for annihilating caste?"**
6. **"How does Ambedkar define the mechanism of caste?"**
7. **"What does Ambedkar say about Varna vs Caste?"**

---

## 🔧 Troubleshooting

### Ollama Not Running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Model Not Found
```bash
ollama pull llama3.2:latest
```

### spaCy Model Missing
```bash
python -m spacy download en_core_web_sm
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8503
```

---

## 📝 Submission Checklist

- [x] Working semantic chunking on Ambedkar_book.pdf
- [x] Knowledge graph with entities and relationships
- [x] Local search (Equation 4) implemented
- [x] Global search (Equation 5) implemented
- [x] Community detection (Leiden algorithm)
- [x] LLM answering questions
- [x] Clean, modular code with tests
- [x] Configurable parameters
- [x] Beautiful Streamlit UI
- [x] README with setup instructions
- [x] Demo ready for live presentation

---

##  Acknowledgments

- Based on the **SEMRAG** research paper by Kezhen Zhong et al.
- Uses Dr. B.R. Ambedkar's "Castes in India" for demonstration
- Built with sentence-transformers, spaCy, NetworkX, and Ollama


