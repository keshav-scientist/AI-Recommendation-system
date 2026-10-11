# AI Recommendation & Semantic Retrieval System

A machine learning and Generative AI repository implementing classical recommendation engines, matrix factorization, vector search databases, and multimodal retrieval workflows.

---

## 📌 Repository Overview

This project bridges traditional recommendation system architectures with modern dense vector embeddings and retrieval techniques:

### 1. Classical Recommendation Systems (`Movie_Recommedation.ipynb`)
Built on the **MovieLens 25M** dataset (`movies.csv`, `ratings.csv` with 62k+ titles, 25M+ ratings, and 99.74% matrix sparsity):
- **Popularity-Based Filtering (IMDb Weighted Rating):**
  Uses a Bayesian mean formula to balance movie rating volume against average scores:
  $$\text{Weighted Score} = \left(\frac{v}{v+m} \cdot R\right) + \left(\frac{m}{v+m} \cdot C\right)$$
  *(where $v$ = vote count, $m$ = 90th percentile threshold, $R$ = mean rating, $C$ = global mean).*
- **Content-Based Filtering:**
  - Extracts genre metadata features via `TfidfVectorizer`.
  - Computes pairwise cosine similarity using `linear_kernel`.
  - Employs `np.argpartition` for fast top-$K$ index selection without full-matrix memory overhead.
- **Collaborative Filtering:**
  - Implements Matrix Factorization via Singular Value Decomposition (**SVD**) using `scikit-surprise`.

---

### 2. Generative AI & Semantic Retrieval (`Gen_AI_3.ipynb`, `gen_ai_5.ipynb`)
- **Dense Text Embeddings:** Generates sentence-level embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Multimodal Search:** Employs OpenAI's CLIP architecture (`clip-vit-B-32`) to map text and images into a shared semantic latent space for cross-modal similarity lookup.
- **Vector Indexing & Persistence:** Fast $L_2$ nearest-neighbor indexing via **FAISS** (`IndexFlatL2`) with local disk serialization and structured vector collection management via **ChromaDB**.
- **LLM Workflows & Advanced Retrieval:** Explores prompt workflows, semantic document indexing, and generative ranking.

---

## 📁 Repository Structure

```text
AI-Recommendation-system/
├── Movie_Recommedation.ipynb   # MovieLens EDA, Weighted Popularity, TF-IDF, and SVD
├── Gen_AI_3.ipynb              # Vector search, FAISS/ChromaDB indexing, CLIP embeddings
├── gen_ai_5.ipynb              # Advanced GenAI retrieval, ranking, and LLM workflows
└── README.md                   # Project documentation
