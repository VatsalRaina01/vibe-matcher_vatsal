# 🧠 Vibe Matcher | AI @ Nexora

A mini AI-powered recommendation system that matches a user's **vibe query** (like “energetic urban chic”) with fashion products using **GitHub AI Embeddings (Text Embedding 3 Large)** or a local fallback model.

---

## 🚀 Features
- Uses **GitHub AI’s text-embedding-3-large** for high-quality semantic understanding.
- Fallback to **SentenceTransformer (all-MiniLM-L6-v2)** if API unavailable.
- Computes **cosine similarity** for vibe-to-product matching.
- Auto-caches embeddings for faster reruns.
- Visualizes latency across different queries.

---

## 🧩 Tech Stack
- **Python 3.9+**
- **Azure AI Inference SDK**
- **Sentence Transformers**
- **Matplotlib**, **Pandas**, **scikit-learn**

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/VatsalRaina01/vibe-matcher_vatsal.git
cd vibe-matcher-ai-nexora
