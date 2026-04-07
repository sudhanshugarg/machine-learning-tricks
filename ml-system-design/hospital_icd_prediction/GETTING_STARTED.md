# Getting Started with Hospital ICD Prediction System

## Quick Navigation

This folder contains a complete ML system design for hospital patient encounter ICD code prediction. Here's where to start based on your role:

### 👨‍💼 If You're an Interviewer
Start with **SUMMARY.md** for a quick overview, then **design.md** for the big picture.

### 🤔 If You're Understanding the System
1. Read **SUMMARY.md** (5 min) - Problem and high-level solution
2. Read **design.md** (15 min) - Requirements and challenges
3. Skim **architecture.md** (10 min) - Technology choices
4. Review **tradeoffs.md** (10 min) - Design decisions

Total: ~40 minutes to understand the full system

### 💻 If You're Implementing
1. Start with **template.py** - Code skeleton with TODOs
2. Review **architecture.md** for component specs
3. Check **tradeoffs.md** for reasoning on each choice
4. Implement in phases (see ROADMAP section below)

## The 30-Second Version

**Problem:** Hospital needs to predict ICD medical codes from 15-20 patient documents (100k possible codes)

**Key Insight:** Can't use standard ML (softmax over 100k classes). Use two-stage approach:
1. **Retrieval:** Narrow from 100k codes to 500 candidates (fast, uses dense + sparse search)
2. **Ranking:** Score 500 codes with transformer model (accurate, feasible)

**Result:** Fast (15-25s), accurate, interpretable predictions with confidence scores

## Why This Design Works

```
Challenge              Solution           Why It Works
─────────────────────  ────────────────   ────────────────────
100k classes           Embedding-based    Can score any # of codes
                       ranking            without softmax

Multi-modal docs       Separate encoders  Each doc type processed
                       + fusion           appropriately

Medical knowledge      BioBERT pre-training  Better medical text
                       + history context     understanding

Patient history        Explicit embedding    Interpretable, practical
                       (past diagnoses)      with limited data

Speed constraint       Two-stage pipeline   Retrieval fast, ranking
(< 30s)                                    on small set

Physician trust        Evidence + scores    Explainable decisions
```

## Architecture at a Glance

```
┌─────────────────────────────────┐
│   Patient Documents (15-20)     │
│   (PDFs, images, text)          │
└──────────┬──────────────────────┘
           │
    ┌──────▼──────────────────────┐
    │  Document Processing        │
    │  (OCR, PDF parsing, etc.)   │
    └──────┬──────────────────────┘
           │
    ┌──────▼──────────────────────┐
    │  Embedding Generation       │
    │  (BioBERT: 768-dim vectors) │
    └──────┬──────────────────────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │  STAGE 1: RETRIEVE (Fast)                   │
    │  ┌─────────────────┬──────────────────┐    │
    │  │ Dense Retrieval │ Sparse Retrieval │    │
    │  │ (Vector sim)    │ (BM25 keywords)  │    │
    │  └─────────┬───────┴────────┬─────────┘    │
    │            │                │              │
    │  Top-200 + Top-100 = Top-500 Candidates   │
    └──────┬──────────────────────────────────────┘
           │
    ┌──────▼──────────────────────┐
    │  Patient History Lookup     │
    │  (Past diagnoses)           │
    └──────┬──────────────────────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │  STAGE 2: RANK (Accurate)                   │
    │  Cross-Encoder Transformer Model            │
    │  Scores all 500 candidates                  │
    └──────┬──────────────────────────────────────┘
           │
    ┌──────▼──────────────────────┐
    │  Confidence Estimation      │
    │  (Model score + evidence)   │
    └──────┬──────────────────────┘
           │
    ┌──────▼──────────────────────┐
    │  Top-20 Predictions         │
    │  + Confidence Scores        │
    │  + Supporting Evidence      │
    └──────────────────────────────┘
```

## Key Components

### 1. Document Processor
**What:** Extract text from PDFs, images, text files
**How:** PDFs → text extraction, Images → OCR (PaddleOCR), Text → chunking
**Output:** Text chunks (512 tokens each)

### 2. Embedding Service
**What:** Convert text chunks to dense vectors
**How:** BioBERT model (pre-trained on medical literature)
**Output:** 768-dimensional vectors

### 3. Retrieval Module
**What:** Find top-500 relevant ICD codes from 100k
**How:** 
  - Dense retrieval: Vector similarity search (FAISS/Milvus)
  - Sparse retrieval: Keyword matching (BM25/Elasticsearch)
  - Combine both for coverage
**Output:** Top-500 candidate codes

### 4. Ranking Model
**What:** Score 500 candidate codes for relevance
**How:** Cross-encoder Transformer (efficient for 500 items)
**Output:** Relevance scores [0, 1]

### 5. Confidence Estimator
**What:** Estimate confidence in each prediction
**How:** Combine model score + evidence strength + history agreement
**Output:** Confidence scores [0, 1]

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Text Encoding** | BioBERT | Medical domain pre-training |
| **Dense Search** | FAISS/Milvus | Fast ANN with filtering |
| **Sparse Search** | BM25/Elasticsearch | Keyword matching |
| **Ranking** | Transformer (PyTorch) | State-of-the-art quality |
| **Storage** | PostgreSQL + S3 | HIPAA-compliant, scalable |
| **Caching** | Redis | Fast history lookups |
| **API** | FastAPI | Async support for 30s latency |

## Performance Budget (30s total)

```
Activity                              Time    Margin
──────────────────────────────────────────────────
File validation & storage            1s
OCR & text extraction                2-3s
Embedding generation (BioBERT)       5-8s
Dense + Sparse retrieval             1.5s
Patient history lookup               0.5s
Ranking 500 codes (Transformer)      3-5s
Confidence & output formatting       1.5s
──────────────────────────────────────────────────
TOTAL                                15-25s    ✅ 5-15s buffer
```

## Implementation Roadmap

### Phase 1: Core Pipeline (Week 1-2)
**Goal:** End-to-end system working with dummy models

- [ ] Document processing for all file types (PDF, OCR, text)
- [ ] BioBERT embedding service
- [ ] BM25 retrieval index
- [ ] Simple baseline ranker (TF-IDF + logistic regression)
- [ ] End-to-end integration + basic API

**Deliverable:** System accepts documents, returns predictions (accuracy TBD)

### Phase 2: Production System (Week 3-4)
**Goal:** Full system with real components

- [ ] Vector database setup (Milvus)
- [ ] Patient history database (PostgreSQL + Redis)
- [ ] Production ranking model (cross-encoder)
- [ ] Confidence scoring
- [ ] Web UI for physician review
- [ ] Logging & monitoring

**Deliverable:** Ready for pilot with real hospital data

### Phase 3: Optimization (Week 5-6)
**Goal:** Performance tuning and quality improvement

- [ ] Caching strategy (patient embeddings, code embeddings)
- [ ] Performance profiling & optimization
- [ ] A/B testing framework
- [ ] Physician feedback collection
- [ ] Continuous learning pipeline

**Deliverable:** < 20s latency, physician feedback loop working

### Phase 4: Advanced Features (Week 7+)
**Goal:** Improved accuracy and flexibility

- [ ] Hierarchical prediction (ICD chapters first)
- [ ] Multi-head model (diagnoses, procedures, complications)
- [ ] Knowledge graph integration
- [ ] Personalization per hospital/physician
- [ ] Auto-scaling to multi-hospital

**Deliverable:** State-of-the-art accuracy, hospital-customizable

## Design Decisions Explained

See **tradeoffs.md** for detailed reasoning, but here are the key ones:

### 1. Two-Stage (Retrieval + Ranking) Instead of Direct Classification
❌ **Bad:** Softmax over 100k classes → memory intensive, slow
✅ **Good:** Retrieve 500 → rank 500 → fast and modular

### 2. Hybrid (Dense + Sparse) Retrieval Instead of One or the Other
❌ **Dense only:** Misses exact keyword matches
❌ **Sparse only:** No semantic understanding
✅ **Both:** Covers both semantic and keyword cases

### 3. BioBERT Instead of General BERT or GPT
❌ **General BERT:** Not trained on medical literature
❌ **GPT:** Too slow (can't fit in 30s budget)
✅ **BioBERT:** Medical pre-training + speed

### 4. Cross-Encoder for Ranking (Not Bi-Encoder)
❌ **Bi-encoder:** Fast inference but lower quality
✅ **Cross-encoder:** Higher quality, acceptable speed (500 codes not 100k)

### 5. Explicit Patient History Embedding
❌ **No history:** Loses important context
❌ **Implicit learning:** Needs lots of training data
✅ **Explicit:** Interpretable, practical with limited data

## Evaluation Metrics

### Model Metrics
- **Precision@10:** "Of the top-10 predictions, how many are correct?"
- **Recall@100:** "Can we find the true codes in top-100?"
- **MRR (Mean Reciprocal Rank):** "What's the average rank of the first correct code?"
- **NDCG:** "How good is our ranking?" (considers partial relevance)

### System Metrics
- **Latency:** Time from document upload to predictions (target: < 30s)
- **Throughput:** Encounters per day (target: 100-200)
- **Physician Adoption:** % of predictions accepted without change

### Business Metrics
- **Time Savings:** Hours saved per hospital per day
- **Cost Reduction:** Labor cost reduction
- **Accuracy:** Medical record accuracy improvement

## Common Questions

**Q: Why two stages instead of one big model?**
A: Softmax over 100k classes doesn't fit in memory and is slow. Two stages is the industry standard for extreme multi-class (100k+).

**Q: Why BioBERT instead of ChatGPT?**
A: Speed. ChatGPT can't process documents in the 30s budget. BioBERT is 100x faster.

**Q: What if a code isn't retrieved in top-500?**
A: Rare, but possible. We mitigate with hybrid retrieval (dense + sparse) and can fall back to hierarchy-based search.

**Q: How do we handle new ICD codes?**
A: Automatically - embeddings don't require retraining. Just add code to vector DB.

**Q: Can physicians override predictions?**
A: Yes! Collect feedback, use for continuous learning and model retraining.

## Next Steps

1. **Explore the design:** Start with SUMMARY.md, then architecture.md
2. **Understand tradeoffs:** Read tradeoffs.md for each design decision
3. **Plan implementation:** Use template.py as starting point
4. **Start coding:** Implement in phases (see roadmap above)
5. **Evaluate:** Measure against metrics, iterate

---

**Questions?** Check tradeoffs.md for detailed reasoning on each design choice.
