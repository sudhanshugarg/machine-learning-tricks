# Hospital ICD Prediction System - Quick Summary

## The Problem

A hospital receives 15-20 documents per patient encounter (PDFs, images, text files, scans, etc.) and needs to automatically predict the relevant ICD codes from a catalog of ~100,000 codes. These codes are used for billing, medical records, and research.

**Challenges:**
- ❌ Can't use standard softmax (100k classes is too large)
- ❌ Multi-modal documents (PDF, images, text require different processing)
- ❌ Medical domain knowledge needed (BioBERT/medical context)
- ❌ Patient history matters (past diagnoses influence current ones)
- ❌ Must be fast (< 30 seconds end-to-end)

## The Solution

### Architecture: Two-Stage Retrieval-Ranking

```
Documents → [STAGE 1: RETRIEVE] → Top-500 candidates
                                    ↓
                            [STAGE 2: RANK] → Final predictions
```

**Why two stages?**
- Retrieval narrows from 100k codes to 500 (fast, coarse)
- Ranking uses expensive model on small set (accurate, fine)
- Much faster than ranking all 100k codes at once

### Component Breakdown

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Document Processor** | Extract text from PDFs, images, etc. | PaddleOCR, PDF parsers |
| **Embedding Service** | Convert text to 768-dim vectors | BioBERT (medical domain) |
| **Dense Retrieval** | Find semantically similar codes | FAISS/Milvus + HNSW index |
| **Sparse Retrieval** | Find keyword matches | BM25 (Elasticsearch) |
| **Hybrid Retrieval** | Combine both approaches | Weighted ensemble |
| **Ranker Model** | Score top-500 codes | Cross-encoder Transformer |
| **Confidence Estimator** | Estimate prediction confidence | Ensemble (model + evidence + history) |
| **Patient History** | Fetch past diagnoses | PostgreSQL + Redis cache |

### Data Flow

```
Patient encounter upload
    ↓
Validate & store documents in S3
    ↓
Extract text from each document
    ↓
Generate embeddings (BioBERT)
    ↓
Dense retrieval (vector similarity) → Top-200 codes
+ Sparse retrieval (BM25) → Top-100 codes
    ↓
Hybrid ensemble → Top-500 deduplicated
    ↓
Fetch patient history (past diagnoses)
    ↓
Score all 500 with cross-encoder ranker
    ↓
Filter by confidence threshold
    ↓
Estimate confidence (model + evidence + history)
    ↓
Return Top-20 predictions with explanations
```

## Key Design Decisions

### 1. Why Retrieval-Ranking Instead of Direct Softmax?
**Direct approach:** documents → 100k softmax → predictions ❌ Too slow, memory intensive
**Our approach:** documents → retrieve 500 → rank 500 ✅ Fast, modular, scalable

### 2. Why Hybrid (Dense + Sparse) Retrieval?
**Dense only:** Great semantics, misses exact keyword matches
**Sparse only:** Good keywords, no semantic understanding
**Hybrid:** Both benefits, captures more candidate codes ✅

### 3. Why BioBERT?
**General BERT:** Not trained on medical literature
**BioBERT:** Pre-trained on medical papers ✅ Better medical terminology understanding
**Clinical-BERT:** Too specialized, less data
**GPT-based:** Too slow for 30s budget

### 4. Why Cross-Encoder for Ranking?
**Bi-encoder:** Pre-compute embeddings, fast inference (but lower quality)
**Cross-encoder:** Compute at inference time, slow but much higher quality ✅
**Acceptable because:** Only ranking 500 codes (not 100k)

### 5. Why Explicit Patient History?
**No history:** Missing important context
**Implicit (learned):** More robust but requires lots of training data
**Explicit embedding:** Interpretable, practical, works with limited data ✅

## Performance Breakdown

**Target:** < 30 seconds end-to-end
**Actual:** ~15-25 seconds (with margin for growth)

| Step | Time |
|------|------|
| File validation | 1s |
| Document processing (OCR) | 2-3s |
| Text extraction | 0.5s |
| Embedding generation | 5-8s |
| Dense + sparse retrieval | 1.5s |
| Patient history lookup | 0.5s |
| Ranking (500 codes) | 3-5s |
| Confidence & DB writes | 1.5s |
| **Total** | **~15-25s** |

## Key Metrics

### Model Performance
- **Precision@10:** Of top-10 predictions, how many are correct?
- **Recall@100:** What % of true codes appear in top-100?
- **MRR (Mean Reciprocal Rank):** Average rank of first correct code
- **NDCG:** Ranking quality (considers partial relevance)

### System Performance
- **Latency:** Time from upload to predictions (target: < 30s)
- **Throughput:** Encounters per day (~100-200 with single GPU)
- **Physician adoption:** % of predictions accepted

## Implementation Roadmap

### Phase 1: MVP (Week 1-2) - Core Pipeline
- Document processing for all file types
- BioBERT embedding service
- BM25 retrieval
- Simple ranking baseline

### Phase 2: System (Week 3-4) - Full Integration
- Vector DB setup
- Patient history integration
- Confidence scoring
- Web UI for predictions

### Phase 3: Optimization (Week 5-6)
- Caching strategy
- Performance tuning
- A/B testing framework
- Monitoring & alerting

### Phase 4: Advanced (Week 7+)
- Hierarchical prediction (ICD chapters first)
- Multi-head model (diagnoses, procedures, complications)
- Continuous learning from physician feedback
- Knowledge graph integration

## File Guide

- **README.md** - Overview and quick start
- **design.md** - High-level design (requirements, challenges, solutions)
- **architecture.md** - Technical details (components, databases, flows)
- **tradeoffs.md** - Design decisions and alternatives considered
- **template.py** - Python code skeleton to implement

## What Makes This Design Good

✅ **Scalable:** Handles 100k codes without softmax over all
✅ **Practical:** 15-25s latency fits real hospital workflows
✅ **Interpretable:** Can explain each prediction (which docs support it)
✅ **Modular:** Each component can be improved independently
✅ **Medical-aware:** Uses domain-specific pre-trained models
✅ **Contextual:** Incorporates patient history
✅ **Safe:** Filters by confidence, provides explanations for physician review

## What to Explore Next

1. **Implement document processing pipeline** - Handle all file types
2. **Set up vector database** - Index code embeddings for fast retrieval
3. **Train ranking model** - Fine-tune cross-encoder on labeled data
4. **Physician feedback loop** - Collect corrections for continuous improvement
5. **A/B testing** - Compare different model versions with real physicians
6. **Cost optimization** - Caching, indexing, batch processing

---

**Total Size of Design:** ~55KB across 5 files covering architecture, tradeoffs, and implementation
