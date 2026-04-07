# Hospital ICD Prediction System - Design Tradeoffs

## 1. Retrieval-Ranking vs. Direct Prediction

### Direct Prediction (Single Model)
```
Documents → Single Neural Model → 100k probabilities → Predictions
```

**Pros:**
- Simple end-to-end training
- Single inference pass
- Easier to maintain

**Cons:**
- Softmax over 100k classes is memory-intensive
- All 100k parameters active for every prediction
- Slow inference (~20s per sample)
- Hard to scale to more codes
- **REJECTED:** Not practical for this scale

### Retrieval-Ranking Approach (Two-Stage)
```
Documents → Retrieval → Top-500 candidates → Ranking → Final predictions
```

**Pros:**
- Scalable to any number of codes
- Fast inference (5-15s)
- Separable components (can improve independently)
- Can use different models for retrieval vs ranking
- Retrieval can leverage both sparse and dense methods

**Cons:**
- Two-stage pipeline more complex
- Retrieval errors cascade to ranking
- Needs careful calibration of retrieval threshold

**CHOSEN:** Two-stage retrieval-ranking

---

## 2. Dense vs. Sparse Retrieval

### Dense Retrieval Only (Vector Similarity)
```
Document embedding → Find similar code embeddings → Top-K codes
```

**Pros:**
- Semantic similarity captures meaning
- Single index (Milvus/FAISS)
- Fast (~1s for 100k codes)

**Cons:**
- May miss exact keyword matches
- OOD (out-of-distribution) documents may not embed well
- Requires good pre-trained embeddings
- Medical terminology might not be well-represented

### Sparse Retrieval Only (BM25)
```
Document text → Keyword matching → Top-K codes
```

**Pros:**
- Handles exact matches perfectly
- Interpretable (which keywords matched)
- Doesn't require embeddings

**Cons:**
- No semantic understanding
- Medical synonyms/abbreviations cause misses
- Slower on large vocabularies

### Hybrid Approach
```
         Documents
         ├─→ Dense Retrieval (Top-200)
         └─→ BM25 Retrieval (Top-100)
              └─→ Union + Rerank (Top-500)
```

**Pros:**
- Combines strengths of both
- Semantic + keyword coverage
- More robust to different document types

**Cons:**
- More complex system
- Need to tune ensemble weights
- Slower than dense alone (~1.5s vs 1s)

**CHOSEN:** Hybrid dense + sparse retrieval

---

## 3. Pre-trained Models

### General BERT vs. Medical BERT

**General BERT (bert-base-uncased)**
```
Pros:
- Widely available
- Fast inference (smaller)
- Good for general text

Cons:
- Not trained on medical literature
- May miss domain-specific terms
- Lower quality medical document representations
```

**Medical BERT (BioBERT, PubMedBERT)**
```
Pros:
- Pre-trained on medical papers
- Better understanding of medical terminology
- Higher quality embeddings for medical text

Cons:
- Slightly larger (more memory)
- Less common (fewer tutorials/tools)
- May overfit if not enough medical data
```

**GPT-based (Clinical-BERT, Clinical-LLaMA)**
```
Pros:
- Better semantic understanding
- Can generate explanations
- Handles long documents better

Cons:
- Much larger (expensive inference)
- Slower inference (not suitable for 30s constraint)
- Requires API call (cloud dependency)
```

**CHOSEN:** BioBERT (middle ground between quality and efficiency)

---

## 4. Single Model vs. Hierarchical Prediction

### Single-Stage Ranking
```
Select all 100k codes as candidates → Rank all → Top-20 predictions
```

**Pros:**
- Simple architecture
- No cascade errors
- Can capture inter-code relationships

**Cons:**
- Ranking 100k codes is slow
- Wastes compute on irrelevant codes
- High variance in predictions

### Hierarchical (Two-Level) Prediction
```
Level 1: Predict ICD chapter (20 categories)
    ↓
Level 2: Predict specific codes within chapter (~5k codes per chapter)
```

**Pros:**
- Search space reduced 5x
- Faster ranking (500 codes/chapter vs 100k)
- Better precision (focus on relevant chapter)
- Aligns with ICD code structure

**Cons:**
- Errors at level 1 cascade to level 2
- Some codes may be missed if wrong chapter selected
- More complex system (two models)

**CHOSEN:** Single-stage ranking with optional hierarchical fallback

**Reasoning:**
- ICD codes have overlaps across chapters (conditions can be categorized multiple ways)
- Hierarchical might miss correct codes due to wrong chapter selection
- Better to rank all 100k with efficient indexing than risk missing codes

---

## 5. Patient History Integration

### No History (Context-Free)
```
Current documents only → Predict ICD codes
```

**Pros:**
- Simpler system
- No need for patient history DB
- Fresh start for each encounter

**Cons:**
- Misses important context
- Can't leverage pattern of recurring conditions
- Worse predictions for chronic diseases

### Explicit History Embedding
```
Patient past codes → Embed → Concatenate with document embedding → Predict
```

**Pros:**
- Explicitly incorporates history
- Can learn history-aware patterns
- Flexible (can weight recent codes more)

**Cons:**
- Adds complexity
- History DB lookups add latency
- History might be wrong/incomplete
- Risk of over-relying on history

### Implicit History Learning
```
Train model to learn history from documents + past encounter data
```

**Pros:**
- Model learns when history is relevant
- More robust to noisy history
- Implicit relationships captured

**Cons:**
- Requires historical training data
- Harder to interpret
- More training time

**CHOSEN:** Explicit history embedding (fast, interpretable, practical)

---

## 6. Confidence Scoring Approaches

### Model Score Only
```
Use ranker's sigmoid output as confidence
```

**Pros:**
- Simple
- Fast

**Cons:**
- Uncalibrated (may overestimate/underestimate)
- Doesn't account for evidence strength
- Same score for different evidence types

### Ensemble Confidence
```
confidence = α × model_score + β × evidence_strength + γ × history_agreement
```

**Where:**
- model_score: Ranker's output (0-1)
- evidence_strength: # documents / total documents
- history_agreement: Code appears in patient history?

**Pros:**
- Captures multiple factors
- More interpretable
- Accounts for evidence quality

**Cons:**
- Need to tune weights
- More computation
- Harder to explain to physicians

**CHOSEN:** Ensemble approach (better reflects prediction quality)

---

## 7. Physician Feedback Loop

### Feedback-Only Approach
```
Collect physician corrections → Store → Use for retraining (weekly batch)
```

**Pros:**
- Simple to implement
- Decoupled from predictions
- No risk of overfitting to individual case

**Cons:**
- Delayed feedback (retraining weekly)
- Can't personalize to individual physicians
- Biased toward high-volume hospitals

### Real-Time Personalization
```
Store corrections → Use to adjust model weights online → Immediate improvement
```

**Pros:**
- Immediate adaptation
- Can catch model drift quickly

**Cons:**
- Risk of overfitting
- Instability if corrections are noisy
- Hard to debug issues
- May diverge from gold-standard labels

### Hybrid Approach
```
Collect corrections → Store locally → Use for inference-time boosting
                  → Batch for retraining (weekly)
```

**Pros:**
- Immediate local adaptation
- Periodic formal retraining for generalization
- Safety buffer before global deployment

**Cons:**
- More complex system
- Need to manage local overrides

**CHOSEN:** Hybrid approach

---

## 8. Latency vs. Accuracy Tradeoff

### Speed-Optimized
```
Retrieval: Top-200 candidates (0.5s)
Ranking: Lightweight model (1s)
Total: ~2-3s
```

**Pros:**
- Fast response to physicians
- Low latency for real-time use

**Cons:**
- May miss correct codes
- Lower recall

### Accuracy-Optimized
```
Retrieval: Top-1000 candidates (2s)
Ranking: Heavy model + ensemble (10s)
Total: ~12-15s
```

**Pros:**
- Higher recall
- Better prediction quality

**Cons:**
- Takes longer
- Still acceptable (< 30s target)
- Higher compute cost

### Adaptive Approach
```
Return fast results (2s) immediately
Continue ranking in background
Update UI when better results available
```

**Pros:**
- Best of both worlds
- Physicians see something immediately
- Can improve over time

**Cons:**
- Complex to implement
- Physicians might not review updates
- Multiple predictions confusing

**CHOSEN:** Accuracy-optimized (15-20s is acceptable, better for patient safety)

---

## 9. Centralized vs. Distributed Models

### Single Centralized Model
```
All hospitals → Single ML model in cloud → Predictions
```

**Pros:**
- Economies of scale
- Consistent quality across hospitals
- Easier to maintain

**Cons:**
- Network latency (important for real-time)
- Single point of failure
- Privacy concerns (data in cloud)
- Worst-case latency spike

### Hospital-Local Models
```
Each hospital → Local ML model → Predictions
```

**Pros:**
- No network latency
- Data stays local (HIPAA friendly)
- Fault tolerance
- Can personalize to hospital

**Cons:**
- Model duplication
- Hard to maintain (20+ hospitals = 20 models)
- Less data per hospital for training
- Higher infrastructure cost

### Hybrid Approach
```
Cloud: Model training, optimization, monitoring
Hospitals: Lightweight inference servers (model copying)
Feedback: Bidirectional (local corrections → cloud for retraining)
```

**Pros:**
- Best of both worlds
- Local inference (no latency)
- Centralized improvements

**Cons:**
- Complex architecture
- Need to manage model versioning
- Slower feedback loop

**CHOSEN:** Hybrid approach (inference local, training central)

---

## 10. Technology Choices

### Vector Database: Milvus vs. FAISS vs. Pinecone

| Feature | Milvus | FAISS | Pinecone |
|---------|--------|-------|----------|
| **Scaling** | Built-in | Manual sharding | Managed |
| **Latency** | 100ms | <10ms | 100-200ms |
| **Cost** | Self-hosted (high ops) | Self-hosted (high ops) | Managed (easy) |
| **Filtering** | Good | Limited | Good |
| **Reliability** | High | Moderate | High (SLA) |

**CHOSEN:** Milvus (self-hosted for cost, good filtering/reliability)

### Document Processing: PaddleOCR vs. Tesseract vs. Cloud Vision API

| Tool | Accuracy | Speed | Cost |
|------|----------|-------|------|
| **Tesseract** | 80% | Fast | Free |
| **PaddleOCR** | 90% | Medium | Free |
| **Cloud Vision** | 95% | Slow | Expensive |

**CHOSEN:** PaddleOCR (good balance, free, open-source)

### Ranking Model: Cross-Encoder vs. Bi-Encoder

| Architecture | Speed | Quality | Index |
|---|---|---|---|
| **Bi-Encoder** | Fast (pre-computed embeddings) | Good | Large |
| **Cross-Encoder** | Slow (compute at inference) | Better | Not needed |

**CHOSEN:** Cross-Encoder for ranking (better quality, latency acceptable)

---

## Summary Table: Key Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Retrieval Strategy | Two-stage (retrieval + ranking) | Scalable, fast, modular |
| Retrieval Method | Hybrid (dense + BM25) | Semantic + keyword coverage |
| Document Encoder | BioBERT | Medical domain knowledge |
| Prediction Strategy | Single-stage ranking | Avoid hierarchical errors |
| History Integration | Explicit embedding | Interpretable, practical |
| Confidence Scoring | Ensemble approach | Multiple signals |
| Physician Feedback | Hybrid loop | Immediate + periodic improvement |
| Latency Target | 15-20s (accuracy-optimized) | Safety > speed |
| Infrastructure | Hybrid (central training, local inference) | Privacy + scalability |

