# Hospital Patient Encounter ICD Prediction System Design

## Problem Statement

Design an ML system that predicts ICD (International Classification of Diseases) codes for patient encounters. The system must:

- **Input:** 15-20 documents of varying types (images, PDFs, text)
- **Context:** Patient medical history
- **Output:** Predicted ICD codes from a catalog of ~100,000 codes
- **Scale:** Hospital-wide patient encounters

---

## 1. High-Level Requirements

### Functional Requirements

1. **Multi-Modal Document Processing**
   - Handle multiple document types: images, PDFs, text files
   - Extract relevant information from each document type
   - Preserve document context and relationships

2. **ICD Code Prediction**
   - Predict multiple ICD codes (multi-label classification)
   - Handle ~100,000 possible ICD codes (extreme multi-class problem)
   - Support hierarchical ICD code relationships

3. **Patient History Integration**
   - Incorporate patient's medical history (previous encounters, diagnoses)
   - Consider temporal aspects (recency, frequency of past conditions)
   - Handle missing or incomplete history

4. **Real-Time Processing**
   - Process patient encounters within acceptable time (< 30 seconds)
   - Return predictions in real-time to physicians
   - Support async processing for batch uploads

5. **Interpretability & Confidence**
   - Provide confidence scores for each predicted code
   - Explain which documents contributed to each prediction
   - Support physician review and correction

### Non-Functional Requirements

1. **Accuracy & Reliability**
   - High precision (false positives are costly in medical context)
   - High recall (missing diagnoses has serious consequences)
   - Balanced approach with configurable thresholds

2. **Scalability**
   - Handle peak hospital load (100s of encounters/day)
   - Support growing ICD code catalog
   - Efficient document storage and retrieval

3. **Security & Compliance**
   - HIPAA compliance for patient data
   - Encrypted document storage
   - Audit trails for all predictions

4. **Performance**
   - Document processing: < 5 seconds per document
   - Model inference: < 20 seconds for all codes
   - End-to-end: < 30 seconds

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│           Patient Portal / User Interface               │
│        (Upload documents, review predictions)           │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼─────┐         ┌────▼──────┐
   │ Document │         │   Patient │
   │ Processing├─────────│   History │
   │ Pipeline  │         │   Lookup  │
   └────┬─────┘         └──────┬────┘
        │                      │
        └──────────┬───────────┘
                   │
            ┌──────▼──────┐
            │  Vector DB  │
            │ (Document   │
            │ Embeddings) │
            └──────┬──────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼──────────┐  ┌──────▼────────┐
   │ Retrieval     │  │ Neural ICD    │
   │ Module        │  │ Prediction    │
   │ (BM25/Vector) │  │ Model         │
   └────┬──────────┘  └──────┬────────┘
        │                    │
        └──────────┬─────────┘
                   │
            ┌──────▼──────────┐
            │ Prediction      │
            │ Ranking &       │
            │ Filtering       │
            └──────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Confidence Scoring  │
        │ & Explanation       │
        └──────┬──────────────┘
               │
        ┌──────▼──────────┐
        │  UI Rendering   │
        │  (Ranked List   │
        │   of Codes)     │
        └─────────────────┘
```

---

## 3. Key Design Challenges

### 3.1 Extreme Multi-Class Problem
- **Challenge:** 100,000 possible classes is too large for standard softmax
- **Solutions:**
  1. **Hierarchical Classification:** Use ICD code hierarchy (chapters → sections → codes)
  2. **Two-Stage Approach:** First predict ICD chapters, then specific codes
  3. **Embedding-Based:** Learn embeddings for codes, use similarity-based ranking
  4. **Retrieve-and-Rank:** Use retrieval to narrow candidates, then rank

### 3.2 Multi-Modal Input
- **Challenge:** Different document types contain different information
- **Solutions:**
  1. **Separate Encoders:** Use specialized encoders for each modality
  2. **Unified Embeddings:** Convert all modalities to common embedding space
  3. **Cross-Modal Attention:** Learn relationships between different document types
  4. **Document Importance:** Learn which document types matter for which codes

### 3.3 Temporal & Contextual Information
- **Challenge:** Patient history influences current encounter
- **Solutions:**
  1. **History Encoding:** Embed patient's past diagnoses
  2. **Temporal Weighting:** Give more weight to recent encounters
  3. **Sequence Models:** Use RNNs/Transformers to capture temporal patterns
  4. **Context Gates:** Learn when history is relevant vs. noise

### 3.4 Medical Domain Knowledge
- **Challenge:** Model needs to understand medical relationships
- **Solutions:**
  1. **Knowledge Graphs:** Incorporate medical ontologies
  2. **Pre-trained Models:** Use BioBERT, PubMedBERT for text
  3. **Domain-Specific Embeddings:** Train on medical literature
  4. **Expert Rules:** Layer in clinical decision rules

---

## 4. Proposed Solution Architecture

### Phase 1: Core Prediction Pipeline

```
INPUT DOCUMENTS
        │
        ▼
┌──────────────────────┐
│  Document Processing │
├──────────────────────┤
│ • OCR (images)       │
│ • PDF extraction     │
│ • Text cleaning      │
│ • Chunking           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Text Embedding      │
├──────────────────────┤
│ • BioBERT/PubMedBERT │
│ • Per-chunk embedding│
│ • Document aggregation
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Patient History     │
├──────────────────────┤
│ • Fetch past codes   │
│ • Encode history     │
│ • Temporal weighting │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Retrieval Stage     │
├──────────────────────┤
│ • Dense retrieval    │
│ • Top-K candidates   │
│ • BM25 + Vector sim  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Ranking Stage       │
├──────────────────────┤
│ • Transformer-based  │
│ • Context-aware      │
│ • Produces scores    │
└──────────┬───────────┘
           │
           ▼
OUTPUT: Ranked ICD Codes with Scores
```

### Phase 2: Advanced Features

1. **Hierarchical Prediction**
   - First predict ICD chapters (top-level categories)
   - Then predict specific codes within relevant chapters
   - Reduces search space significantly

2. **Multi-Head Prediction**
   - Primary diagnosis prediction
   - Secondary diagnosis prediction
   - Procedure code prediction
   - Different models for different prediction types

3. **Physician Feedback Loop**
   - Collect corrections from physicians
   - Retrain or fine-tune model
   - A/B test improvements

---

## 5. Data Flow

```
ENCOUNTER UPLOAD
    │
    ├─→ Validation & Security
    │   └─→ Check file types, scan for malware, encrypt
    │
    ├─→ Document Processing
    │   ├─→ OCR for images
    │   ├─→ PDF text extraction
    │   ├─→ Text normalization
    │   └─→ Store in Document DB
    │
    ├─→ Vector Embedding
    │   ├─→ Chunk documents (512 tokens)
    │   ├─→ BioBERT embedding
    │   ├─→ Store in Vector DB
    │   └─→ Index for retrieval
    │
    ├─→ Patient History Lookup
    │   ├─→ Fetch patient ID from encounter
    │   ├─→ Retrieve past diagnoses
    │   ├─→ Weight by recency
    │   └─→ Create history embedding
    │
    ├─→ Candidate Retrieval
    │   ├─→ Dense vector search
    │   ├─→ BM25 keyword search
    │   ├─→ Combine & deduplicate
    │   └─→ Get top-500 candidates
    │
    ├─→ Ranking & Prediction
    │   ├─→ Pass candidates + context to ranker
    │   ├─→ Score all candidates
    │   ├─→ Apply threshold filtering
    │   └─→ Sort by confidence
    │
    └─→ Prediction Output
        ├─→ Top-20 predicted codes
        ├─→ Confidence scores
        ├─→ Evidence (which documents)
        └─→ Display to physician for review
```

---

## 6. Model Selection

### Document Encoding
- **Pre-trained:** BioBERT or PubMedBERT (trained on medical literature)
- **Fine-tuning:** Optional, if labeled data available
- **Alternative:** Multi-modal CLIP-like model if training budget allows

### Retrieval Module
- **Dense Retrieval:** FAISS or Milvus for vector similarity
- **Sparse Retrieval:** BM25 for keyword matching
- **Hybrid:** Combine both approaches with weighted ensemble

### Ranking Module
- **Architecture:** Transformer-based cross-encoder
- **Inputs:** Document embedding + history embedding + code embedding
- **Output:** Relevance score per code
- **Training:** Contrastive learning on medical encounter data

### Baselines to Beat
1. **Rule-based:** Template matching + keyword extraction
2. **TF-IDF + Logistic Regression:** Simple baseline
3. **Fine-tuned BERT:** Single-task classification
4. **Hierarchical approach:** Two-stage prediction

---

## 7. Implementation Roadmap

### Phase 1: MVP (Week 1-2)
- [ ] Document processing pipeline
- [ ] Text extraction + basic cleaning
- [ ] Pre-trained model integration
- [ ] Simple BM25 retrieval
- [ ] Basic ranking module

### Phase 2: Core System (Week 3-4)
- [ ] Vector embedding & storage
- [ ] Patient history integration
- [ ] Confidence scoring
- [ ] UI for prediction display

### Phase 3: Optimization (Week 5-6)
- [ ] Caching & performance optimization
- [ ] A/B testing framework
- [ ] Feedback collection system
- [ ] Monitoring & alerting

### Phase 4: Advanced Features (Week 7+)
- [ ] Hierarchical prediction
- [ ] Multi-head prediction (diagnoses, procedures)
- [ ] Continuous learning from physician feedback
- [ ] Explainability dashboard

---

## 8. Key Metrics

### Model Performance
- **Precision@10:** Of top-10 predictions, how many are correct?
- **Recall@100:** Can we find the true codes in top-100?
- **MRR (Mean Reciprocal Rank):** Average rank of first correct prediction
- **NDCG:** Ranking quality considering relevance gradations

### System Performance
- **Latency:** Time from upload to predictions (target: < 30s)
- **Throughput:** Encounters processed per day
- **Accuracy:** Physician agreement with predictions
- **Adoption:** % of predictions accepted by physicians

### Business Metrics
- **Physician satisfaction:** Survey feedback
- **Time savings:** Time to assign codes (before vs. after)
- **Cost reduction:** Labor hours saved
- **Patient outcomes:** Indirect measures (faster billing, better follow-up)

---

## 9. Challenges & Considerations

### Data Challenges
- [ ] Limited labeled medical encounter data
- [ ] Class imbalance (some codes rarely used)
- [ ] Document quality varies widely
- [ ] Privacy concerns restrict data sharing

### Model Challenges
- [ ] Extreme multi-class (100k classes) requires careful architecture
- [ ] Medical domain requires specialized knowledge
- [ ] Needs to be interpretable for physician trust
- [ ] Must handle out-of-distribution documents gracefully

### Operational Challenges
- [ ] HIPAA compliance & secure infrastructure
- [ ] Need physician feedback for continuous improvement
- [ ] Model updates must not break existing workflows
- [ ] Fallback mechanisms for system failures

---

## Next Steps

1. **Define detailed data requirements** - What labeled data do we have?
2. **Choose pre-trained models** - BioBERT vs PubMedBERT vs GPT-based
3. **Design document processing pipeline** - Handle all document types
4. **Implement retrieval module** - Dense + sparse search
5. **Build ranking model** - Score candidate codes
6. **Evaluate & iterate** - A/B test with physicians
