# Hospital Patient Encounter ICD Prediction System

## Overview

This is a comprehensive ML system design for predicting ICD (International Classification of Diseases) codes from patient encounter documents. The system handles 15-20 documents of varying types and predicts codes from a catalog of ~100,000 codes.

## Problem Statement

**Input:** 15-20 patient documents (PDFs, images, text files)
**Context:** Patient's medical history
**Output:** Predicted ICD codes from 100,000 possible codes
**Constraint:** Prediction latency < 30 seconds

## Key Design Principles

### 1. Scalability
- **Two-stage architecture:** Retrieval (narrow candidates) + Ranking (score them)
- **Avoids softmax over 100k classes** - uses embedding-based ranking instead
- **Horizontal scaling:** Can handle growing ICD code catalog

### 2. Accuracy
- **Hybrid retrieval:** Combines dense (semantic) + sparse (keyword) search
- **Pre-trained medical models:** Uses BioBERT for medical text understanding
- **Context integration:** Incorporates patient history for better predictions

### 3. Interpretability
- **Confidence scores:** Multiple signals (model score + evidence + history)
- **Evidence extraction:** Shows which documents support each prediction
- **Explainable architecture:** Each component's role is clear

## System Architecture

```
Patient Documents
        ↓
Document Processing (PDF, OCR, text extraction)
        ↓
Embedding Generation (BioBERT)
        ↓
Candidate Retrieval (Dense + BM25) → Top-500 codes
        ↓
Patient History Lookup
        ↓
Ranking Model (Cross-encoder) → Scored predictions
        ↓
Confidence Estimation → Final predictions with scores
        ↓
Physician Review (with evidence & explanations)
```

## Files in This Folder

### design.md
High-level system design with:
- Problem statement and requirements
- Key design challenges and solutions
- Data flow diagrams
- Implementation roadmap
- Key metrics and monitoring

**Key sections:**
- 3. Key Design Challenges (Extreme multi-class, multi-modal, temporal)
- 4. Proposed Solution Architecture
- 8. Key Metrics

### architecture.md
Detailed technical architecture with:
- Component-by-component specifications
- Data storage design (PostgreSQL, Vector DB, S3, Redis)
- Request/response flow
- Performance breakdown by component
- Throughput and latency calculations

**Key sections:**
- Component details (API Gateway, Document Processing, Embedding, etc.)
- Data storage schemas
- Request flow (Upload → Process → Retrieve → Predict)
- Performance analysis (target 15-25s out of 30s budget)

### tradeoffs.md
Design decisions and tradeoffs:
- Retrieval-Ranking vs. Direct Prediction
- Dense vs. Sparse vs. Hybrid Retrieval
- Pre-trained model selection (General BERT vs. BioBERT vs. GPT)
- Single-stage vs. Hierarchical prediction
- History integration approaches
- Confidence scoring methods
- Technology choices (Vector DB, OCR, etc.)

**Useful for understanding "why" each decision was made.**

### template.py
Python implementation template with:
- Abstract base classes for extensibility
- Document processing pipeline
- Embedding service
- Retrieval module
- Ranking model (PyTorch)
- Full orchestrator
- Example usage

**Use this to start implementing the system.**

## Building on This Foundation

### Next Steps

1. **Phase 1: MVP**
   - [ ] Implement DocumentProcessor for all file types
   - [ ] Integrate BioBERT for embeddings
   - [ ] Set up BM25 + Vector DB retrieval
   - [ ] Build simple ranking model
   - [ ] End-to-end integration

2. **Phase 2: Core System**
   - [ ] Patient history integration
   - [ ] Confidence scoring
   - [ ] Evidence extraction
   - [ ] UI for physician review

3. **Phase 3: Optimization**
   - [ ] Caching strategy
   - [ ] Performance tuning
   - [ ] A/B testing framework
   - [ ] Monitoring & alerting

4. **Phase 4: Advanced**
   - [ ] Hierarchical prediction (ICD chapters)
   - [ ] Multi-head model (diagnoses, procedures, complications)
   - [ ] Continuous learning from physician feedback
   - [ ] Knowledge graph integration

### Key Technical Decisions

| Aspect | Choice | Why |
|--------|--------|-----|
| **Retrieval** | Two-stage (retrieval + ranking) | Scalable, modular, fast |
| **Retrieval Method** | Hybrid (dense + BM25) | Semantic + keyword coverage |
| **Encoder** | BioBERT | Medical domain knowledge |
| **Ranking** | Single-stage | Avoid hierarchical cascade errors |
| **History** | Explicit embedding | Interpretable, practical |
| **Confidence** | Ensemble (model + evidence + history) | Multiple quality signals |
| **Infrastructure** | Hybrid (central training, local inference) | Privacy + scalability |

See `tradeoffs.md` for detailed reasoning on each choice.

## Estimated Performance

### Latency Breakdown
| Component | Time | Notes |
|-----------|------|-------|
| File validation | 1s | Parallel |
| Document processing | 2-3s | OCR depends on image quality |
| Text extraction | 0.5s | Cached for common formats |
| Embedding generation | 5-8s | Batch 32, GPU |
| Retrieval (dense + BM25) | 1.5s | HNSW + Elasticsearch |
| Ranking (500 codes) | 3-5s | Transformer, GPU |
| Confidence & scoring | 0.5s | Simple computations |
| DB writes | 1s | Batch insert |
| **Total** | **~15-25s** | **Healthy margin to 30s target** |

### Throughput
- **Per GPU:** 2000 documents/hour
- **Effective:** 100-200 encounters/hour (15-20 docs each)
- **Scale:** 1-2 hospitals (grows with more GPUs)

## Key Research Areas

1. **Handling 100k classes efficiently**
   - Embedding-based ranking instead of softmax
   - Hierarchical predictions (ICD structure)
   - Retrieval-then-rank pattern

2. **Multi-modal document processing**
   - Specialized encoders for different document types
   - Fusion strategies
   - Document importance learning

3. **Medical domain knowledge**
   - Pre-trained models (BioBERT, PubMedBERT, ClinicalBERT)
   - Knowledge graphs and ontologies
   - Expert rules and constraints

4. **Temporal reasoning**
   - Patient history patterns
   - Recency weighting
   - Chronic vs. acute conditions

## Implementation Checklist

### Data Preparation
- [ ] Collect labeled patient encounters with ICD codes
- [ ] Build ICD code embeddings
- [ ] Set up patient history database
- [ ] Create train/val/test splits

### Model Training
- [ ] Fine-tune BioBERT on medical documents
- [ ] Train cross-encoder ranking model
- [ ] Calibrate confidence scoring
- [ ] Evaluate against baselines

### System Integration
- [ ] Document processing pipeline
- [ ] Vector DB setup and indexing
- [ ] API gateway and load balancing
- [ ] Caching and optimization

### Deployment & Monitoring
- [ ] Physician feedback collection
- [ ] A/B testing framework
- [ ] Model performance monitoring
- [ ] Continuous learning pipeline

## Questions to Explore

1. **Data:** How much labeled training data is available?
2. **Baseline:** What's the current process? (Manual coding time)
3. **Constraints:** HIPAA compliance requirements?
4. **Hierarchy:** Should we use ICD code hierarchy (chapters → sections → codes)?
5. **Feedback:** How will physician feedback be collected and used?
6. **Multi-task:** Separate models for diagnoses vs. procedures vs. complications?
7. **Personalization:** Should predictions be personalized per hospital/physician?
8. **Integration:** How does this integrate with hospital EHR system?

## References & Further Reading

- **ICD Coding:** https://www.cdc.gov/nchs/icd/
- **BioBERT:** Lee et al., "BioBERT: a pre-trained biomedical language representation model for biomedical text mining"
- **Information Retrieval:** Humeau et al., "Large-scale Learnable Graph Convolutional Networks"
- **Cross-Encoders:** Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **Medical NLP:** Xie et al., "A Benchmark Study of Evaluation Metrics in Medical QA Systems"

## Contributing

When extending this system, follow these principles:

1. **Modularity:** Keep components loosely coupled
2. **Testability:** Write tests for new components
3. **Documentation:** Document design decisions in tradeoffs.md
4. **Evaluation:** Measure impact on metrics (precision, recall, latency)
5. **Safety:** Medical predictions require careful validation

