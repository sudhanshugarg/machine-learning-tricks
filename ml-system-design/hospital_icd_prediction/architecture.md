# Hospital ICD Prediction System - Detailed Architecture

## System Components

### 1. API Gateway & Load Balancer
```
┌────────────────────────┐
│   Hospital Portal      │
│   (React/Vue Frontend) │
└───────────┬────────────┘
            │
┌───────────▼────────────┐
│   Load Balancer (ALB)  │
│   Health Checks        │
└───────────┬────────────┘
            │
    ┌───────┴────────┐
    │                │
┌───▼──────┐    ┌───▼──────┐
│ API Pod  │    │ API Pod  │  (Horizontally scalable)
│ Instance1│    │ Instance2│
└────┬─────┘    └────┬─────┘
     │               │
     └───────┬───────┘
             │
     (REST API: /predict)
```

### 2. Document Processing Pipeline

```
INPUT: Document File
         │
         ▼
┌─────────────────────┐
│ File Validation     │
├─────────────────────┤
│ • MIME type check   │
│ • Size limits       │
│ • Virus scan        │
└─────────┬───────────┘
          │
┌─────────▼────────────────────────────┐
│ Document Type Router                 │
├──────────────────────────────────────┤
│ ├─ .pdf   → PDF Parser               │
│ ├─ .jpg   → OCR Pipeline             │
│ ├─ .txt   → Raw Text Extraction      │
│ └─ .docx  → Document Parser          │
└─────────┬────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│ Text Extraction & Cleaning          │
├─────────────────────────────────────┤
│ • Remove headers/footers            │
│ • Normalize whitespace              │
│ • Remove special characters         │
│ • Spelling correction (optional)    │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│ Document Chunking                   │
├─────────────────────────────────────┤
│ • Chunk size: 512 tokens            │
│ • Overlap: 128 tokens               │
│ • Preserve document boundaries      │
└─────────┬───────────────────────────┘
          │
          ▼
    Store in Document DB
    (PostgreSQL + S3)
```

### 3. Embedding & Vector Storage

```
DOCUMENTS (chunks)
         │
         ▼
┌──────────────────────────┐
│ BioBERT/PubMedBERT       │
│ Encoder                  │
├──────────────────────────┤
│ Input: Text chunk (512)  │
│ Output: 768-dim vector   │
│ Batch size: 32           │
│ GPU-accelerated          │
└──────────┬───────────────┘
           │
    ┌──────▼──────┐
    │ Normalize   │
    │ L2 norm     │
    └──────┬──────┘
           │
           ▼
┌──────────────────────────────────┐
│ Vector DB (Milvus/FAISS)         │
├──────────────────────────────────┤
│ Storage Format:                  │
│ ├─ encounter_id (indexed)        │
│ ├─ document_type (filtered)      │
│ ├─ chunk_id (metadata)           │
│ ├─ vector (768-dim)              │
│ └─ created_at (temporal filter)  │
│                                  │
│ Indexes:                         │
│ ├─ HNSW (hierarchical)           │
│ ├─ IVF (inverted file)           │
│ └─ PQ (product quantization)     │
└──────────────────────────────────┘
```

### 4. Patient History Module

```
┌──────────────────────────────────┐
│ Patient ID Lookup                │
│ (from encounter metadata)        │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Patient History DB               │
│ (PostgreSQL)                     │
├──────────────────────────────────┤
│ Table: patient_diagnoses         │
│ ├─ patient_id (indexed)          │
│ ├─ icd_code                      │
│ ├─ encounter_date                │
│ ├─ frequency                     │
│ └─ relevance_score               │
│                                  │
│ Aggregation:                     │
│ ├─ Top-20 codes by frequency     │
│ ├─ Time decay (recent > old)     │
│ └─ Normalize to embedding        │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ History Embedding                │
├──────────────────────────────────┤
│ • Sum of code embeddings         │
│ • Weighted by frequency          │
│ • Weighted by recency            │
│ • Output: 768-dim vector         │
└──────────────────────────────────┘
```

### 5. Retrieval Module

```
INPUTS:
  • Document embeddings
  • Patient history embedding
  • Query encoding
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Dense    │ │ BM25     │ │ History  │
│ Retrieval│ │ Retrieval│ │ Matching │
│ (ANN)    │ │ (Full-T.)│ │ (Cached) │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     │ Top-200    │ Top-100    │ Top-50
     │ codes      │ codes      │ codes
     │            │            │
     └────────┬───┴────────┬───┘
              │            │
        ┌─────▼────────────▼──────┐
        │ Ensemble & Deduplication│
        │ • Union of candidates   │
        │ • Rerank by combined    │
        │   score                 │
        │ • Keep top-500          │
        └─────┬──────────────────┘
              │
         TOP-500 CANDIDATES
         (ICD codes with scores)
```

### 6. Ranking Module (Cross-Encoder)

```
INPUT: Candidate codes + document context
         │
         ▼
┌────────────────────────────────┐
│ Cross-Encoder Transformer      │
├────────────────────────────────┤
│ Architecture:                  │
│ • Input: [CLS] doc_tokens      │
│   code_tokens [SEP] history    │
│ • Hidden size: 768             │
│ • Num layers: 12               │
│ • Num heads: 12                │
│ • FFN: 3072                    │
│                                │
│ Training:                      │
│ • Contrastive loss             │
│ • Hard negative mining         │
│ • Focal loss for imbalance     │
└─────┬──────────────────────────┘
      │
      ▼
┌────────────────────────────────┐
│ Score Normalization            │
├────────────────────────────────┤
│ • Sigmoid activation           │
│ • Score in range [0, 1]        │
│ • Per-encounter normalization  │
└─────┬──────────────────────────┘
      │
      ▼
RANKED CODES WITH SCORES
(sorted by relevance)
```

### 7. Confidence & Filtering

```
RANKED CODES WITH SCORES
         │
         ▼
┌─────────────────────────────────┐
│ Threshold Filtering             │
├─────────────────────────────────┤
│ • Global threshold: 0.3         │
│ • Code-specific threshold       │
│ • Physician-configured          │
└─────┬───────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Confidence Estimation           │
├─────────────────────────────────┤
│ • Model score (0-1)             │
│ • Calibration via Platt scaling │
│ • Evidence strength:            │
│   - How many docs support it    │
│   - Location in doc (header)    │
│   - Frequency of mentions       │
└─────┬───────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Explanation Generation          │
├─────────────────────────────────┤
│ For each predicted code:        │
│ • Which documents mentioned it  │
│ • Exact snippets (top-3)        │
│ • Confidence score              │
│ • Similar past cases            │
└─────┬───────────────────────────┘
      │
      ▼
OUTPUT: Top-20 codes with scores
        and explanations
```

## Data Storage Architecture

### 1. PostgreSQL (Relational Data)

```sql
-- Encounters
CREATE TABLE encounters (
    encounter_id UUID PRIMARY KEY,
    patient_id UUID,
    hospital_id UUID,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    predicted_icd_codes JSONB,  -- Quick access to predictions
    INDEX(patient_id),
    INDEX(created_at)
);

-- Documents
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY,
    encounter_id UUID REFERENCES encounters,
    doc_type VARCHAR(20),  -- 'pdf', 'image', 'text', etc.
    file_path VARCHAR(500),  -- S3 location
    text_content TEXT,  -- Extracted text
    created_at TIMESTAMP,
    INDEX(encounter_id)
);

-- Patient History
CREATE TABLE patient_diagnoses (
    id BIGSERIAL PRIMARY KEY,
    patient_id UUID,
    icd_code VARCHAR(10),
    encounter_id UUID,
    encounter_date DATE,
    frequency INT DEFAULT 1,
    created_at TIMESTAMP,
    UNIQUE(patient_id, icd_code, encounter_id),
    INDEX(patient_id, encounter_date DESC)
);

-- Predictions
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    encounter_id UUID REFERENCES encounters,
    icd_code VARCHAR(10),
    score FLOAT,
    rank INT,
    created_at TIMESTAMP,
    physician_approved BOOLEAN DEFAULT NULL,
    INDEX(encounter_id),
    INDEX(physician_approved)
);
```

### 2. Vector Database (Milvus)

```
Collection: document_chunks
├─ Vector field:
│  ├─ Name: embeddings
│  ├─ Dimension: 768
│  ├─ Type: float32
│  └─ Index: HNSW
│
├─ Scalar fields:
│  ├─ encounter_id (indexed)
│  ├─ document_type (tagged)
│  ├─ chunk_id
│  ├─ text_content
│  └─ created_at
│
└─ Indexes:
   ├─ HNSW on embeddings (ANN search)
   ├─ Sorted on created_at (temporal)
   └─ Tag on document_type (filtering)
```

### 3. Document Storage (S3)

```
s3://hospital-icd-bucket/
├─ encounters/
│  ├─ {encounter_id}/
│  │  ├─ documents/
│  │  │  ├─ {doc_id}_original.pdf
│  │  │  ├─ {doc_id}_original.jpg
│  │  │  └─ ...
│  │  ├─ extracted_text/
│  │  │  └─ {doc_id}_text.txt
│  │  └─ metadata.json
│  └─ ...
│
└─ models/
   ├─ encoder/
   │  ├─ config.json
   │  ├─ pytorch_model.bin
   │  └─ vocab.txt
   └─ ranker/
       ├─ config.json
       └─ pytorch_model.bin
```

### 4. Cache Layer (Redis)

```
Redis Cluster:
├─ Patient history embeddings
│  ├─ Key: patient:{patient_id}:history_embedding
│  ├─ Value: 768-dim vector (serialized)
│  └─ TTL: 24 hours
│
├─ Code embeddings
│  ├─ Key: icd_code:{code}:embedding
│  ├─ Value: 768-dim vector
│  └─ TTL: Never expires
│
├─ Recent predictions
│  ├─ Key: encounter:{encounter_id}:predictions
│  ├─ Value: JSON list of (code, score)
│  └─ TTL: 7 days
│
└─ Hot codes (frequently predicted)
   ├─ Key: code:popular
   ├─ Value: Sorted set of codes by frequency
   └─ TTL: 1 hour
```

## Request Flow

```
1. UPLOAD
   POST /api/encounters/predict
   Body: [file1, file2, ...]
   │
   ├─ Validate files
   ├─ Store in S3
   ├─ Return encounter_id
   └─ Queue for async processing

2. PROCESS (Background Job)
   ├─ Process documents
   ├─ Extract text & generate chunks
   ├─ Generate embeddings
   ├─ Retrieve candidates
   ├─ Rank candidates
   ├─ Store predictions in DB
   └─ Notify API that results ready

3. RETRIEVE
   GET /api/encounters/{encounter_id}/predictions
   │
   ├─ Fetch from DB
   ├─ Generate explanations
   ├─ Format response
   └─ Return to frontend

4. FEEDBACK
   POST /api/predictions/{prediction_id}/feedback
   ├─ Save physician correction
   ├─ Update training data
   ├─ Trigger retraining (batched)
   └─ Log for audit
```

## Performance Considerations

### Latency Breakdown (Target: 30s end-to-end)

| Component | Time | Notes |
|-----------|------|-------|
| File validation | 1s | Parallel for multiple files |
| OCR (per image) | 2-3s | Depends on image quality |
| Text extraction | 0.5s | Cached for common formats |
| Document chunking | 0.2s | In-memory |
| Embedding generation | 5-8s | Batch 32, GPU |
| Vector retrieval | 1s | HNSW index, top-500 |
| BM25 retrieval | 0.5s | Elasticsearch |
| Ranking (top-500) | 3-5s | Transformer, GPU batch |
| Confidence scoring | 0.5s | Simple computations |
| DB writes | 1s | Batch insert |
| **Total** | **~15-25s** | **Healthy margin to 30s** |

### Throughput

- **GPU memory:** 16GB
- **Batch size:** 32 documents
- **Time per batch:** ~1 second
- **Throughput:** ~2000 documents/hour
- **Effective:** 100-200 encounters/hour (15-20 docs per encounter)

### Cost Optimization

1. **Caching Strategy**
   - Cache patient history embeddings (24h)
   - Cache code embeddings (permanent)
   - Cache popular candidates (1h)

2. **Indexing**
   - HNSW for fast ANN search
   - Filtered to recent documents only
   - Pruned low-quality documents

3. **Batch Processing**
   - Accumulate 10-20 encounters
   - Run embedding batches
   - Reduce model loading overhead
