# Hospital ICD Prediction System - Complete Index

## 📋 System Overview

**Problem:** Design an ML system that predicts ICD codes from 15-20 patient documents (multi-modal) with ~100,000 possible codes in a hospital setting.

**Solution:** Two-stage retrieval-ranking architecture using BioBERT embeddings, hybrid retrieval (dense + sparse), and cross-encoder ranking.

**Performance:** 15-25 seconds end-to-end, interpretable predictions with confidence scores and evidence.

---

## 📚 Complete File Guide

### Quick Start (Choose Your Path)

#### 🚀 **5-Minute Overview**
→ Read: **SUMMARY.md**
- Problem statement
- High-level solution
- Key design decisions
- Performance breakdown

#### 🎯 **30-Minute Deep Dive**
→ Read: **GETTING_STARTED.md** → **design.md**
- Complete problem context
- Requirements and challenges
- System architecture overview
- Implementation roadmap

#### 💻 **Full Study (2-3 hours)**
→ Read all files in order:
1. **GETTING_STARTED.md** - Navigation and quick overview
2. **design.md** - Requirements, challenges, solutions
3. **architecture.md** - Technical details, components, databases
4. **tradeoffs.md** - Design decisions and alternatives
5. **template.py** - Implementation skeleton

---

## 📄 Detailed File Descriptions

### 1. **GETTING_STARTED.md** (12KB)
**Purpose:** Navigation guide and quick-start
**Contains:**
- Role-based entry points (interviewer, learner, implementer)
- 30-second problem summary
- Architecture diagram
- Component explanations
- Technology stack
- Performance budget breakdown
- Implementation roadmap
- FAQ

**Start here if:** You're new to this design or want a quick overview

---

### 2. **SUMMARY.md** (6.4KB)
**Purpose:** Executive summary with key insights
**Contains:**
- Problem statement with challenges
- Solution architecture (two-stage)
- Component breakdown
- Data flow
- Key design decisions with justification
- Performance metrics
- What makes this design good
- What to explore next

**Start here if:** You want the absolute fastest introduction

---

### 3. **design.md** (14KB)
**Purpose:** High-level system design document
**Contains:**
- Problem statement (detailed)
- Functional & non-functional requirements
- System architecture overview (ASCII diagram)
- Key design challenges (3 main ones):
  1. Extreme multi-class problem (100k codes)
  2. Multi-modal input documents
  3. Temporal & contextual information
- Proposed solution architecture
- Data flow
- Model selection rationale
- Implementation roadmap (4 phases)
- Key metrics (model, system, business)
- Challenges & considerations

**Section breakdown:**
- §1-2: Requirements and context
- §3: Key challenges (what makes this hard)
- §4: Proposed solution (two-stage approach)
- §5: Data flow (step-by-step)
- §6: Model selection
- §7: Implementation roadmap
- §8: Metrics

**Best for:** Understanding the "what" and "why"

---

### 4. **architecture.md** (16KB)
**Purpose:** Detailed technical architecture
**Contains:**
- System components (API gateway, processors, embeddings, retrieval, ranking)
- Document processing pipeline (flowchart)
- Embedding & vector storage
- Patient history module
- Retrieval module (dense + sparse + ensemble)
- Ranking module (cross-encoder)
- Confidence & filtering
- Data storage design:
  - PostgreSQL schemas (encounters, documents, diagnoses, predictions)
  - Vector database (Milvus) structure
  - S3 file organization
  - Redis cache design
- Request/response flow (4 stages)
- Performance analysis:
  - Latency breakdown by component
  - Throughput calculations
  - Cost optimization strategies

**Section breakdown:**
- §1-7: Component details with diagrams
- §8: Data storage (SQL, vector DB, files, cache)
- §9: Request flow (upload → process → predict → feedback)
- §10: Performance analysis

**Best for:** Understanding the "how" (implementation details)

---

### 5. **tradeoffs.md** (11KB)
**Purpose:** Design decisions and alternative approaches
**Contains:**
- 10 major design decisions:
  1. Retrieval-ranking vs. direct softmax (why two stages)
  2. Dense vs. sparse vs. hybrid retrieval
  3. General BERT vs. BioBERT vs. GPT models
  4. Single-stage vs. hierarchical prediction
  5. Patient history approaches (explicit, implicit, none)
  6. Confidence scoring methods
  7. Physician feedback loop strategies
  8. Latency vs. accuracy tradeoff
  9. Centralized vs. distributed models
  10. Technology choices (databases, OCR, etc.)
- For each: pros, cons, and chosen approach with reasoning

**Structure:** Each decision has:
- Approach comparison (2-3 alternatives)
- Pros/cons for each
- Chosen solution with justification

**Best for:** Understanding "why we chose X over Y"

---

### 6. **template.py** (14KB)
**Purpose:** Python implementation skeleton
**Contains:**
- Starter classes and interfaces:
  - `Document` dataclass
  - `Prediction` dataclass
  - `PatientHistory` dataclass
  - `DocumentProcessor` (abstract base)
  - `PDFProcessor`, `ImageOCRProcessor`, `TextProcessor`
  - `EmbeddingService` (BioBERT wrapper)
  - `RetrieverModule` (dense + sparse retrieval)
  - `RankerModel` (PyTorch cross-encoder)
  - `ConfidenceEstimator`
  - `ICDPredictor` (main orchestrator)
- Each class has:
  - Clear docstrings
  - TODO comments with implementation steps
  - Method signatures
  - Expected inputs/outputs
- Main example usage
- Helper method stubs

**Structure:**
- Components 1-9: Individual classes (can implement independently)
- ICDPredictor: Orchestrator that ties everything together
- main(): Example usage

**Best for:** Starting implementation, understanding class responsibilities

---

### 7. **README.md** (8.1KB)
**Purpose:** Project overview and quick reference
**Contains:**
- Overview paragraph
- Problem statement
- Key design principles (scalability, accuracy, interpretability)
- System architecture diagram
- File descriptions
- Building on foundation (next steps)
- Key technical decisions table
- Estimated performance (latency, throughput)
- Key research areas
- Implementation checklist (4 sections)
- Questions to explore
- References

**Best for:** Project overview and quick reference

---

### 8. **INDEX.md** (this file)
**Purpose:** Navigation and file organization guide
**Contains:**
- Quick start paths
- File descriptions
- Section breakdowns
- Best use cases for each file
- Reading recommendations
- Cross-references

---

## 🗺️ Reading Paths

### Path 1: Quick Learner (30 minutes)
1. **SUMMARY.md** (5 min) - Get the big picture
2. **architecture.md** - Components section (10 min) - Understand pieces
3. **GETTING_STARTED.md** - Design decisions section (15 min) - Understand tradeoffs

**Outcome:** Can explain the system and why it was designed this way

---

### Path 2: Interview Preparation (1 hour)
1. **GETTING_STARTED.md** (15 min) - Overview
2. **design.md** (20 min) - Full design context
3. **tradeoffs.md** (20 min) - Design decision reasoning
4. **SUMMARY.md** (5 min) - Talking points

**Outcome:** Can give a polished interview answer with good reasoning

---

### Path 3: Implementation (2-3 hours)
1. **GETTING_STARTED.md** (15 min) - Get context
2. **design.md** (15 min) - Understand requirements
3. **architecture.md** (30 min) - Technical specs
4. **template.py** (30 min) - Code structure
5. **tradeoffs.md** (20 min) - Design rationale
6. **Start coding!**

**Outcome:** Ready to implement with clear architecture

---

### Path 4: Deep Analysis (3-4 hours)
Read all files in order:
1. GETTING_STARTED.md
2. SUMMARY.md
3. design.md
4. architecture.md
5. tradeoffs.md
6. template.py
7. README.md

**Outcome:** Complete understanding of system, can extend or improve it

---

## 🎯 Quick Reference

### Key Numbers
- **100,000** possible ICD codes (extreme multi-class)
- **15-20** documents per patient
- **30 seconds** latency budget
- **15-25 seconds** actual latency (85% of budget)
- **500** candidate codes retrieved
- **768** dimensions for embeddings
- **100-200** encounters/day throughput

### Key Technologies
- **BioBERT** - Medical text embeddings
- **Milvus/FAISS** - Vector similarity search
- **BM25/Elasticsearch** - Keyword search
- **Transformer** - Cross-encoder ranking
- **PostgreSQL + S3** - Data storage
- **Redis** - Caching

### Key Performance Metrics
- **Precision@10** - Top-10 accuracy
- **Recall@100** - Can find true codes
- **MRR** - Rank of first correct code
- **NDCG** - Ranking quality
- **Latency** - Response time
- **Throughput** - Encounters/day

---

## ❓ FAQ by Question

**Q: What's the core insight?**
A: Can't use standard softmax on 100k classes. Use retrieval to find 500 candidates, then rank them.

**Q: Why not use ChatGPT?**
A: Too slow - need < 30s latency. BioBERT is 100x faster.

**Q: What if we miss the correct code in retrieval?**
A: Mitigated by hybrid retrieval (dense + sparse) and optional hierarchical fallback.

**Q: How do we handle new codes?**
A: Automatically - just add to vector DB without retraining.

**Q: Can we personalize per hospital?**
A: Yes - Phase 4 includes hospital-specific customization.

**Q: How do physicians provide feedback?**
A: Collect corrections in UI, retrain weekly, immediate local adjustments.

---

## 🔗 Cross-References

### By Topic

**Extreme Multi-Class Problem:**
- design.md §3.1
- tradeoffs.md §1
- GETTING_STARTED.md "Why two stages instead of one big model?"

**Multi-Modal Documents:**
- design.md §3.2
- architecture.md §2
- template.py DocumentProcessor classes

**Patient History Integration:**
- design.md §3.3
- tradeoffs.md §5
- architecture.md §4

**Performance & Optimization:**
- GETTING_STARTED.md "Performance Budget"
- architecture.md §10
- design.md §8

**Implementation Roadmap:**
- design.md §7
- GETTING_STARTED.md "Implementation Roadmap"
- README.md "Implementation Checklist"

**Technology Choices:**
- tradeoffs.md §10
- architecture.md "Performance Considerations"
- GETTING_STARTED.md "Technology Stack"

---

## 📊 Complexity & Time Estimates

| Document | Complexity | Read Time | Best For |
|----------|-----------|-----------|----------|
| SUMMARY.md | Low | 5 min | Quick overview |
| GETTING_STARTED.md | Low-Medium | 15 min | Quick intro + FAQ |
| design.md | Medium | 15 min | Requirements & high-level |
| architecture.md | Medium-High | 20 min | Technical details |
| tradeoffs.md | Medium | 15 min | Design reasoning |
| template.py | High | 20 min | Implementation |
| README.md | Low | 10 min | Reference |
| **Total** | **N/A** | **~90 min** | **Complete understanding** |

---

## 🚀 Next Actions

**I want to...**

- **...understand the design quickly**
  → SUMMARY.md + GETTING_STARTED.md

- **...prepare for an interview**
  → GETTING_STARTED.md + design.md + tradeoffs.md

- **...implement this system**
  → template.py + architecture.md + design.md

- **...understand a specific decision**
  → tradeoffs.md (find topic) + design.md (find section)

- **...set up all the infrastructure**
  → architecture.md §8 (Data Storage Architecture)

- **...get performance specifications**
  → architecture.md §10 or GETTING_STARTED.md "Performance Budget"

---

## 📝 Summary

This folder contains a **complete ML system design** for hospital patient encounter ICD code prediction. All 8 files work together:

- **SUMMARY.md** - The elevator pitch
- **GETTING_STARTED.md** - Navigation and quick answers
- **design.md** - Problem and solution overview
- **architecture.md** - Technical implementation details
- **tradeoffs.md** - Why we chose each approach
- **template.py** - Code skeleton to start with
- **README.md** - Project overview
- **INDEX.md** - This file

**Start with SUMMARY.md or GETTING_STARTED.md, then go deeper based on your needs.**

