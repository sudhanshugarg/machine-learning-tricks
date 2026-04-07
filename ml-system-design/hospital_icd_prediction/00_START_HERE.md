# 🚀 START HERE: Hospital ICD Prediction System

Welcome! This folder contains a **complete ML system design** for hospital patient encounter ICD code prediction.

## ⏱️ How Much Time Do You Have?

### ⚡ 5 Minutes
Read: **SUMMARY.md**
- Get the core idea in 5 minutes
- Understand the two-stage retrieval-ranking approach

### 🚴 30 Minutes  
Read: **SUMMARY.md** → **GETTING_STARTED.md**
- Understand the problem and solution
- Learn the key components
- See performance breakdown

### 🏃 1 Hour (Interview Ready)
Read: **GETTING_STARTED.md** → **design.md** → **tradeoffs.md**
- Full context on requirements and challenges
- Why we chose each approach
- Can discuss in an interview

### 🏋️ 2-3 Hours (Implementation Ready)
Read all files in order:
1. **GETTING_STARTED.md** - Overview
2. **design.md** - Requirements and challenges
3. **architecture.md** - Technical details
4. **tradeoffs.md** - Design decisions
5. **template.py** - Code skeleton
6. **README.md** - Reference

Ready to start coding!

---

## 📚 File Quick Reference

| File | Time | Purpose | Best For |
|------|------|---------|----------|
| **SUMMARY.md** | 5 min | Executive summary | Quick overview |
| **GETTING_STARTED.md** | 15 min | Intro + FAQ + diagrams | Learning basics |
| **design.md** | 15 min | Requirements + challenges | Understanding problem |
| **architecture.md** | 20 min | Technical details | Implementation |
| **tradeoffs.md** | 15 min | Design decisions | Understanding "why" |
| **template.py** | 20 min | Code skeleton | Starting implementation |
| **README.md** | 10 min | Project reference | Quick lookup |
| **INDEX.md** | 10 min | Navigation guide | Finding specific topics |

---

## 🎯 The Problem (30 seconds)

A hospital receives 15-20 patient documents (PDFs, images, text) and needs to predict ICD medical codes from a catalog of **100,000 codes**.

**Challenge:** Can't use standard neural networks (softmax over 100k classes = too slow/memory intensive)

**Solution:** Two-stage approach
- **Stage 1:** Retrieve top-500 candidates using BioBERT embeddings + BM25
- **Stage 2:** Rank 500 candidates using transformer cross-encoder

**Result:** 15-25 seconds end-to-end, fully interpretable, shows supporting evidence

---

## 💡 The Big Insight

Standard ML approach: `documents → 100k softmax → predictions` ❌ Too slow/memory intensive

Our approach: `documents → retrieve 500 → rank 500 → predictions` ✅ Fast, modular, scalable

---

## 🗺️ Choose Your Path

### Path A: "I just want to understand"
1. Read **SUMMARY.md** (5 min)
2. Skim **GETTING_STARTED.md** (10 min)
3. **Done!** You understand the system.

### Path B: "I'm interviewing for this role"
1. Read **GETTING_STARTED.md** (15 min)
2. Read **design.md** (20 min)
3. Read **tradeoffs.md** (20 min)
4. Practice explaining to a friend
5. **Done!** You're ready to interview.

### Path C: "I want to implement this"
1. Read **GETTING_STARTED.md** (15 min)
2. Read **design.md** (15 min)
3. Study **architecture.md** (30 min)
4. Review **template.py** (30 min)
5. Check **tradeoffs.md** for each decision (20 min)
6. **Start coding!**

### Path D: "I want to understand everything"
Read all 8 files in this order:
1. GETTING_STARTED.md
2. SUMMARY.md
3. design.md
4. architecture.md
5. tradeoffs.md
6. template.py
7. README.md
8. INDEX.md

**Total time:** ~90 minutes for complete mastery

---

## ✅ Key Takeaways

After reading, you should understand:

1. **The Problem**
   - Hospital receives 15-20 documents per patient
   - Need to predict ICD codes from 100k possibilities
   - Must be fast (< 30s) and interpretable

2. **The Solution**
   - Two-stage retrieval-ranking architecture
   - Stage 1: Retrieve top-500 using hybrid search (dense + sparse)
   - Stage 2: Rank using cross-encoder transformer

3. **Why This Works**
   - Retrieval is fast and narrows search space
   - Ranking is accurate on small set of 500
   - Together: Fast + Accurate + Interpretable

4. **Key Technologies**
   - BioBERT for medical text embeddings
   - Hybrid retrieval (FAISS for dense, BM25 for sparse)
   - Cross-encoder transformer for ranking
   - PostgreSQL + Redis for caching

5. **Performance**
   - 15-25 seconds end-to-end (out of 30s budget)
   - 100-200 encounters/day throughput
   - Fully interpretable (shows supporting evidence)

---

## 🚀 Next Steps

1. **Choose a reading path** above (5 min - 3 hours)
2. **Read the appropriate files** based on your time
3. **Ask questions** - review **GETTING_STARTED.md** FAQ
4. **Understand design decisions** - read **tradeoffs.md**
5. **Start implementing** - use **template.py** as skeleton

---

## ❓ Quick FAQ

**Q: What if I only have 5 minutes?**
A: Read SUMMARY.md for the essentials.

**Q: Why two stages instead of one big model?**
A: Softmax over 100k classes doesn't fit in memory and is slow. Two-stage is the industry standard for extreme multi-class problems.

**Q: Why BioBERT instead of ChatGPT?**
A: BioBERT is 100x faster and fits our 30-second budget. ChatGPT can't process documents that quickly.

**Q: Can this actually work?**
A: Yes! This is based on proven ML system design patterns used by companies like Google, Amazon, and Meta for similar problems (e-commerce search, recommendation systems, etc.).

**Q: Where do I start coding?**
A: Use template.py as your skeleton. It has all the classes and TODO comments showing what to implement.

---

## 📊 System Architecture (Simplified)

```
Patient Encounter Documents (15-20)
        ↓
Extract Text (OCR, PDF parsing)
        ↓
Generate Embeddings (BioBERT: 768-dim vectors)
        ↓
━━━━━━━ STAGE 1: RETRIEVE ━━━━━━━━
│ Dense Search (vector similarity) → Top-200
│ + Sparse Search (BM25 keywords) → Top-100
│ = Hybrid Ensemble → Top-500 candidates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ↓
Fetch Patient History (past diagnoses)
        ↓
━━━━━━━ STAGE 2: RANK ━━━━━━━━━
│ Cross-Encoder Transformer
│ Scores all 500 candidates
│ Estimate confidence
━━━━━━━━━━━━━━━━━━━
        ↓
Top-20 Predictions
+ Confidence Scores
+ Supporting Evidence
```

---

## 🎓 Learning Resources in This Folder

- **Beginner friendly:** SUMMARY.md, GETTING_STARTED.md
- **Intermediate:** design.md, architecture.md
- **Advanced:** template.py, tradeoffs.md
- **Navigation:** INDEX.md
- **Reference:** README.md

---

## ✨ Ready to Go?

### Quickest Path (5 minutes)
→ Open **SUMMARY.md** right now

### Best Path (30 minutes)
→ Open **SUMMARY.md** then **GETTING_STARTED.md**

### Implementation Path (2-3 hours)
→ Read **GETTING_STARTED.md** then **design.md** then **architecture.md** then **template.py**

---

**Happy learning! 🚀**

Any questions? Check **GETTING_STARTED.md** for FAQ or **INDEX.md** for cross-references.
