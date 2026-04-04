# ChatGPT Clone - Design Tradeoffs

## 1. Model Size vs Latency vs Cost

| Model Size | TTFT | Cost/req | Quality | Use Case |
|-----------|------|----------|---------|----------|
| 1B        | 50ms | $0.0001  | Poor    | Mobile only |
| 7B        | 200ms| $0.0003  | Good    | Cost-sensitive |
| 13B       | 500ms| $0.0010  | Excellent | **Balanced** |
| 70B       | 2s   | $0.0050  | Exceptional | Premium tier |

**Tradeoff:**
- **Smaller models**: Faster, cheaper, but noticeably lower quality
- **Larger models**: High quality, but slow and expensive

**Decision:** Use 13B model (Llama-13B or equivalent)
- Best balance of quality and performance
- Scales to millions of users
- Cost-effective ($0.001 per response = $0.003 per 3-response conversation)

**For premium users:** Offer 70B model option
- Charge 5x more ($0.015 per response)
- Deliver superior quality

---

## 2. Context Window Size vs Memory vs Latency

| Window | Tokens | Memory | Latency | Benefit |
|--------|--------|--------|---------|---------|
| 2K     | 2,048  | 2GB    | 100ms   | Basic context |
| 4K     | 4,096  | 4GB    | 200ms   | **Good** |
| 8K     | 8,192  | 8GB    | 400ms   | Extended |
| 32K    | 32K    | 32GB   | 1.5s    | Full convos |
| 128K   | 128K   | 100GB  | >5s     | Impractical |

**Tradeoff:**
- **Smaller windows**: Faster, less memory, less context
- **Larger windows**: More context, but exponentially slower/costlier

**Decision:** 4K context window
- Retrieve last 10-15 messages (sufficient for most conversations)
- Balance between context and performance
- Compress old messages or summarize long conversations

**Implementation:**
```
Total prompt tokens = system_prompt (50) + context (2000) + user_input (500) = 2550
Response tokens = 1000
Total = 3550 tokens well within 4K limit
```

---

## 3. Streaming vs Batch Response

**Streaming:**
- Pros: Better UX (tokens appear instantly), lower perceived latency
- Cons: More complex, WebSocket overhead, client buffering

**Batch:**
- Pros: Simpler implementation, easier to cache, can optimize generation
- Cons: Poor UX (wait for complete response), harder to parallelize

**Hybrid Approach (Recommended):**
```
Option 1 (Chat UI): Stream tokens (WebSocket)
Option 2 (API): Batch response (HTTP POST)
Option 3 (Mobile): Stream with gRPC (more efficient)
```

**Decision:** Stream for web clients, offer both
- Web: WebSocket streaming
- API: Optional streaming parameter
- Mobile: gRPC for efficiency

---

## 4. Self-Hosted vs API-Based

| Approach | Upfront | Marginal | Scale | Control |
|----------|---------|----------|-------|---------|
| **OpenAI API** | $0 | $0.015/response | Easy | None |
| **Hugging Face API** | $0 | $0.001/response | Medium | Some |
| **Self-hosted (13B)** | $1M | $0.001/response | Hard | Full |
| **Self-hosted (70B)** | $2M | $0.005/response | Hard | Full |

**Breakeven analysis:**
```
Self-hosted annual cost: $1M (upfront amortized) + $500K (ops) = $1.5M
OpenAI API annual cost: $0.015/req × 100M req = $1.5M

At 100M requests/year → Breakeven
Above 100M → Self-hosted is cheaper
Below 100M → Use OpenAI API
```

**Decision:** Start with OpenAI API, migrate to self-hosted at 10M+ users
- Phase 1 (0-1M users): OpenAI API
- Phase 2 (1-5M users): Hybrid (OpenAI + self-hosted small model)
- Phase 3 (5M+ users): Fully self-hosted

**Reasons:**
- Reduces operational complexity
- Lower initial investment
- Time to market is critical
- Scale gradually as revenue grows

---

## 5. Fine-tuning vs Prompt Engineering

| Approach | Cost | Time | Quality Gain | Flexibility |
|----------|------|------|--------------|-------------|
| Prompt Engineering | Low | 1 week | +10% | High |
| LoRA Fine-tuning | Medium | 2 weeks | +20% | Medium |
| Full Fine-tuning | High | 4 weeks | +30% | Low |

**Tradeoff:**
- **Prompt engineering**: Cheap and fast, but limited gains
- **Fine-tuning**: Expensive and slow, but better specialized performance

**Decision:** Start with prompt engineering
- Invest in high-quality system prompts
- Use few-shot examples in context
- Fine-tune only if needed for specific use case

**When to fine-tune:**
- Cost savings justify it (>10M requests/year)
- Domain-specific knowledge needed (medical, legal)
- Consistent style/tone critical (branded assistant)

---

## 6. Retrieval-Augmented Generation (RAG) vs Pure LLM

| Approach | Hallucinations | Freshness | Cost | Complexity |
|----------|---|---|---|---|
| Pure LLM | High (5-10%) | Stale (training cutoff) | Low | Low |
| RAG with KB | Low (1-2%) | Fresh (if indexed) | Medium | High |
| Hybrid | Low (2-3%) | Mixed | Medium-High | Medium |

**Tradeoff:**
- **Pure LLM**: Fast, simple, but unreliable for facts
- **RAG**: Reliable for facts, but slower and more complex

**Decision:** Hybrid approach
- Use pure LLM for creative/conversational queries
- Use RAG for factual/knowledge queries
- Detect query type and route appropriately

**Implementation:**
```
User query: "What is the capital of France?"
→ [Query classifier: factual]
→ [Retrieve from knowledge base]
→ [Augment prompt with retrieved facts]
→ [Generate response with citations]

User query: "Write a poem about roses"
→ [Query classifier: creative]
→ [Skip retrieval]
→ [Pure LLM generation]
```

---

## 7. Safety: Permissive vs Restrictive

**Permissive (allow most content):**
- Pros: More useful, fewer false positives
- Cons: Risk of misuse (misinformation, jailbreaks)

**Restrictive (filter aggressively):**
- Pros: Safer, reduces liability
- Cons: User frustration, competitive disadvantage

**Balanced Approach (Recommended):**
```
Input Filtering:
- Block clear harmful requests
- Allow nuanced/philosophical discussions
- Manual review for edge cases

Output Filtering:
- Remove explicit violence/graphic content
- Remove personal identifying information
- Flag uncertainty for sensitive topics
- Allow user to override with warning
```

**Decision:** Balanced with transparency
- Clear community guidelines
- Explain why requests are blocked (when possible)
- Appeal process for legitimate use cases
- Regular human review of filtering

---

## 8. Cost Optimization: Caching vs Compute

**No caching:**
- Cost per response: $0.001 (inference only)
- Cache hardware: $0
- Hit rate: 0%
- Total cost: $100K/year for 100M requests

**With semantic cache (10% hit):**
- Cost per response: $0.0009 (90% × $0.001 + 10% × free)
- Cache hardware: $10K
- Hit rate: 10%
- Total cost: $90K/year (saves $10K)

**Aggressive caching (30% hit):**
- Cost per response: $0.0007 (70% × $0.001)
- Cache hardware: $30K
- Hit rate: 30%
- Total cost: $70K/year (saves $30K)

**Tradeoff:**
- Each 1% improvement in hit rate saves $1K/year
- Storage cost: ~$50 per 100K cached responses
- Breakeven: Hit rate > 2%

**Decision:** Implement semantic cache aggressively
- Target 20-30% hit rate
- Easy quick wins (FAQ questions, common patterns)
- Redis + vector search + regular cleanup

---

## 9. Conversation Length: Keep vs Summarize

**Keep all messages:**
- Pros: Full context, better responses
- Cons: Token budget consumed, higher latency/cost

**Summarize old messages:**
- Pros: Saves tokens, faster
- Cons: Information loss, degraded quality

**Tradeoff:**
```
200 message conversation = 20K tokens
- First 100 messages: Summarize to ~500 tokens
- Last 100 messages: Keep in full (~10K tokens)
- Total: ~10.5K tokens vs 20K (47% savings)
```

**Decision:** Summarize on-demand
- Keep full context for first 50 messages
- Summarize when conversation exceeds context window
- Use extractive + abstractive summarization
- Keep summary in separate message in history

---

## 10. Real-time vs Offline Analytics

| Approach | Latency | Accuracy | Cost | Insights |
|----------|---------|----------|------|----------|
| Real-time | Immediate | 100% | High | Live metrics |
| Batch (hourly) | 1 hour | 100% | Medium | Trends |
| Sampled (1%) | Immediate | 99% | Low | Estimated |

**Tradeoff:**
- **Real-time**: Accurate but expensive
- **Sampled**: Cheap and fast but less accurate

**Decision:** Hybrid
- **Critical metrics** (revenue, errors): Real-time
- **Quality metrics** (satisfaction, latency): Batch every hour
- **Exploratory**: Sample-based (1%)

**Implementation:**
```
Real-time (Redis):
- User message count
- API error rate
- GPU utilization

Batch (Hourly):
- Average response latency
- User satisfaction
- Model accuracy

Sampled (1%):
- Token distribution
- Common topics
- Rare edge cases
```

---

## Decision Summary Matrix

| Decision | Chosen | Alternative | Reasoning |
|----------|--------|-------------|-----------|
| **Model** | 13B | 70B | Balance quality and scale |
| **Context** | 4K | 8K | Sufficient + faster |
| **Response** | Stream | Batch | Better UX |
| **Hosting** | OpenAI → self-hosted | All in-house | Cost-effective path |
| **Optimization** | Prompt eng + RAG | Pure fine-tuning | Faster iteration |
| **Safety** | Balanced + appeal | Restrictive | User trust + safety |
| **Cache** | Semantic 20-30% | No cache | Cost savings |
| **History** | Summarize on-demand | Keep all | Token efficiency |
| **Analytics** | Hybrid | Real-time only | Cost control |
| **Latency Target** | P99 < 2s | P99 < 1s | Acceptable vs cost |

---

## Key Principles

1. **Start simple**, optimize later
   - Use OpenAI API initially
   - Migrate to self-hosted only when cost-justified

2. **Optimize high-impact items first**
   - Semantic cache (10-20% cost savings)
   - Prompt optimization (improve UX without cost)
   - Model size selection (fundamental tradeoff)

3. **Don't over-engineer**
   - Avoid RAG until needed (extra complexity)
   - Avoid fine-tuning until significant benefit
   - Batch analytics is sufficient for most use cases

4. **Iterate based on data**
   - Track actual hit rates, latencies, user satisfaction
   - Adjust cache size based on metrics
   - Change model size if latency/cost targets missed

5. **Plan for scale**
   - Use decisions that don't lock in
   - OpenAI API can be swapped for self-hosted
   - Caching adds value regardless of model
   - Architecture supports 100x growth
