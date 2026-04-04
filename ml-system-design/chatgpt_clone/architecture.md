# ChatGPT Clone - System Architecture

## Complete System Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    User Interface Layer                        │
│   (Web browser, Mobile app, API clients)                       │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │   CDN / Global Load Balancer │
        │   (Cloudflare, CloudFront)   │
        └──────────────┬───────────────┘
                       │
        ┌──────────────────────────────┐
        │    API Gateway / Router       │
        │  - Auth & rate limiting       │
        │  - Request validation         │
        │  - Streaming setup            │
        └──────────────┬───────────────┘
                       │
      ┌────────────────┼────────────────┐
      ↓                ↓                ↓
  [User Svc]   [Chat Svc]        [Admin Svc]
      │                │                │
      └────────────────┼────────────────┘
                       │
        ┌──────────────────────────────┐
        │   Semantic Cache Layer        │
        │   (Redis + Embeddings)        │
        └──────────────┬───────────────┘
                       │
        ┌──────────────────────────────┐
        │   Request Processing          │
        │  - Context retrieval          │
        │  - Prompt building            │
        │  - Safety filtering           │
        └──────────────┬───────────────┘
                       │
        ┌──────────────────────────────┐
        │   Request Queue               │
        │   (Redis or Kafka)            │
        │   Priority queue with TTL    │
        └──────────────┬───────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ↓                  ↓                  ↓
[Model Server 1]  [Model Server 2]  [Model Server 3]
  8x A100 GPUs      8x A100 GPUs      8x A100 GPUs
    ├─ vLLM          ├─ vLLM          ├─ vLLM
    ├─ Batching      ├─ Batching      ├─ Batching
    └─ Streaming     └─ Streaming     └─ Streaming
    │                  │                  │
    └──────────────────┼──────────────────┘
                       │
        ┌──────────────────────────────┐
        │   Response Post-processing    │
        │  - Output filtering           │
        │  - Cache storage              │
        │  - Analytics                  │
        └──────────────┬───────────────┘
                       │
                       ↓
   ┌────────────────────────────────────┐
   │   Persistent Data Layer            │
   │                                    │
   │  ┌──────────────────────────────┐  │
   │  │  Primary Database            │  │
   │  │  (PostgreSQL)                │  │
   │  │  - Users                     │  │
   │  │  - Conversations             │  │
   │  │  - Messages                  │  │
   │  └──────────────────────────────┘  │
   │                                    │
   │  ┌──────────────────────────────┐  │
   │  │  Vector DB                   │  │
   │  │  (Pinecone/Weaviate)         │  │
   │  │  - Embeddings                │  │
   │  │  - Retrieval index           │  │
   │  └──────────────────────────────┘  │
   │                                    │
   │  ┌──────────────────────────────┐  │
   │  │  Analytics DB                │  │
   │  │  (ClickHouse)                │  │
   │  │  - User metrics              │  │
   │  │  - Token usage               │  │
   │  └──────────────────────────────┘  │
   └────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ↓                  ↓                  ↓
[Logging]        [Monitoring]        [Async Jobs]
(ELK stack)      (Prometheus)        (Message queue)
```

---

## Component Deep Dive

### 1. API Gateway

**Responsibilities:**
- SSL/TLS termination
- Authentication (JWT/API keys)
- Rate limiting (token bucket)
- Request validation
- WebSocket upgrade for streaming

**Technologies:** Nginx, Kong, or custom Go service

**Rate limiting implementation:**
```
Per user per hour: 100 messages × 1000 tokens avg = 100K tokens
Peak burst: 10 messages in 10 seconds

Token bucket:
- Capacity: 100,000 tokens
- Refill rate: 100,000 tokens / 3600 seconds ≈ 28 tokens/second
- Burst allowance: 10,000 tokens (burst 10 messages)
```

---

### 2. Semantic Cache

**Architecture:**
```
User Query → Embed → Vector Search → Similarity Check → Cache Hit/Miss

Query: "What is machine learning?"
Embedding: [0.12, -0.34, 0.56, ...]
Search: Find similar embeddings (cosine similarity > 0.95)
Result: Return cached response if found
```

**Implementation details:**
- Embedding model: OpenAI text-embedding-3 or open-source (nomic-embed)
- Vector DB: Redis with RedisSearch, FAISS, or Weaviate
- Similarity threshold: 0.95 (high bar to ensure quality)
- Cache TTL: 24 hours

**Estimated hit rate:** 10-20% for diverse queries

**Cost-benefit:**
- Hit reduces inference cost by 100x
- Storage cost: ~$10/month per 100K cached responses
- Typical breakeven: 5% hit rate

---

### 3. Request Router

**Build conversation context:**
```python
messages = [
    {"role": "system", "content": system_prompt},
    # Retrieve last N messages from DB
    {"role": "user", "content": "Tell me about ML"},
    {"role": "assistant", "content": "Machine learning is..."},
    # Current message
    {"role": "user", "content": "What about deep learning?"}
]

# Apply safety filters
messages = apply_content_filters(messages)

# Apply prompt compression if too long
if token_count(messages) > 4000:
    messages = compress_messages(messages, target=3000)

# Route to model server
send_to_gpu(messages)
```

**Context management:**
- Sliding window: Keep last 10-20 messages
- Summarization: Compress old messages into summary
- Prompt compression: Remove redundant tokens

---

### 4. Model Serving (vLLM)

**Request flow:**
```
Batched Requests
    ↓
[Tokenization]
    ├─ Convert text → token IDs
    └─ Build attention masks
    ↓
[Model Inference]
    ├─ Forward pass (optimized with FlashAttention)
    ├─ Generate logits for next token
    └─ Sample next token
    ↓
[De-tokenization]
    ├─ Convert token ID → text
    └─ Update state
    ↓
[Streaming Output]
    ├─ Send tokens in real-time
    └─ 100ms batch intervals
```

**Optimization techniques:**

1. **Batching:**
   - Continuous batching: Add new requests while processing
   - Batch size: 32-64 requests
   - Target latency: 100ms per batch

2. **Quantization:**
   - INT8 or INT4 quantization
   - Reduces memory by 4-8x
   - Slight quality degradation (~1-2%)

3. **FlashAttention:**
   - Optimized attention computation
   - 2-4x faster than standard attention
   - No accuracy loss

4. **KV Cache optimization:**
   - Paged attention (vLLM's technique)
   - Reduces fragmentation
   - Supports longer sequences

---

### 5. Token Streaming

**WebSocket implementation:**
```javascript
// Client side
const ws = new WebSocket('wss://api.example.com/chat/stream');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'token') {
    // Append token to UI
    document.getElementById('response').innerText += data.token;
  } else if (data.type === 'done') {
    // Stop spinner, finalize response
    ws.close();
  }
};

ws.send(JSON.stringify({
  message: userInput,
  conversation_id: currentConvId
}));
```

**Server side:**
```python
# vLLM streaming
async def generate_response(request):
    async for token in model.generate_stream(
        request.message,
        max_tokens=2048,
        temperature=0.7
    ):
        await websocket.send(json.dumps({
            "type": "token",
            "token": token
        }))

    await websocket.send(json.dumps({"type": "done"}))
```

---

### 6. Database Schema

**Messages table with embeddings:**
```sql
CREATE TABLE messages (
  id UUID PRIMARY KEY,
  conversation_id UUID NOT NULL,
  role ENUM('user', 'assistant', 'system'),
  content TEXT NOT NULL,
  tokens INT,
  embedding VECTOR(1536),  -- For semantic search
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  -- Indexes
  INDEX idx_conversation (conversation_id, created_at),
  INDEX idx_embedding (embedding) USING IVFFLAT,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

**Conversation metrics:**
```sql
CREATE TABLE conversation_metrics (
  conversation_id UUID PRIMARY KEY,
  total_tokens INT,
  total_cost DECIMAL(10,6),
  user_satisfaction FLOAT,  -- 1-5 rating
  model_version VARCHAR(50),
  created_at TIMESTAMP,

  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

---

### 7. Load Balancing Strategy

**Request routing:**
```
Incoming Request
    ↓
[Consistent Hashing by User ID]
    └─ Ensures same user hits same cache instance
    ↓
[API Server Selection]
    ├─ Round-robin across healthy servers
    └─ Health check every 10 seconds
    ↓
[Queue Management]
    ├─ Priority: paid users get priority
    ├─ Age: older requests get priority
    └─ Max wait: 60 seconds, then return 429
```

**GPU assignment:**
```
Request → [Load Predictor]
          ├─ Predict inference time
          └─ Route to least loaded GPU
          ↓
          [GPU monitoring]
          ├─ Memory usage
          ├─ Queue depth
          └─ Temperature
```

---

## Failure Handling

### GPU Server Failure

```
Scenario: GPU server 1 crashes

1. Health check detects failure (3 consecutive failures)
2. Remove from load balancer
3. Re-queue pending requests to other servers
4. Trigger alert to ops team
5. User experience: Slight latency increase, no data loss
```

### Model Inference Timeout

```
Scenario: Request exceeds 30 second timeout

1. Cancel inference
2. Return partial response or error message
3. Log for analysis
4. Don't charge user

Possible causes:
- Malicious input (infinite loop)
- Model stuck/hanging
- System overload
```

### Database Failure

```
Scenario: PostgreSQL becomes unavailable

1. Stop accepting new conversations
2. Return error to users
3. Cache chat history in Redis (temporary)
4. Failover to replica (if configured)
5. RTO: <5 minutes, RPO: <1 minute
```

---

## Performance Optimization

### Caching Layers

**Layer 1: Browser Cache**
- Conversation list (5 minute TTL)
- User settings (1 hour TTL)

**Layer 2: CDN Cache**
- Static assets (images, scripts)
- API responses for non-personalized queries

**Layer 3: Redis Cache**
- Semantic cache (responses for similar queries)
- Session cache (JWT validation)
- Rate limit counters

**Layer 4: Application Cache**
- Model parameters (in GPU memory)
- Frequently used embeddings

---

### Query Optimization

**Retrieve conversation context efficiently:**
```sql
-- Fast retrieval of last N messages
SELECT id, role, content, tokens
FROM messages
WHERE conversation_id = ?
ORDER BY created_at DESC
LIMIT 20;

-- With index on (conversation_id, created_at)
-- Execution: ~5ms
```

**Pagination for long conversations:**
```sql
-- Get messages from offset
SELECT * FROM messages
WHERE conversation_id = ?
ORDER BY created_at DESC
LIMIT 50 OFFSET 50;

-- Cursor-based pagination (better)
SELECT * FROM messages
WHERE conversation_id = ?
  AND created_at < last_timestamp
ORDER BY created_at DESC
LIMIT 50;
```

---

## Monitoring Dashboard

**Real-time metrics:**
- Concurrent users
- Requests/second
- Average response latency (p50, p99)
- GPU utilization
- Queue depth
- Cache hit rate
- Error rate

**Alerts:**
- GPU memory > 90%
- Queue depth > 1000
- Cache hit rate < 5%
- Error rate > 1%
- TTFT > 2 seconds

---

## Deployment Strategy

**Blue-Green Deployment:**
```
Blue (current version)
  └─ 50% of traffic
Green (new version)
  └─ 50% of traffic (canary)

If metrics good: Shift all traffic to green
If metrics bad: Rollback to blue
```

**Model updates:**
- Download new model weight
- Start shadow inference (don't use responses)
- Compare quality metrics
- Gradual rollout (10% → 25% → 50% → 100%)
