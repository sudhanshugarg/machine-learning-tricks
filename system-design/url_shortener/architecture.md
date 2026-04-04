# URL Shortener Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Requests                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │  CDN / Load Balancer   │
            │    (Geographic based)  │
            └──────────┬─────────────┘
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
    ┌───────────────────────────────────────┐
    │     API Server Cluster                │
    │  ┌─────────────────────────────────┐  │
    │  │  - Shorten Service              │  │
    │  │  - Redirect Service             │  │
    │  │  - Analytics Service            │  │
    │  │  - Auth Service                 │  │
    │  └─────────────────────────────────┘  │
    └────┬──────────────────────────────┬───┘
         │                              │
    ┌────↓────────────────────────────┬─↓───────┐
    │     ┌──────────────────────────┐ │        │
    │     │   Redis Cache Layer      │ │        │
    │     │  - Hot URL mappings      │ │        │
    │     │  - Analytics counts      │ │        │
    │     │  - Rate limit counters   │ │        │
    │     │  - Session data          │ │        │
    │     └──────────────────────────┘ │        │
    │                                   │        │
    │     ┌──────────────────────────┐ │        │
    │     │  Primary Database        │ │        │
    │     │  (Master)                │ │        │
    │     │  - URL mappings table    │ │        │
    │     │  - Users table           │ │        │
    │     │  - Settings              │ │        │
    │     └──────────────────────────┘ │        │
    │              ↓                    │        │
    │     ┌──────────────────────────┐ │        │
    │     │  Database Replicas       │ │        │
    │     │  (Read-only)             │ │        │
    │     │  - 2-3 replicas          │ │        │
    │     └──────────────────────────┘ │        │
    │                                   │        │
    │     ┌──────────────────────────┐ │        │
    │     │  Analytics DB            │ │        │
    │     │  (Time-series)           │ │        │
    │     │  - Clicks & events       │ │        │
    │     │  - Aggregated metrics    │ │        │
    │     └──────────────────────────┘ │        │
    └─────────────────────────────────┬─────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
            ┌─────────────────┐  ┌──────────────┐  ┌─────────────┐
            │ Message Queue   │  │ Unique ID    │  │   Logging/  │
            │ (Kafka/RabbitMQ)│  │ Generator    │  │  Monitoring │
            │ - Analytics     │  │ (Snowflake)  │  │             │
            │ - Notifications │  │              │  │             │
            └─────────────────┘  └──────────────┘  └─────────────┘
                    │
                    ↓
            ┌─────────────────┐
            │ Background Jobs │
            │ - Analytics     │
            │ - Cleanup       │
            │ - Notifications │
            └─────────────────┘
```

---

## Component Details

### 1. API Servers

**Technologies:** Golang, Python, Java
- Lightweight, stateless
- Handle request routing
- Input validation
- Rate limiting enforcement

**Key responsibilities:**
- Request parsing and validation
- Authentication and authorization
- Error handling and response formatting
- Health checks

### 2. Shorten Service

**Workflow:**
```
1. Receive long URL and user info
2. Validate URL format and length
3. Check for duplicate (optional)
4. Generate unique short code:
   - Call Unique ID Generator
   - Convert to base62
5. Store in database
6. Update cache
7. Return short URL
```

**Latency target:** <500ms

### 3. Redirect Service

**Workflow:**
```
1. Extract short code from URL
2. Look up in Redis cache
3. If hit: return original URL
4. If miss:
   - Query database (read replica)
   - Check expiration
   - Update cache
   - Return URL
5. Log redirect event (async)
6. Return 302 redirect response
```

**Latency target:** <100ms (p99)

### 4. Unique ID Generator (Snowflake-like)

**Components:**
```
64-bit ID:
[timestamp (41 bits)] [worker_id (10 bits)] [sequence (12 bits)]

- Timestamp: milliseconds since epoch
- Worker ID: 0-1023 (supports 1024 servers)
- Sequence: counter for same millisecond (4096 IDs/ms)
```

**Conversion to short code:**
```
id = 123456789
base62(id) = "abc123"  // 6-7 character string
```

**Benefits:**
- No database calls for ID generation
- Distributed (no single point of failure)
- Time-ordered (cache locality)
- No collisions

### 5. Redis Cache

**Data structures:**
```
String: short_code -> original_url
Hash: short_code -> {url, user_id, created_at, expires_at}
Counter: short_code:clicks -> count
HyperLogLog: short_code:visitors -> unique_count
```

**TTL strategy:**
- Default: 24 hours
- Custom expiration: Based on URL setting
- Refresh on access: Extend TTL

**Cache size:**
- ~50GB for hot URLs (1% of total)
- Multiple Redis nodes with consistent hashing
- Sharded by short_code

### 6. Database

**Primary database:** PostgreSQL or MySQL
- Handles writes (short URL creation)
- Supports read replicas
- ACID transactions
- Replication lag: <100ms

**Read replicas:**
- 2-3 replicas for read scaling
- Redirect service queries replicas
- Slightly stale data acceptable

**Indexes:**
```
PRIMARY KEY (id)
UNIQUE (short_code)
INDEX (user_id, created_at)
INDEX (expires_at)
```

### 7. Analytics Store

**Time-series database:** InfluxDB or TimescaleDB
- Optimized for metrics
- High write throughput
- Time-range queries

**Data collected:**
- Clicks per short code
- Unique visitors
- Referrers
- Geographic data
- Device information

**Data retention:**
- Raw events: 30 days
- Hourly aggregates: 1 year
- Daily aggregates: 3 years

### 8. Message Queue

**Technology:** Kafka or RabbitMQ

**Topics/Queues:**
- `url_created`: New short URLs
- `url_clicked`: Redirect events
- `url_expired`: Expiration events

**Consumers:**
- Analytics processor
- Notification sender
- Logging service

---

## Data Flow Patterns

### Create Short URL

```
POST /shorten
↓
[Load Balancer]
↓
[API Server]
├─ Validate URL
├─ Check rate limit (Redis)
├─ Get unique ID (Snowflake ID Gen)
├─ Insert into DB (Primary)
├─ Update cache (Redis)
├─ Publish event (Kafka)
└─ Return response
```

**Latency breakdown:**
- ID generation: 1ms
- URL validation: 5ms
- DB write: 10ms
- Cache update: 5ms
- Response time: 25ms average

### Redirect to Original URL

```
GET /r/abc123
↓
[Load Balancer / CDN]
↓
[API Server]
├─ Check Redis cache
│  ├─ Hit (95% case): 5ms
│  └─ Miss (5% case):
│     ├─ Query DB replica: 20ms
│     ├─ Update cache: 5ms
├─ Check expiration
├─ Publish click event (async, Kafka)
└─ Return 302 with Location header
```

**Latency breakdown:**
- Cache lookup: 2ms (average)
- DB query (on miss): 20ms
- Redirect response: 1ms
- **P50: 5ms, P99: 50ms**

---

## Handling Scale

### 1. Sharding Strategy

**Option A: Shard by short_code**
```
shard_id = hash(short_code) % num_shards

Example:
short_code = "abc123"
hash("abc123") % 4 = 2
→ queries go to shard 2
```

**Benefits:**
- Even distribution
- Easy to add shards (consistent hashing)
- Cache locality

### 2. Database Replication

**Master-Slave replication:**
```
Write traffic → Primary DB
Read traffic → Read Replicas

Master (1 node)
  ↓ (async replication lag: <100ms)
Slaves (2-3 nodes)
```

### 3. Cache Locality

**Consistent hashing for cache:**
```
Redis node = hash(short_code) % num_cache_nodes

Benefits:
- Cache hits for same code on same server
- Minimal data movement on scaling
- Reduces network traffic
```

### 4. Geographic Distribution

**Multi-region setup:**
```
Region 1 (US-East)         Region 2 (Europe)
├─ API Servers             ├─ API Servers
├─ Redis                   ├─ Redis
├─ Primary DB              ├─ Replica DB
└─ Analytics               └─ Analytics

Global Load Balancer
  ↓ (routes to nearest region)
```

---

## Failure Scenarios

### 1. Cache Miss / Cache Failure

**Impact:** Slight latency increase
**Handling:**
- Query database directly
- Rebuild cache from DB
- Circuit breaker to prevent cascading

### 2. Database Replica Lag

**Impact:** Stale data in rare cases
**Handling:**
- Read from master on critical operations
- Monitor replication lag
- Alert if lag > threshold

### 3. ID Generator Failure

**Impact:** Cannot create new short URLs
**Handling:**
- Multiple ID generator instances
- Automatic failover
- Fallback: Database sequence

### 4. Complete Database Failure

**Impact:** Service degradation
**Handling:**
- Read from cache (stale data)
- Queue writes to replay later
- Failover to replica
- Manual intervention needed

---

## Performance Optimization

### 1. Connection Pooling
- Database connection pool: 100-1000 connections
- Redis connection pool: 50-500 connections
- Reuse connections, minimize overhead

### 2. Query Optimization
```sql
-- Efficient query for redirect
SELECT long_url FROM url_mappings
WHERE short_code = ? AND (expires_at IS NULL OR expires_at > NOW())
LIMIT 1;
```

### 3. Compression
- Compress URLs before storage (when applicable)
- Cache compressed data
- Decompress on retrieval

### 4. Batching
- Batch analytics writes
- Write 100-1000 events per batch
- Reduces database load

---

## Security Implementation

1. **HTTPS:** TLS 1.3 for all communications
2. **Rate Limiting:** Token bucket, 1000 req/min per IP
3. **Input Validation:** URL length < 2048, allowed protocols
4. **SQL Injection:** Parameterized queries, ORM
5. **DDoS Protection:** WAF, rate limiting, traffic shaping
6. **Authentication:** API keys + JWT for sessions
7. **Encryption:** At rest (database), in transit (TLS)

---

## Deployment Architecture

**Container-based:**
- Docker containers for each service
- Kubernetes orchestration
- Auto-scaling based on metrics

**CI/CD:**
- Git-based workflow
- Automated testing
- Blue-green deployment
- Canary releases for new versions
