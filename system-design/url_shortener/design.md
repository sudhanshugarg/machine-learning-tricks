# URL Shortener System Design

## Problem Statement

Design a URL shortening service (like bit.ly, tinyurl) that converts long URLs into short, easy-to-share links. The system should handle millions of URLs and serve redirect requests at high throughput.

## Functional Requirements

1. **Shorten URL**: Convert long URL → short URL (short_code)
2. **Redirect**: Given short_code → retrieve original URL and redirect (302)
3. **Custom aliases**: Allow users to specify custom short codes
4. **URL expiration**: Support TTL-based URL expiration
5. **Analytics**: Track click counts, referrer info, geographic data
6. **User accounts**: Support user authentication and URL history

## Non-Functional Requirements

1. **Scale**:
   - 1M new URLs per day (12 URLs/second average)
   - 100M redirects per day (1,200 requests/second average)
   - Peak: 10x normal traffic

2. **Performance**:
   - Redirect latency: <100ms (p99)
   - Create URL latency: <500ms
   - Availability: 99.9% uptime

3. **Storage**:
   - Store URLs, mappings, analytics data
   - ~1M URLs/day × 365 days × ~500 bytes = ~183 GB/year
   - Consider compression and archival

4. **Uniqueness**:
   - Short codes must be globally unique
   - High collision resistance

---

## High-Level Design

```
User Request
    ↓
[Load Balancer]
    ↓
[API Servers]
    ├─ Shorten service
    ├─ Redirect service
    └─ Analytics service
    ↓
[Cache Layer (Redis)]
    ↓
[Database]
    ├─ Primary DB (Master)
    └─ Replicas (Read)
    ↓
[Analytics/Logging]
```

---

## Detailed Components

### 1. API Endpoints

**POST /shorten**
```
Request:
{
  "long_url": "https://example.com/very/long/path",
  "custom_alias": "my_link"  // optional
}

Response:
{
  "short_url": "http://short.io/abc123",
  "short_code": "abc123",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**GET /redirect/<short_code>**
```
Response: 302 Redirect to original URL
Headers: Location: https://example.com/very/long/path
```

**GET /analytics/<short_code>**
```
Response:
{
  "clicks": 1523,
  "unique_visitors": 892,
  "referrers": {...},
  "devices": {...},
  "geolocation": {...}
}
```

### 2. Short Code Generation

**Option 1: Collision-free unique ID (ZooKeeper/Zookeeper)**
- Central service generates unique IDs
- Convert ID to base62 string
- Pros: Guaranteed unique, simple
- Cons: Single point of failure, coordination overhead

**Option 2: UUID with collision handling**
- Generate random base62 string (6-7 chars)
- Check database for collision
- Retry if collision
- Pros: Distributed, no coordination
- Cons: Possible collisions, retry logic

**Option 3: Distributed unique ID generator (Snowflake-like)**
- Timestamp (ms) + Worker ID + Sequence number
- 64-bit ID → convert to base62
- Pros: High throughput, no collisions
- Cons: More complex

**Recommendation:** Use Option 3 for large scale

### 3. Database Design

**URL Mapping Table**
```
CREATE TABLE url_mappings (
  id BIGINT PRIMARY KEY,
  short_code VARCHAR(10) UNIQUE NOT NULL,
  long_url VARCHAR(2048) NOT NULL,
  user_id BIGINT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP,
  is_active BOOLEAN DEFAULT TRUE,
  custom_alias BOOLEAN,
  INDEX (short_code),
  INDEX (user_id, created_at),
  INDEX (expires_at)
);
```

**Analytics Table**
```
CREATE TABLE url_clicks (
  id BIGINT PRIMARY KEY,
  short_code VARCHAR(10) NOT NULL,
  user_id_visitor VARCHAR(100),  // Anonymous identifier
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  referrer VARCHAR(2048),
  user_agent VARCHAR(500),
  ip_address VARCHAR(45),
  country_code VARCHAR(2),
  device_type VARCHAR(20),
  INDEX (short_code, timestamp),
  INDEX (timestamp)
);
```

**Users Table**
```
CREATE TABLE users (
  id BIGINT PRIMARY KEY,
  email VARCHAR(255) UNIQUE,
  username VARCHAR(100) UNIQUE,
  password_hash VARCHAR(255),
  created_at TIMESTAMP,
  INDEX (email)
);
```

### 4. Caching Strategy

**Redis Cache:**
- Key: `short_code`
- Value: Original URL, metadata, expiration
- TTL: 24 hours or custom expiration
- Size: Keep hot URLs (80/20 rule)

**Cache Flow:**
```
GET /redirect/abc123
  ↓
Check Redis
  ├─ Hit → Return URL
  └─ Miss → Query DB → Update cache → Return URL
```

**Cache Invalidation:**
- TTL-based expiration (24 hours)
- Explicit deletion on URL expiration
- Update on custom settings change

### 5. Load Balancing

**For high throughput:**
- Round-robin load balancer (e.g., Nginx)
- Consistent hashing for cache locality
- Geographic routing (CDN) for redirect latency

**Redirect servers:** Keep minimal logic, use cache heavily
**Shorten servers:** Handle more complex logic (DB writes, ID generation)

### 6. Concurrency & Uniqueness

**Database constraints:**
- Unique constraint on `short_code`
- Unique constraint on `(user_id, custom_alias)` for custom aliases

**Handling collisions:**
- Application level: Retry with new code
- Database level: Retry logic + exponential backoff

### 7. Analytics

**Real-time counting:**
- Use Redis counters for real-time clicks
- Periodic flush to database (batch writes)
- HyperLogLog for unique visitor counts

**Time-series data:**
- Store hourly/daily aggregates
- Use time-series DB (InfluxDB, TimescaleDB)
- Keep raw events for 30 days, aggregates for 1 year

---

## API Rate Limiting

**Per user:**
- 100 URLs/hour (creating new short URLs)
- 10,000 requests/minute (redirects)

**Per IP:**
- 1,000 requests/minute to prevent abuse

**Implementation:**
- Token bucket algorithm with Redis
- Return 429 Too Many Requests

---

## Scalability Considerations

### 1. Database Sharding

**Shard by short_code:**
```
shard_id = hash(short_code) % num_shards
```
- Each shard handles ~1/n URLs
- Enables horizontal scaling
- Trade-off: Increased complexity

### 2. Caching

- Most URLs are accessed rarely (long tail)
- Cache hot URLs (top 1% = 80% of traffic)
- Use consistent hashing for cache locality

### 3. Asynchronous Processing

- Analytics writes: Queue to message broker
- Email notifications: Async workers
- Batch operations: Background jobs

### 4. CDN for Redirects

- Store redirect mappings in edge locations
- Serve from closest server to user
- Dramatically reduce latency

---

## Security Considerations

1. **URL Validation**: Sanitize, validate, prevent phishing
2. **Rate Limiting**: Prevent abuse and DoS attacks
3. **HTTPS**: All communications encrypted
4. **User Auth**: API keys, OAuth for user accounts
5. **Malicious URLs**: Scan against known malware databases
6. **Data Privacy**: GDPR compliance for analytics data
7. **SQL Injection**: Use parameterized queries
8. **Short Code Length**: 6-7 characters (~62^6 ≈ 56 billion combinations)

---

## Key Tradeoffs

| Aspect | Choice | Tradeoff |
|--------|--------|----------|
| **ID Generation** | Snowflake-like | Complexity vs guaranteed uniqueness |
| **Caching** | Redis with TTL | Memory cost vs latency |
| **Sharding** | By short_code | Complexity vs scalability |
| **Analytics** | Async batch | Eventual consistency vs load reduction |
| **Storage** | Relational DB | Structured queries vs horizontal scaling |
| **Short code length** | 6-7 chars | Collision risk vs URL length |

---

## Estimated Capacity

**Daily growth:**
- 1M new URLs
- 100M redirects
- ~183 GB storage/year

**Storage per URL:**
- Short code: 10 bytes
- Long URL: 2000 bytes
- Metadata: 100 bytes
- **Total: ~2.1 KB per mapping**

**For 1 billion URLs:**
- Storage: ~2.1 TB
- Indexes: ~500 GB
- **Total: ~2.6 TB**

---

## Monitoring & Observability

1. **Metrics:**
   - Requests/second (shorten vs redirect)
   - Latency (p50, p99, p99.9)
   - Cache hit rate
   - Error rates

2. **Logging:**
   - API request logs
   - Database query logs
   - Error/exception logs

3. **Alerting:**
   - High error rate (>1%)
   - Latency SLA breach
   - Cache hit rate drop
   - Database replication lag
