# Short Code Generation Strategies

## Overview

Generating unique, collision-free short codes is one of the most critical components of a URL shortening service. This document explores three distinct approaches, each with different tradeoffs in terms of complexity, scalability, and reliability.

---

## Option 1: ZooKeeper-Based Unique ID Generation

### Architecture

ZooKeeper is a distributed coordination service that provides strong consistency guarantees. Each ID generation request acquires a distributed lock to fetch the next sequence number from a central counter.

```
Client Request
    ↓
[API Server]
    ↓
[ZooKeeper Cluster]
  ├─ Acquires distributed lock
  ├─ Reads current counter
  ├─ Increments counter
  ├─ Writes back to ZooKeeper
  └─ Releases lock
    ↓
Convert ID to base62
    ↓
Store in database
```

### Implementation Details

**Counter Storage in ZooKeeper:**
```
/url_shortener/id_counter = 10000000
```

**Generation Process:**
```python
def generate_short_code(zk_client):
    # Acquire distributed lock
    lock = zk_client.Lock("/url_shortener/id_lock")
    lock.acquire()

    try:
        # Read current counter
        counter, _ = zk_client.get("/url_shortener/id_counter")
        next_id = int(counter) + 1

        # Write back incremented counter
        zk_client.set("/url_shortener/id_counter", str(next_id))

        # Convert to base62
        short_code = base62_encode(next_id)
        return short_code
    finally:
        lock.release()

def base62_encode(num):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if num == 0:
        return alphabet[0]

    result = []
    while num > 0:
        result.append(alphabet[num % 62])
        num //= 62

    return ''.join(reversed(result))
```

### ID Space Coverage

```
Base62 encoding:
- 6-character code: 62^6 = 56,800,235,584 combinations (~56 billion)
- 7-character code: 62^7 = 3,521,614,606,208 combinations (~3.5 trillion)

For 1M URLs/day:
- 6-char codes cover 1M × 365 × 1000+ years ✓
- Sequential assignment ensures no collisions
```

### Pros

1. **Guaranteed uniqueness**: Sequential assignment eliminates collisions entirely
2. **Deterministic**: Same counter value always produces same short code
3. **Simple collision handling**: No retry logic needed
4. **Auditable**: Counter is a single source of truth
5. **Consistent ordering**: IDs are monotonically increasing
6. **Works well for small-medium scale**: Sufficient for billions of URLs

### Cons

1. **Single point of failure**: All ID generation depends on ZooKeeper cluster
2. **Coordination overhead**: Every ID requires distributed lock acquisition
3. **Network latency**: Lock acquisition adds 5-10ms per ID
4. **Throughput bottleneck**: ZooKeeper throughput ~1000-5000 writes/sec
   - Cannot handle 12+ URLs/sec easily without sharding
5. **Scaling complexity**: Requires ZooKeeper cluster management
6. **Not horizontally scalable**: Central counter is bottleneck

### Failure Scenarios

**ZooKeeper cluster unavailable:**
```
Impact: Cannot generate new short codes
Handling:
- Queue shorten requests during outage
- Fallback to pre-allocated ID ranges
- Retry with exponential backoff
- Manual intervention if outage > 5 minutes
```

**Lock contention:**
```
With 100 concurrent requests:
- Average lock wait time: 100ms (60-70ms per lock/unlock cycle)
- Effective throughput: ~10 IDs/sec (bottleneck!)
```

### When to Use

- **Small-medium scale systems** (< 100K URLs/day)
- **When you already have ZooKeeper** running for other purposes
- **When consistency is more important than throughput**
- **Development/testing environments**

### Comparison to Alternatives

| Metric | ZooKeeper | UUID+Collision | Snowflake |
|--------|-----------|----------------|-----------|
| Throughput | ~1K-5K/sec | ~50K/sec | ~100K/sec |
| Latency | 5-10ms | <1ms | 1-2ms |
| Collisions | 0 | Possible | 0 |
| Setup Complexity | Medium-High | Low | Medium |
| Scalability | Poor | Fair | Excellent |

---

## Option 2: UUID with Collision Handling

### Architecture

Generate random base62 strings and handle collisions at the database level using retry logic.

```
Client Request
    ↓
[API Server]
    ├─ Generate random base62 string (6-7 chars)
    ├─ Try to insert into database
    │
    ├─ Database constraint check
    │  ├─ Collision detected (UNIQUE violation)
    │  └─ Retry with new random string
    │
    └─ Success → Return short_code
```

### Implementation Details

**Generation Process:**
```python
def generate_short_code(db_connection, max_retries=5):
    for attempt in range(max_retries):
        # Generate random 6-character base62 string
        short_code = generate_random_base62(length=6)

        try:
            # Try to insert into database
            insert_url_mapping(db_connection, short_code, long_url)
            return short_code

        except IntegrityError:  # UNIQUE constraint violated
            if attempt == max_retries - 1:
                raise Exception("Failed to generate unique code after retries")

            # Backoff before retry
            wait_time = exponential_backoff(attempt)
            time.sleep(wait_time)

    return None

def generate_random_base62(length=6):
    import random
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return ''.join(random.choice(alphabet) for _ in range(length))

def exponential_backoff(attempt):
    """Exponential backoff: 10ms, 20ms, 40ms, 80ms"""
    base_wait = 0.01  # 10ms
    return base_wait * (2 ** attempt) + random.uniform(0, 0.01)
```

**Database Schema:**
```sql
CREATE TABLE url_mappings (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_code VARCHAR(10) UNIQUE NOT NULL,
    long_url VARCHAR(2048) NOT NULL,
    user_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,

    INDEX idx_short_code (short_code),
    INDEX idx_user_id (user_id),
    UNIQUE KEY uk_short_code (short_code)
);
```

### Collision Probability Analysis

**Birthday Paradox:**
```
With n = number of generated codes, S = size of code space:

P(collision) ≈ 1 - e^(-n²/2S)

For 6-character base62 codes:
- S = 62^6 = 56.8 billion
- After 1M codes generated:
  P(collision) ≈ 1 - e^(-(10^6)²/(2×56.8×10^9))
               ≈ 1 - e^(-0.0088)
               ≈ 0.87% collision probability
  → Average 1 collision per 115 URLs

For 7-character base62 codes:
- S = 62^7 = 3.5 trillion
- After 1B codes generated:
  P(collision) ≈ 0.00001%
  → Essentially negligible
```

**Practical Retry Statistics:**

```
Collision rate over time:
─────────────────────────────
URLs Generated  | 6-char codes | 7-char codes
─────────────────────────────
1M              | 0.87%        | <0.001%
10M             | 8.7%         | 0.01%
100M            | 87% (UNUSABLE)| 1%
1B              | UNUSABLE     | ~10% collisions
─────────────────────────────
```

### Pros

1. **No coordination needed**: Fully distributed, no central service
2. **Simple implementation**: Just generate random string + retry
3. **High throughput**: No lock contention, parallel operations
4. **Stateless servers**: Each API server independent
5. **Low latency**: No network calls for ID generation (~1ms)
6. **Easy to scale**: Add servers without coordination

### Cons

1. **Possible collisions**: Becomes problematic at scale (>100M URLs)
2. **Retry overhead**: Must handle database constraints
3. **Retry latency spikes**: On collision, request latency increases 10-50ms
4. **Database load**: Failed inserts still hit database
5. **Unpredictable latency**: p99 latency depends on collision rate
6. **Requires careful code length**: Too short → many collisions; too long → UX issue

### Retry Impact on Performance

```
Collision rate: 0.5% (1 in 200)

For 1000 concurrent requests:
- 995 succeed immediately: 10ms avg
- 5 collide, retry once more: 30ms avg

Overall impact:
- p50: 10ms (cache locality)
- p99: 30ms (one retry)
- p99.9: 50ms (two retries)

At scale (1B URLs), collision rate becomes prohibitive
```

### When to Use

- **Small-medium scale systems** (< 100M URLs)
- **When you want simplicity and no infrastructure overhead**
- **When most requests can tolerate occasional retry latency**
- **Development/MVP phase**
- **Prefer 7-character codes** to reduce collisions to acceptable levels

### Failure Scenarios

**Database unavailable:**
```
Impact: All requests fail (no fallback)
Handling: Queue requests, return 503 Service Unavailable
```

**High collision rate:**
```
At 1B URLs with 6-char codes:
- 87% collision rate
- Each request retries 5-10+ times
- Database gets hammered with failed inserts
- System effectively breaks down
```

---

## Option 3: Distributed Unique ID Generator (Snowflake-like)

### Architecture

A Snowflake-like distributed ID generator uses a combination of timestamp, worker ID, and sequence number to create globally unique IDs without coordination.

```
[64-bit ID Structure]
┌────────────────────────────────────────────────┐
│ Timestamp (41 bits) │ Worker ID (10 bits) │ Seq (12 bits) │
└────────────────────────────────────────────────┘
│                    │                     │
│                    │                     └─ 4096 IDs per ms
│                    └──────────────────────── 1024 workers
└─────────────────────────────────────────── 69 years of time
```

### Implementation Details

**Snowflake Generator:**
```python
import time
import threading

class SnowflakeIDGenerator:
    # Constants
    EPOCH = 1609459200000  # 2021-01-01 00:00:00 UTC (ms)
    TIMESTAMP_BITS = 41
    WORKER_ID_BITS = 10
    SEQUENCE_BITS = 12

    MAX_WORKER_ID = (1 << WORKER_ID_BITS) - 1  # 1023
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1    # 4095

    def __init__(self, worker_id):
        if worker_id > self.MAX_WORKER_ID:
            raise ValueError(f"Worker ID must be <= {self.MAX_WORKER_ID}")

        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()

    def generate(self):
        """Generate next unique ID"""
        with self.lock:
            current_time = int(time.time() * 1000)

            if current_time == self.last_timestamp:
                # Same millisecond
                self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE

                if self.sequence == 0:
                    # Sequence overflow, wait for next millisecond
                    current_time = self._wait_next_millis(self.last_timestamp)
            else:
                # New millisecond
                self.sequence = 0

            if current_time < self.last_timestamp:
                raise Exception("Clock went backwards!")

            self.last_timestamp = current_time

            # Build ID
            timestamp = (current_time - self.EPOCH) & ((1 << self.TIMESTAMP_BITS) - 1)
            id_value = (timestamp << (self.WORKER_ID_BITS + self.SEQUENCE_BITS)) | \
                       (self.worker_id << self.SEQUENCE_BITS) | \
                       self.sequence

            return id_value

    def _wait_next_millis(self, last_timestamp):
        """Wait until next millisecond"""
        current_time = int(time.time() * 1000)
        while current_time <= last_timestamp:
            current_time = int(time.time() * 1000)
        return current_time

# Usage
generator = SnowflakeIDGenerator(worker_id=1)
unique_id = generator.generate()  # e.g., 123456789012345
short_code = base62_encode(unique_id)  # e.g., "abc123def"
```

**Worker ID Assignment:**

```
[Configuration in each API server]

Worker ID assignments:
- API Server 1: worker_id = 1
- API Server 2: worker_id = 2
- ...
- API Server 1024: worker_id = 1024

Methods for assignment:
1. Static config file (for fixed servers)
2. ZooKeeper ephemeral node registration
3. Consul service registration
4. Kubernetes init container
5. Redis-based pool
```

### ID Space and Throughput Analysis

**Capacity:**
```
64-bit ID breakdown:
- 1 bit: unused (avoid negative numbers)
- 41 bits: timestamp (69.7 years from epoch)
- 10 bits: worker_id (1024 workers)
- 12 bits: sequence (4096 per millisecond)

Total capacity per node:
- 4096 IDs per millisecond
- 4.096 million IDs per second per worker
- Across 1024 workers: ~4.2 billion IDs/second (peak)

In practice:
- With 100 workers: ~409.6M IDs/second
- With 10 workers: ~40.96M IDs/second
- Far exceeds 12 URLs/second requirement
```

**Base62 Conversion:**
```
ID:        123456789012345
Base62:    "1pFLfM2"       (7 characters)

ID range coverage:
- 64-bit max: 18,446,744,073,709,551,615
- Base62(max): "LygHa16AHYF"  (11 characters)
- Our max: 2^63-1 = "AzL8n0Y58d9" (11 characters)
- Avg output: 6-7 characters (fits requirements!)
```

### Pros

1. **No collisions**: Mathematically guaranteed unique
2. **No coordination**: Fully distributed, stateless
3. **High throughput**: 4K IDs per millisecond per worker
4. **Low latency**: Generation < 1ms (single machine operation)
5. **Time-ordered**: IDs are roughly monotonic (cache locality)
6. **Scalable**: Add workers without changing algorithm
7. **No database overhead**: No constraint checks needed
8. **Deterministic**: Same inputs always produce same outputs
9. **Simple to implement**: Pure math, no external dependencies

### Cons

1. **Clock dependency**: Relies on accurate system time
2. **Worker ID management**: Need service to track worker IDs
3. **Leap second issues**: Rare but can cause IDs to go backwards
4. **More complex**: Requires bit manipulation understanding
5. **Time synchronization**: All servers must have NTP sync

### Handling Edge Cases

**Clock skew/NTP adjustments:**
```python
def generate_with_error_handling(self):
    """Handle clock adjustments gracefully"""
    with self.lock:
        current_time = int(time.time() * 1000)

        if current_time < self.last_timestamp:
            # Clock went backwards
            # Option 1: Throw error and rely on retry
            # raise Exception("Clock skew detected")

            # Option 2: Wait for clock to catch up
            # (This can happen during NTP corrections)
            time.sleep(self.last_timestamp - current_time + 1)
            current_time = int(time.time() * 1000)
```

**Worker ID collisions:**
```
Prevented by:
1. Centralized worker ID allocation (ZooKeeper/Consul)
2. Ephemeral nodes in coordination service
3. Automatic cleanup on worker death
```

### When to Use

- **Large-scale systems** (> 1M URLs/day) ← **RECOMMENDED**
- **When you want guaranteed uniqueness without collisions**
- **When you need predictable low latency**
- **When you want minimal operational overhead**
- **Production systems with high SLA requirements**

---

## Comparative Analysis

### Performance Comparison

```
┌──────────────────┬────────────────┬──────────────┬──────────────┐
│ Metric           │ ZooKeeper      │ UUID+Retry   │ Snowflake    │
├──────────────────┼────────────────┼──────────────┼──────────────┤
│ Throughput       │ 1K-5K/sec      │ 50K/sec      │ 4M/sec       │
│ Latency (p50)    │ 5-10ms         │ <1ms         │ <1ms         │
│ Latency (p99)    │ 50-100ms       │ 30ms         │ <1ms         │
│ Collision prob   │ 0%             │ 0.87% @ 1M   │ 0%           │
│ Retry needed     │ No             │ Yes          │ No           │
│ Coordination     │ Heavy          │ None         │ Light        │
│ Setup complexity │ Medium-High    │ Low          │ Medium       │
│ Operational OPS  │ Manage cluster │ None         │ Light        │
│ Single point?    │ Yes (cluster)  │ No           │ No           │
│ Scale to 1B URLs │ No             │ No           │ Yes          │
└──────────────────┴────────────────┴──────────────┴──────────────┘
```

### Operational Complexity

**ZooKeeper:**
- Deploy 3-5 node cluster
- Monitor ensemble health
- Handle failovers
- Tune timeouts
- Monitor lock contention

**UUID with Collision:**
- No external systems
- But need to handle retries gracefully
- Monitor collision rates
- Carefully choose code length

**Snowflake:**
- Allocate worker IDs
- Sync system clocks (NTP)
- Monitor time drift
- Graceful handling of clock skew

### Cost Analysis

**ZooKeeper:**
```
- Cluster: 3 nodes @ 2GB RAM, 2 CPU = 6GB RAM, 6 CPU
- Additional coordination overhead
- Estimated: $200-500/month
```

**UUID+Collision:**
```
- No additional infrastructure
- Database handles constraints
- Estimated: $0 (uses existing DB)
- Hidden cost: Database load from failed inserts
```

**Snowflake:**
```
- No additional infrastructure
- Lightweight (per-server, not centralized)
- Just NTP synchronization
- Estimated: $0 (uses existing servers)
```

---

## Recommendation Matrix

| System Scale | Primary Concern | Recommended | Rationale |
|---|---|---|---|
| MVP / < 1M URLs/day | Time to market | UUID+Collision | Simple, fast to implement |
| Growing (1M-100M URLs) | Balance scale/complexity | Snowflake | Best tradeoff |
| Large (> 100M URLs) | Reliability/performance | Snowflake | Only viable option at scale |
| Existing ZK ecosystem | Operational consistency | ZooKeeper | Leverage existing infra |
| Extreme scale (1T+ URLs) | Future-proof | Snowflake + sharding | Can shard by worker ID |

---

## Implementation Checklist

### ZooKeeper Option
- [ ] Deploy ZooKeeper cluster (3+ nodes)
- [ ] Implement distributed lock
- [ ] Add counter initialization
- [ ] Handle lock failures/timeouts
- [ ] Implement base62 encoding
- [ ] Add monitoring for lock wait times
- [ ] Test failover scenarios

### UUID+Collision Option
- [ ] Implement random base62 generation
- [ ] Add retry logic with backoff
- [ ] Handle UNIQUE constraint errors
- [ ] Choose appropriate code length (recommend 7)
- [ ] Monitor collision rates
- [ ] Add exponential backoff
- [ ] Test under high load

### Snowflake Option
- [ ] Design worker ID allocation
- [ ] Implement Snowflake generator
- [ ] Integrate worker ID service (ZK/Consul)
- [ ] Implement base62 conversion
- [ ] Test clock skew handling
- [ ] Monitor NTP synchronization
- [ ] Load test for throughput

---

## Migration Path

If starting with UUID+Collision and scaling to Snowflake:

```
1. Initial state: UUID+Collision
   └─ Works fine for < 100M URLs

2. Early warning signs:
   - Collision rate rising above 1%
   - Retry latency spikes in p99
   - Database constraint check load increasing

3. Migration steps:
   a. Deploy Snowflake generator alongside UUID
   b. New URLs use Snowflake, old ones still work
   c. No rewrite of existing URLs needed!
   d. Monitor for stability
   e. Deprecate UUID+Collision generation

4. Final state: Pure Snowflake
   └─ Scales to billions of URLs
```

---

## Conclusion

| **Choose ZooKeeper if:** | **Choose UUID+Collision if:** | **Choose Snowflake if:** |
|---|---|---|
| • Already using ZK | • Building MVP | • Building for scale |
| • < 10K URLs/day | • < 100M URLs total | • > 1M URLs/day |
| • Operational team familiar | • Want simplicity | • Want high performance |
| | • Can tolerate occasional latency spikes | • Need guaranteed uniqueness |
| | | • Want minimal coordination |

**For most production systems at scale, Snowflake-like ID generation is the recommended approach** due to its superior performance, simplicity, and lack of external dependencies while maintaining guaranteed uniqueness.
