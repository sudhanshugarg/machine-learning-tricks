# URL Shortener - Database Design

## Schema Design

### Core Tables

#### 1. Users Table

```sql
CREATE TABLE users (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  email VARCHAR(255) UNIQUE NOT NULL,
  username VARCHAR(100) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  is_active BOOLEAN DEFAULT TRUE,
  INDEX idx_email (email),
  INDEX idx_username (username),
  INDEX idx_created_at (created_at)
);
```

**Fields:**
- `id`: User identifier (primary key)
- `email`: Unique email address for login
- `username`: Display name
- `password_hash`: Hashed password (bcrypt/argon2)
- `created_at`: Registration timestamp
- `updated_at`: Last modification
- `is_active`: Account status

---

#### 2. URL Mappings Table (PRIMARY)

```sql
CREATE TABLE url_mappings (
  id BIGINT PRIMARY KEY,
  short_code VARCHAR(10) UNIQUE NOT NULL,
  long_url VARCHAR(2048) NOT NULL,
  user_id BIGINT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP NULL,
  is_active BOOLEAN DEFAULT TRUE,
  is_custom BOOLEAN DEFAULT FALSE,
  title VARCHAR(255),
  description TEXT,
  tags VARCHAR(500),

  -- Foreign key
  CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,

  -- Indexes
  UNIQUE INDEX idx_short_code (short_code),
  INDEX idx_user_id_created (user_id, created_at),
  INDEX idx_expires_at (expires_at),
  INDEX idx_created_at (created_at),
  INDEX idx_is_active (is_active)
);
```

**Field explanations:**
- `id`: Unique identifier from Snowflake ID generator
- `short_code`: 6-7 character unique code (base62 encoded)
- `long_url`: Original URL (up to 2048 characters)
- `user_id`: Creator of URL (NULL for anonymous)
- `created_at`: Timestamp of creation
- `expires_at`: Optional expiration time
- `is_active`: Soft delete flag
- `is_custom`: Whether user provided custom alias
- `title`: Optional user-provided title
- `description`: Optional metadata
- `tags`: Comma-separated tags for organization

**Index strategy:**
- `short_code`: UNIQUE for fast lookups
- `user_id, created_at`: Range queries by user
- `expires_at`: Cleanup queries for expired URLs
- `created_at`: Time-based queries

**Sharding consideration:**
- Can be sharded by `short_code % num_shards`
- Each shard has replica for read scaling

---

#### 3. URL Clicks Table (Analytics)

```sql
CREATE TABLE url_clicks (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  short_code VARCHAR(10) NOT NULL,
  user_id BIGINT,  -- If registered user
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  ip_address VARCHAR(45),  -- IPv4 or IPv6
  user_agent VARCHAR(500),
  referrer VARCHAR(2048),
  country_code VARCHAR(2),
  city VARCHAR(50),
  device_type ENUM('desktop', 'mobile', 'tablet', 'other'),
  browser VARCHAR(100),
  os VARCHAR(100),

  -- Indexes for common queries
  INDEX idx_short_code_timestamp (short_code, timestamp),
  INDEX idx_timestamp (timestamp),
  INDEX idx_user_id (user_id),
  INDEX idx_country_code (country_code)
);
```

**Field explanations:**
- `id`: Unique click event identifier
- `short_code`: Which URL was clicked
- `user_id`: Registered user (if known)
- `timestamp`: When click occurred
- `ip_address`: User's IP (for geolocation)
- `user_agent`: Browser/device info
- `referrer`: Source of traffic
- `country_code`: Derived from IP
- `device_type`: Mobile/desktop classification
- `browser`, `os`: Parsed from user_agent

**Index strategy:**
- `short_code, timestamp`: Most common query pattern
- `timestamp`: For time-range queries
- `user_id`: For per-user analytics

**Partitioning:**
- Partition by date (monthly or daily)
- Old partitions can be archived/deleted
- Improves query performance on large table

---

#### 4. URL Stats Table (Aggregated)

```sql
CREATE TABLE url_stats (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  short_code VARCHAR(10) NOT NULL,
  stat_date DATE NOT NULL,
  stat_hour INT,  -- 0-23 for hourly, NULL for daily
  click_count INT DEFAULT 0,
  unique_visitors INT DEFAULT 0,

  UNIQUE INDEX idx_code_date_hour (short_code, stat_date, stat_hour),
  INDEX idx_short_code (short_code),
  INDEX idx_stat_date (stat_date)
);
```

**Field explanations:**
- `short_code`: Which URL
- `stat_date`: Date of stats
- `stat_hour`: Hour of day (for granular stats)
- `click_count`: Total clicks
- `unique_visitors`: HyperLogLog count

**Strategy:**
- Hourly aggregates: Keep for 30 days
- Daily aggregates: Keep for 1 year
- Use materialized views or batch jobs to compute

---

#### 5. URL Settings Table (Optional)

```sql
CREATE TABLE url_settings (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  short_code VARCHAR(10) UNIQUE NOT NULL,
  allow_analytics BOOLEAN DEFAULT TRUE,
  password_protected BOOLEAN DEFAULT FALSE,
  password_hash VARCHAR(255),
  custom_domain VARCHAR(255),
  redirect_code INT DEFAULT 302,  -- 301, 302, 307

  CONSTRAINT fk_short_code FOREIGN KEY (short_code)
    REFERENCES url_mappings(short_code) ON DELETE CASCADE,

  INDEX idx_short_code (short_code)
);
```

**Field explanations:**
- `allow_analytics`: Whether to track clicks
- `password_protected`: Require password to redirect
- `custom_domain`: Custom domain for short URL
- `redirect_code`: HTTP status (301 permanent, 302 temporary)

---

## Query Patterns

### Frequently Used Queries

**1. Redirect Query (Most Critical)**
```sql
-- Cache in Redis, fallback to this
SELECT long_url
FROM url_mappings
WHERE short_code = ?
  AND is_active = TRUE
  AND (expires_at IS NULL OR expires_at > NOW())
LIMIT 1;
```

**Performance:**
- Indexed lookup on `short_code`
- Typical latency: 5-20ms
- Read from replica is acceptable

---

**2. Create Mapping**
```sql
INSERT INTO url_mappings
(id, short_code, long_url, user_id, created_at, expires_at)
VALUES (?, ?, ?, ?, NOW(), ?);
```

**Performance:**
- Write to primary database only
- ID generated by Snowflake, no DB sequence
- Typical latency: 10-30ms

---

**3. User's URL History**
```sql
SELECT id, short_code, long_url, created_at, expires_at
FROM url_mappings
WHERE user_id = ?
  AND is_active = TRUE
ORDER BY created_at DESC
LIMIT 50 OFFSET ?;
```

**Performance:**
- Index on `(user_id, created_at)`
- Pagination with LIMIT/OFFSET
- Typical latency: 50-100ms

---

**4. Analytics Query**
```sql
SELECT
  DATE(timestamp) as date,
  HOUR(timestamp) as hour,
  COUNT(*) as clicks,
  COUNT(DISTINCT ip_address) as unique_ips
FROM url_clicks
WHERE short_code = ?
  AND timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY date, hour
ORDER BY date DESC, hour DESC;
```

**Performance:**
- Use aggregated stats table (url_stats) for reporting
- Raw url_clicks only for recent data (last 7 days)
- Typical latency: 100-500ms

---

**5. Cleanup Expired URLs**
```sql
DELETE FROM url_mappings
WHERE expires_at IS NOT NULL
  AND expires_at < NOW()
  AND created_at < DATE_SUB(NOW(), INTERVAL 30 DAY);
```

**Performance:**
- Run as background job (nightly)
- Batch delete to avoid locking
- Can be soft-delete instead (set `is_active = FALSE`)

---

## Indexing Strategy

### Index Summary

```
url_mappings:
├─ UNIQUE (short_code)          -- Primary lookup
├─ (user_id, created_at)        -- User history
├─ (expires_at)                 -- Cleanup queries
└─ (created_at)                 -- Time-based queries

url_clicks:
├─ (short_code, timestamp)      -- Analytics
├─ (timestamp)                  -- Time-range
├─ (user_id)                    -- User analytics
└─ (country_code)               -- Geographic

users:
├─ UNIQUE (email)               -- Login
├─ UNIQUE (username)            -- Display lookup
└─ (created_at)                 -- Sorting
```

### Index Size Estimation

For 1 billion URLs:
- Primary table: ~2.1 TB
- Indexes (all): ~500 GB
- **Total: ~2.6 TB**

For 1 year of analytics (365B clicks):
- Raw clicks table: ~150 TB (partitioned)
- Aggregates table: ~10 GB
- **Total: ~160 TB** (mostly archived)

---

## Denormalization & Caching

### Denormalized Fields

```sql
-- Add to url_mappings for faster reads
ALTER TABLE url_mappings ADD COLUMN (
  click_count INT DEFAULT 0,
  last_clicked_at TIMESTAMP,
  unique_visitors_count INT DEFAULT 0
);
```

**Benefits:**
- Avoid joining with stats table
- Cache these values (update periodically)
- Trade storage for query speed

**Update strategy:**
- Cache in Redis
- Sync to database every 10 minutes
- Accept stale data (eventual consistency)

---

## Backup & Disaster Recovery

### Backup Strategy

**Frequency:**
- Incremental backups: Every hour
- Full backups: Daily
- Retention: 30 days

**Backup locations:**
- Primary: Local SSD
- Secondary: Different availability zone
- Tertiary: Cloud storage (S3/GCS)

**Recovery time:**
- RTO (Recovery Time Objective): 1 hour
- RPO (Recovery Point Objective): 5 minutes

### Replication

```
Primary DB (writes)
  ↓ (binary log replication)
Replica 1 (async, <100ms lag)
Replica 2 (async, <100ms lag)
```

**Multi-region:**
- Cross-region replica for disaster recovery
- Higher replication lag (acceptable)
- Can promote to primary if needed

---

## Monitoring & Maintenance

### Queries to Monitor

```sql
-- Check replication lag
SHOW SLAVE STATUS\G

-- Check slow queries
SELECT * FROM slow_query_log ORDER BY query_time DESC;

-- Check table fragmentation
SELECT table_name, data_free, data_length
FROM information_schema.TABLES
WHERE table_schema = 'db_name';

-- Check index usage
SELECT object_schema, object_name, count_read, count_write
FROM performance_schema.table_io_waits_summary_by_index_usage
ORDER BY count_read DESC;
```

### Maintenance Tasks

1. **Weekly:** `OPTIMIZE TABLE` on large tables
2. **Monthly:** Rebuild fragmented indexes
3. **Quarterly:** Archive old data
4. **Ongoing:** Monitor slow query log, add indexes as needed
