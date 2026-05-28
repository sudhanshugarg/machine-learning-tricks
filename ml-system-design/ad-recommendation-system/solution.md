# Solution: Ad Recommendation System for Social Media

## Step 1: Clarifying Questions & Requirements

### Questions to Ask the Interviewer

Before proposing a design, ask these questions:

#### Scale & Volume
- **Users:** Millions? Billions? (Affects architecture complexity)
- **Ads:** How many active ads in the system? (Millions? Tens of millions?)
- **Daily impressions:** Expected ad impressions per day?
- **Growth trajectory:** Expecting rapid scaling?

#### User Behavior & Signals
- **Post consumption:** What data do we have? (impressions, dwell time, clicks, saves, shares)
- **Ad interactions:** What counts as positive signal? (clicks, conversions, dwells, time spent)
- **Feedback latency:** Do we get immediate feedback or delayed (hours/days)?

#### Ad Inventory
- **Ad lifecycle:** How long does an ad stay active? (days? months?)
- **New ads:** How frequently are new ads added? (hourly? continuously?)
- **Ad formats:** Just images/text or complex formats (video, carousel)?
- **Ad metadata:** Do we have categorical tags, descriptions, advertiser info?

#### Technical Constraints
- **Latency:** Hard requirement for serving? (e.g., < 200ms)
- **Compute budget:** Cost or compute constraints?
- **Privacy:** GDPR/CCPA compliant? Can we use third-party data?
- **Offline vs. online:** Can we batch or must it be real-time?

#### Business Metrics
- **Primary goal:** CTR, conversion rate, revenue, user satisfaction?
- **Secondary goals:** Diversity, freshness, reducing ad fatigue?
- **Trade-offs:** Revenue vs. user experience?

#### Special Constraints
- **Cold-start:** How do we handle new users and new ads?
- **Budget caps:** Do ads have daily budgets we must respect?
- **Ad quality:** Do we filter or blacklist certain ads?

---

## Step 2: Goals and Constraints

### Assumptions (Based on Typical Social Media Platform)

**Scale:**
- 500M monthly active users
- 1M active advertisers
- 10M ads in system
- 5B ad impressions per day
- ~6K impressions per second

**Success Metrics:**
- **Click-through rate (CTR):** Increase from 0.5% baseline to 1%+
- **Conversion rate:** Increase advertiser conversions by 30%+
- **Revenue:** Maximize advertiser spend and platform revenue
- **User satisfaction:** Keep ratio of satisfied clicks to total impressions high (> 70%)
- **Ad quality:** Avoid showing irrelevant ads (relevance score > 0.6)

**Non-Functional Requirements:**
- **Latency:** Serve ad recommendations in < 200ms (p99)
- **Throughput:** Handle 6K ad requests per second
- **Availability:** 99.99% uptime
- **Freshness:** User profiles update within 1 hour; ad rankings update within 1 day
- **Cost:** Minimize compute while maintaining accuracy

**Business Constraints:**
- Must comply with GDPR/CCPA (user privacy)
- Cannot show inappropriate ads to children
- Must respect advertiser budgets (daily caps)
- Must avoid showing same ad too frequently (ad fatigue)

---

## Step 3: System Architecture (High-Level)

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER ACTIONS                               │
│  (User scrolls through feed, views posts, clicks ads)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              DATA COLLECTION & STREAMING                        │
│  - User impressions (posts viewed)                              │
│  - Ad interactions (clicks, conversions)                        │
│  - Real-time event logs                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Offline │    │  Online  │    │ Feedback │
    │ Feature  │    │ Feature  │    │   Loop   │
    │ Store    │    │ Store    │    │          │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   USER PROFILE & EMBEDDINGS        │
        │  (User interests from post history)│
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   CANDIDATE GENERATION             │
        │  (Retrieve ~1K relevant ads via    │
        │   nearest neighbor search)         │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   RANKING MODEL                    │
        │  (Score & rank ~1K candidates)     │
        │  (Estimate CTR, conversion, etc.)  │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   POST-PROCESSING                  │
        │  - Apply business rules            │
        │  - Enforce budget caps             │
        │  - Diversify ads                   │
        │  - Filter blocked ads              │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   SERVING (< 200ms)                │
        │  Return top-K ads to user          │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   FEEDBACK LOGGING                 │
        │  - Ad impression logged            │
        │  - Click logged                    │
        │  - Conversion logged               │
        └────────────────────────────────────┘
```

### Key Components

1. **Feature Pipeline**
   - Offline: Batch compute user profiles daily
   - Online: Fetch user's recent post history, real-time interests

2. **Candidate Generation**
   - Embedding-based retrieval using approximate nearest neighbors (ANN)
   - Find ads similar to user's recent post interests

3. **Ranking Model**
   - Deep learning model (e.g., two-tower architecture)
   - Predict CTR, conversion probability
   - Score and rank candidates

4. **Post-Processing**
   - Business rules (budget caps, adult content filtering)
   - Diversity enforcement (don't repeat ads)
   - Diversity sampling (show variety)

5. **Serving**
   - Caching layer (cache hot user profiles)
   - In-memory ad embeddings
   - Fast inference (< 200ms)

6. **Feedback Loop**
   - Collect clicks, conversions, dwell time
   - Use for model training and online learning
   - A/B test new models

---

## Step 4: Detailed Design

### 4A. Feature Engineering

#### User Features (From Post History)

**Explicit post topics:**
- Posts viewed: Categories (Sports, Food, Fashion, Tech, News, etc.)
- Post embeddings: Dense representation of post content
- Hashtags: Topics user engages with

**User engagement patterns:**
- Time spent per post (dwell time)
- Posts saved/shared (higher intent signal)
- Comments/likes on posts
- Search queries (if available)

**Demographic/context:**
- User location (approximate)
- Device type (mobile vs. desktop)
- Time of day
- Day of week

**User history:**
- Ads clicked in past (similarity to current ads)
- Advertiser categories user has engaged with
- Ad fatigue signal: Time since last ad from same advertiser

#### Ad Features

**Static features:**
- Ad embeddings: Dense representation of ad content
- Advertiser category (Fashion, Tech, Finance, etc.)
- Ad format (image, text, video)
- Advertiser ID
- Ad age (freshness)
- Ad's historical CTR

**Dynamic features:**
- Advertiser budget remaining today
- Ad's remaining daily budget
- Competitive demand for this ad (popularity)
- Quality score (historical relevance)

#### Interaction Features

- **Similarity:** Cosine similarity between user interest embedding and ad embedding
- **Recency decay:** Discount very old posts/ads
- **Category match:** Does ad category match user's interests?

### 4B. Model Architecture

#### Two-Tower Architecture

**Tower 1: User Embedding**
```
Posts Viewed → LSTM/Transformer → User Embedding (d=128)
(weighted by dwell time)
```

**Tower 2: Ad Embedding**
```
Ad Content → CNN/Transformer → Ad Embedding (d=128)
(image/text encoding)
```

**Scoring:**
```
Score = user_embedding · ad_embedding + neural_ranker(user, ad)
```

**Why two towers?**
- User tower can be precomputed offline (efficient)
- Ad tower can be precomputed for each ad (offline)
- Score is computed in milliseconds at serving time
- Easily scales to millions of ads

#### Ranking Model (Neural Network)

```
Input Features:
- user_embedding (d=128)
- ad_embedding (d=128)
- user_category_interest (sparse)
- ad_category (sparse)
- user_location_ad_location_match (1D)
- dwell_time_last_post (1D)
- advertiser_quality_score (1D)
- ad_age_days (1D)
- user_ad_category_history (sparse)

Hidden Layers:
- Embedding layer (categorical features)
- Dense layers: 512 → 256 → 128 → 64
- ReLU activation, Batch norm, Dropout (0.2)

Output:
- CTR prediction: sigmoid → probability of click
- Conversion prediction: sigmoid → probability of purchase
```

**Loss Function:**
```
Loss = λ₁ * cross_entropy(CTR) + λ₂ * cross_entropy(Conversion)
     + λ₃ * L2_regularization
```

### 4C. Candidate Generation (Retrieval)

**Challenge:** We have 10M ads, but only ~200ms to serve. Can't score all ads.

**Solution: Two-stage approach**

**Stage 1: Offline (Batch)**
- Every 24 hours:
  - Compute user interest embedding from last 100 posts they viewed
  - Compute ad embeddings for all 10M ads
  - Build HNSW (Hierarchical Navigable Small World) index on ad embeddings

**Stage 2: Online (At request time)**
- Retrieve top-1000 ads using ANN search (< 50ms)
  - Query: user embedding
  - Return: top 1000 most similar ad embeddings
- This 1000 becomes input to ranking model

**Why this approach?**
- ANN retrieval is < 50ms for 10M items
- Ranking model only processes 1000 ads (< 100ms)
- Total: < 150ms, within 200ms budget
- Offline computation amortized

### 4D. Ranking Model

**Input:** 1000 candidate ads + user features

**Process:**
1. Vectorize features for each candidate (1000 × feature_dim)
2. Pass through ranking neural network
3. Get CTR + Conversion predictions for each ad
4. Compute relevance score: `score = CTR * value_per_click + Conversion * value_per_conversion`

**Constraints applied:**
- Filter out ads from blocked advertisers
- Remove ads the user already clicked
- Enforce advertiser daily budgets

**Output:** Ranked list of ads (top 10-20)

### 4E. Online Learning & Feedback Loop

**Real-time Feedback:**
```
User interaction (click, conversion, dwell)
    → Event logged to Kafka
    → Stream processor
    → Update online feature store
    → Update model serving cache
```

**Batch Updates (Daily):**
```
1. Collect last 24 hours of feedback
2. Retrain ranking model with new data
3. Evaluate on holdout set
4. If better: deploy new model
5. Update user embeddings
6. Rebuild ANN index
```

**Bandit Algorithms (Optional):**
- Use Thompson Sampling or ε-greedy to balance exploration vs. exploitation
- Explore ads with high uncertainty to learn faster

---

## Step 5: Serving & Deployment

### Serving Architecture

```
┌─────────────┐
│   User      │
│   Request   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  API Gateway (Rate limiting)        │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Feature Service                    │
│  - Fetch user embedding (cached)    │
│  - Fetch context features           │
│  Latency: ~20ms (cache hit)         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  ANN Search Service                 │
│  - Query HNSW index                 │
│  - Return top 1K ads                │
│  Latency: ~50ms                     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Ranking Service                    │
│  - Score 1K ads                     │
│  - Apply business rules             │
│  - Return top 20 ads                │
│  Latency: ~80ms                     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Response Assembly                  │
│  - Format ads for client            │
│  - Add tracking pixels              │
│  Latency: ~10ms                     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Response   │ (Total: ~160ms)
└─────────────┘
```

### Caching Strategy

**Cache Layer 1: User Profile Cache (Hot)**
- LRU cache of user embeddings for top 10% most active users
- Eviction policy: LRU with TTL = 1 hour
- Reduces feature fetch latency to 5ms

**Cache Layer 2: Ad Embedding Cache**
- All 10M ad embeddings in memory
- Read-only, updated daily
- Enables fast ANN search

**Cache Layer 3: Query Result Cache**
- Cache ranking model results for (user, context) pairs
- TTL = 5 minutes
- Reduces repeated requests overhead

### Model Serving Options

#### Option A: Online Inference (Real-time Serving)
```
Request → Model Server (TensorFlow Serving / TorchServe)
       → Load model from disk/cache
       → Inference
       → Return predictions
```
- Pros: Fresh predictions, handles personalization
- Cons: Higher latency, requires robust model servers

#### Option B: Pre-computed Serving (Batch)
```
Daily batch job:
- For each user: score all 10M ads
- Store top-1K for each user
- At request time: retrieve pre-scored ads
```
- Pros: Ultra-low latency (just lookup)
- Cons: Can't handle personalization beyond stored scores, stale

#### Option C: Hybrid (Recommended)
```
- Pre-compute base scores for all (user, ad) pairs
- At request time: apply real-time features (context, budget)
- Minimal online inference
```

### Deployment & Safety

**Canary Deployment:**
```
1. Train new model
2. Evaluate on holdout set
3. Deploy to 1% of traffic
4. Monitor CTR, conversion, latency
5. If metrics improve: gradually increase to 100%
```

**A/B Testing:**
```
Test A (Control): Current ranking model
Test B (Treatment): New ranking model

Run for 7 days:
- Measure CTR, Conversion, Revenue
- Check user satisfaction metrics
- If B > A: deploy, else rollback
```

**Monitoring & Alerts:**
```
Monitor:
- p50, p95, p99 latency
- Model inference time
- ANN retrieval time
- Feature fetch time
- Error rates
- CTR/Conversion rates
- Budget overage

Alert on:
- Latency p99 > 300ms
- Error rate > 0.1%
- CTR drop > 5%
- Revenue drop > 5%
```

---

## Step 6: Evaluation Metrics

### Offline Metrics (Batch Evaluation)

**Ranking Quality:**
- **NDCG (Normalized Discounted Cumulative Gain):** How well are good ads ranked at top?
- **MAP (Mean Average Precision):** Is relevant ad in top-K?
- **AUC (Area Under ROC Curve):** CTR prediction quality

**Model Performance:**
- **CTR Prediction Accuracy:** Calibration of predicted CTR
- **Conversion Prediction Accuracy:** Calibration of predicted conversion

**Retrieval Quality:**
- **Recall@1K:** Of truly relevant ads, what % do we retrieve?
- **Diversity of retrieved ads:** How different are candidates?

### Online Metrics (A/B Tests)

**User Engagement:**
- **Click-through rate (CTR):** % of impressions that result in clicks
- **Conversion rate:** % of clicks that result in purchase
- **Average revenue per user (ARPU):** Platform revenue per user

**User Satisfaction:**
- **Relevance score:** User-rated relevance (1-5 stars)
- **Ad fatigue:** How often same ad shown to user?
- **Click feedback:** Positive/negative feedback on ad

**Business Metrics:**
- **Advertiser ROI:** Return on ad spend
- **Platform revenue:** Total ad revenue
- **Cost per acquisition (CPA):** Cost to acquire one customer

**Guardrail Metrics (Don't decrease):**
- **User retention:** % of users returning next day/month
- **Time on app:** Total time spent by users
- **Churn rate:** % of users leaving platform

### Metrics Trade-offs

**CTR vs. Relevance:**
- Pure CTR optimization → may show cheap/low-quality ads repeatedly
- Solution: Combine CTR with quality score

**Personalization vs. Diversity:**
- Pure personalization → same ad repeatedly (ad fatigue)
- Solution: Add diversity constraint: "Show at most 1 ad per advertiser per day"

**Revenue vs. User Experience:**
- Pure revenue → show high-value ads regardless of relevance
- Solution: Weight by relevance: `revenue * relevance_score`

---

## Step 7: Addressing Cold-Start Problems

### New User Cold-Start

**Problem:** New user has no post history, no embedding

**Solutions:**
1. **Demographic-based targeting:** Use age, location to infer interests
2. **Contextual targeting:** Use app context (which section of app user in?)
3. **Popular ads:** Show trending/popular ads until profile builds
4. **Exploration:** Explore broad categories to learn interests
5. **Social signals:** If user has friends on platform, use their interests

### New Ad Cold-Start

**Problem:** New ad has no click history, no quality score

**Solutions:**
1. **Advertiser history:** Use advertiser's historical CTR
2. **Ad similarity:** Find similar existing ads, use their performance
3. **Initial exploration:** Start with high exploration/bandit algorithms
4. **Advertiser information:** Use ad metadata (category, description) for targeting
5. **Gradual ramp:** Ramp up impressions as quality score builds

### New Advertiser Cold-Start

**Problem:** Advertiser has no ads in the system yet

**Solutions:**
1. **Advertiser metadata:** Use industry classification, target audience
2. **Similar advertisers:** Find similar companies, use their performance
3. **Budget-aware:** Give them more opportunities since they're new
4. **Quality review:** Manual review to ensure ad quality

---

## Step 8: Privacy & Safety Considerations

### User Privacy

**Data Collection:**
- Collect minimally necessary data (e.g., posts viewed, not detailed browsing)
- Hash/anonymize user IDs
- Differential privacy on feature aggregation

**Regulations Compliance:**
- GDPR: User right to access/delete data
- CCPA: User right to opt-out
- Implement data deletion pipeline

**Feature Engineering:**
- Don't use sensitive attributes (race, religion, health status)
- Avoid building profiles based on sensitive behavior

### Ad Quality & Safety

**Filtering:**
- Block ads from certain categories (weapons, drugs, adult content)
- Block ads from blacklisted advertisers
- Quality review before ads go live

**Ad Fraud Detection:**
- Monitor for click fraud (bot clicks)
- Monitor for invalid traffic (IVT)
- Flag suspicious patterns

**Brand Safety:**
- Track complaints about ads
- Remove ads with high complaint rates
- Respect advertiser brand safety preferences

---

## Step 9: Tradeoffs & Design Decisions

### Tradeoff 1: Accuracy vs. Latency

**Decision:** Two-stage retrieval + ranking

**Tradeoff:**
- Can't score all 10M ads (too slow)
- But can find good candidates with ANN (fast)
- Trade some accuracy for 200ms latency requirement

**Alternative:** Spend more on compute → process more candidates, higher latency

### Tradeoff 2: Personalization vs. Privacy

**Decision:** Use only post viewing data, no external tracking

**Tradeoff:**
- Better accuracy if we use external data (browsing history, location)
- But violates user privacy and GDPR
- Choose privacy-first approach, trade some accuracy

**Alternative:** Request explicit user consent for broader tracking

### Tradeoff 3: CTR Optimization vs. Relevance

**Decision:** Weight by relevance score, not pure CTR

**Tradeoff:**
- Pure CTR → show low-quality cheap ads repeatedly
- Weighted → balanced approach
- Slightly lower CTR but higher user satisfaction

**Alternative:** Pure CTR maximization if revenue trumps UX

### Tradeoff 4: Freshness vs. Efficiency

**Decision:** Update daily (batch), not real-time

**Tradeoff:**
- Daily updates → user profiles 1 day stale
- Real-time → more compute, more complex
- Daily is sweet spot for efficiency vs. freshness

**Alternative:** More frequent updates if compute budget allows

### Tradeoff 5: Exploration vs. Exploitation

**Decision:** Use ε-greedy: mostly exploit best ads, sometimes explore

**Tradeoff:**
- Pure exploitation → miss good new ads
- Pure exploration → waste impressions on bad ads
- ε-greedy finds balance (e.g., ε=0.1 → 10% exploration)

**Alternative:** Thompson Sampling for more sophisticated exploration

### Tradeoff 6: Centralized vs. Distributed Serving

**Decision:** Distributed with caching (edge, regional)

**Tradeoff:**
- Centralized → simple, single source of truth
- Distributed → low latency for users worldwide, complexity
- Distributed necessary for 6K req/sec

---

## Step 10: Common Follow-up Questions

### Q: How do you handle budget constraints?

**A:** Track advertiser daily budgets in database:
```python
# Before ranking
for ad in candidates:
    remaining_budget = get_advertiser_budget(ad.advertiser_id)
    if remaining_budget < min_bid_price:
        candidates.remove(ad)

# After serving
# Allocate budget proportionally to ads shown
for ad in served_ads:
    deduct_from_budget(ad.advertiser_id, cost)
```

### Q: How do you prevent showing the same ad repeatedly?

**A:** Track ad impressions per user + time:
```python
# Feature: days_since_last_impression = days since user saw this ad
# In ranking model: low score if days_since_last_impression < threshold

# Hard constraint in post-processing:
user_ads_last_7_days = get_user_ads_history(user_id, days=7)
for ad in candidates:
    if ad.id in user_ads_last_7_days:
        if ad.frequency >= max_frequency_per_week:
            candidates.remove(ad)
```

### Q: What about advertisers with limited budgets?

**A:** Allocate budgets more carefully:
```
# Priority-based allocation
# High-budget advertisers: broad targeting
# Low-budget advertisers: narrow targeting (specific user segments)

# Auction mechanism (optional):
# If demand > supply, use second-price auction
# Allocate impressions to highest-bidding ads
```

### Q: How do you evaluate a new model safely?

**A:** Staged rollout with monitoring:
```
1. Offline evaluation: Compare offline metrics
2. Canary test: 1% traffic for 1 day
3. If good: 10% traffic for 3 days
4. If still good: 50% for 7 days
5. If metrics hold: 100% rollout

Monitor: CTR, Conversion, Latency, Revenue, User Satisfaction
```

### Q: What about adversarial advertisers trying to game the system?

**A:** Multiple layers:
1. **Quality scoring:** Penalize ads with high negative feedback
2. **Click fraud detection:** Detect bot clicks, flag suspicious patterns
3. **Account monitoring:** Flag accounts with unusual spend/behavior
4. **Manual review:** Human review of flagged accounts
5. **Rate limiting:** Throttle accounts showing suspicious behavior

### Q: How do you measure relevance (not just CTR)?

**A:** Multi-faceted approach:
```
1. Explicit user feedback: Ask "Was this ad relevant?" (1-5 stars)
2. Implicit signals:
   - Click = relevant
   - Dwell on ad > 3 seconds = relevant
   - Share ad = highly relevant
   - Report ad as inappropriate = not relevant
3. Survey-based evaluation: Periodic user surveys
4. Conversion data: If ad led to purchase, it was relevant
```

### Q: How do you handle real-time events (trending topics)?

**A:** Event-driven updates:
```
1. Detect trending topics in real-time
2. Update user interests based on engagement with trending posts
3. Update ad ranking to favor ads matching trending topics
4. Special handling: e.g., show COVID-related ads during pandemic surge

# Event listener:
on_trending_topic(topic) {
    affected_users = get_users_interested_in(topic)
    for user in affected_users:
        update_user_interests(user, topic)
}
```

### Q: What if a new advertiser wants high-precision targeting?

**A:** Segment-based approach:
```
# Advertiser specifies: Target only users interested in "Fashion"
# Implementation:
targeting_criteria = {
    'interests': ['Fashion'],
    'age_range': [18, 35],
    'location': ['NYC', 'LA']
}

# In ranking, filter candidates:
for ad in candidates:
    if not matches_targeting_criteria(user, ad.targeting):
        continue
    # Then score normally
```

---

## Summary: Key Takeaways

1. **Start with questions:** Understand requirements, constraints, scale
2. **Define metrics explicitly:** Business metrics drive all decisions
3. **Two-stage architecture:** Retrieval (ANN) + Ranking (neural net)
4. **Offline + Online:** Precompute what you can, infer what you must
5. **Latency is critical:** 200ms budget requires efficient design
6. **Safety first:** Privacy, quality, fraud prevention must be baked in
7. **Feedback loops:** Clicks/conversions drive continuous improvement
8. **A/B testing everything:** Validate with real user data, not just offline
9. **Monitor and iterate:** Watch metrics, alert on regressions, update models
10. **Discuss tradeoffs:** Show you understand why you made each choice

