# Recommendation System Design

## Problem Statement

Design a recommendation system for an e-commerce platform with millions of users and products. The system should recommend relevant products to users in real-time.

## System Requirements

### Functional Requirements
1. Recommend top-N products for a given user
2. Handle new users with limited history (cold start problem)
3. Provide real-time recommendations with <100ms latency
4. Support batch recommendations for multiple users
5. Log interactions for model retraining

### Non-Functional Requirements
1. Scale to 10M users and 1M products
2. Availability: 99.9% uptime
3. Latency: <100ms for single user recommendations
4. Handle peak traffic (100x normal)

## High-Level Approach

### 1. Recommendation Strategy (Hybrid)
- **Content-based**: User features + product features
- **Collaborative Filtering**: User-user or item-item similarity
- **Matrix Factorization**: Learn latent factors for users/products
- **Combine**: Use ensemble to blend signals

### 2. Architecture Components

```
User Request
    ↓
[Load Balancer]
    ↓
[API Gateway]
    ↓
[Recommendation Service]
    ├── Cache Layer (Redis) - Hot recommendations
    ├── Feature Store - User/product features
    ├── ML Model Service - Score candidates
    └── Ranking Service - Re-rank candidates
    ↓
[Return Top-N]
```

### 3. Key Data Flows

**Online (Real-time Prediction):**
- Fetch user profile & recent interactions
- Retrieve candidate pool from index
- Score candidates with model
- Re-rank based on business rules
- Cache results

**Offline (Model Training):**
- Collect user-item interactions
- Extract features
- Train collaborative filtering model
- Update embeddings in feature store
- Deploy new model version

## Detailed Design

### Candidate Generation
1. Retrieve similar users (collaborative filtering)
2. Get popular products in user segments
3. Personalized ranking based on user profile
4. De-duplicate and limit to top-1000

### Scoring
1. Matrix factorization score
2. Content similarity score
3. Popularity score
4. Diversity penalty
5. Final score = weighted combination

### Ranking
1. Apply business rules (inventory, margins)
2. Diversity constraints (avoid duplicates)
3. Explore-exploit tradeoff (15% new items)
4. Position bias correction
5. Return top-N

## Data Models

**User Entity**
- user_id, demographics, segments
- user_embedding (latent factors)
- historical_preferences

**Product Entity**
- product_id, category, tags
- product_embedding (latent factors)
- metadata (price, rating, inventory)

**Interaction**
- user_id, product_id, timestamp
- interaction_type (view, click, purchase)
- duration, position

## Scalability Considerations

1. **Caching**: Cache top recommendations for popular users (80/20 rule)
2. **Batch Processing**: Pre-compute for user cohorts
3. **Distributed ML**: Parallel model training and serving
4. **Indexing**: Use FAISS/Annoy for fast similarity search
5. **Sharding**: Partition by user_id for horizontal scaling

## Cold Start Problem

- **New Users**: Show popular products + use demographic similarity
- **New Products**: Show to users interested in similar categories
- **Exploration**: Multi-armed bandit for new items
