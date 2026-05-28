# System Design: Ad Recommendation System for Social Media

## Problem Statement

You are designing an **ad recommendation system** for a social media platform (similar to Instagram, Facebook, TikTok).

**Context:**
- Users scroll through a feed of posts (photos, videos, text)
- The platform monetizes through ads displayed between posts
- Currently, ads are shown randomly or based on basic targeting (location, age, demographics)
- You're tasked to build a personalized ad recommendation system that:
  - **Increases ad engagement** (clicks, views, conversions)
  - **Improves user experience** (show relevant ads, not intrusive spam)
  - **Scales to millions of users and billions of impressions per day**

**The Challenge:**
- Users are in a **passive browsing state** — they're not searching or explicitly requesting ads
- The primary content they consume is **posts**, not ads
- You need to **infer their interests** from the posts they've scrolled through
- Real-time latency is critical (ads must be served in < 200ms)
- The system must handle continuous updates as new posts/ads arrive

---

## Your Task

Design the **end-to-end ML system** for ad recommendations.

### What You Should Cover:

1. **Clarifying Questions**
   - Before jumping into design, ask questions to understand requirements and constraints

2. **Goals and Constraints**
   - Define clear business metrics and success criteria
   - Identify latency, throughput, and cost constraints

3. **System Architecture**
   - High-level design: components and data flow
   - How do you ingest user behavior (posts viewed)?
   - How do you generate candidate ads?
   - How do you rank them?
   - How do you serve predictions at scale?

4. **ML Components**
   - Feature engineering: What signals matter?
   - Model architecture: How do you capture user interests and ad relevance?
   - Candidate generation: Efficient retrieval from millions of ads
   - Ranking: How to score and order ads
   - Online learning: How to adapt in real-time

5. **Serving and Deployment**
   - How do you serve predictions in < 200ms?
   - Batch vs. online inference?
   - Model versioning, A/B testing, deployment safety

6. **Evaluation**
   - Offline metrics: What can you measure on historical data?
   - Online metrics: What to A/B test in production?
   - How to measure quality (CTR, conversion, revenue) vs. user satisfaction (relevance, diversity)

7. **Tradeoffs**
   - Proactively discuss design tradeoffs:
     - **Personalization vs. Privacy**: How much user data do you collect?
     - **Accuracy vs. Latency**: Do you need real-time features or can you use batch?
     - **Cold-start problem**: How to recommend ads for new users/ads?
     - **Diversity**: Should you recommend the same ad repeatedly or diversify?
     - **Freshness**: How often do you update user profiles and ad rankings?

---

## Clarifying Questions to Ask (As a Candidate)

Before designing, you should ask:

1. **Scale**: How many users, ads, and daily impressions?
2. **Types of ads**: Text ads, image ads, video ads, carousel ads?
3. **Ad inventory**: How frequently do new ads get added? How long does an ad stay active?
4. **User behavior signals**: Do we track clicks, conversions, dwells, or just impressions?
5. **Latency requirements**: How fast must ads be served? Can we batch?
6. **Cold-start**: How do we handle new users with no history?
7. **Constraints**: Privacy regulations (GDPR, CCPA)? Cost/compute limits?
8. **Success metrics**: Are we optimizing for CTR, revenue, user satisfaction, or a combination?

---

## Expected Answer Structure

A strong answer should:

1. ✓ Ask clarifying questions before diving into design
2. ✓ Define success metrics and non-functional requirements explicitly
3. ✓ Propose a scalable architecture with clear data flow
4. ✓ Discuss feature engineering and model choices
5. ✓ Address serving and latency challenges
6. ✓ Explain how to evaluate and iterate
7. ✓ Proactively discuss tradeoffs and defend choices
8. ✓ Think end-to-end (from user scroll → ad impression → feedback loop)

---

## Common Follow-up Questions

Expect questions like:

- **"How do you handle the cold-start problem for new ads?"**
- **"What if an advertiser wants high precision on a narrow audience?"**
- **"How do you prevent showing the same ad repeatedly?"**
- **"How do you ensure diversity vs. pure personalization?"**
- **"How does this differ from YouTube's recommendation system?"**
- **"What happens when a user blocks an advertiser?"**
- **"How do you handle budget constraints for ads (daily budget limits)?"**
- **"How do you measure and prevent ad fraud?"**
- **"How do you balance CTR vs. revenue per ad impression?"**

---

## Hints for a Strong Answer

- **Start with users:** Understand user intent from posts they've scrolled
- **Two-tower architecture:** Separate embeddings for users and ads; score via similarity
- **Retrieval + Ranking:** Use approximate nearest neighbor search for candidate generation, then precise ranker
- **Feedback loops:** Clicks, conversions, dwell time feed back into the system
- **A/B testing:** Use experiments to validate design choices
- **Latency matters:** Use caching, approximate methods, and parallelization
- **Privacy-aware:** Consider user privacy in feature collection
