# FAQ — Semantic Search & Retrieval over Autonomous Driving Video Data

This is a **living FAQ** for [design.md](design.md) / [solution.md](solution.md). Any question raised while working through this problem — a term definition, "why this over that," a deeper dive on a component — gets **flagged and recorded here** so it can feed back into the solution.

> **How this file is maintained (convention):**
> Whenever a question is asked, it is:
> 1. **Flagged** with a status tag — `[NEW]` (captured, not yet answered), `[ANSWERED]` (answer written below), or `[TODO-SOLUTION]` (reveals a gap that should be folded into `solution.md` or `design.md`).
> 2. **Categorized** (Terminology / Architecture / Tradeoffs / Math / Follow-up).
> 3. **Logged** in the Question Log below, dated, with a pointer to where it belongs.
>
> When an answer here exposes a gap in `solution.md`, update that file **and** flip the tag to `[ANSWERED]` with a pointer.

---

## Question Log (chronological)

| # | Date | Question | Status | Category | Belongs in |
|---|------|----------|--------|----------|-----------|
| 1 | 2026-07-13 | What is a "CLIP embedding" and exactly how is such a model trained — contrastive loss, training rows, labels? | `[ANSWERED]` | Architecture / Math | [solution.md](solution.md) Step 4A |
| 2 | 2026-07-13 | What's the difference between video embeddings and image embeddings, and how exactly is a video embedding calculated? | `[ANSWERED]` | Terminology / Architecture | [solution.md](solution.md) Step 4A |
| 3 | 2026-07-13 | How exactly does the re-ranking cross-attention model work? What is cross-attention? | `[ANSWERED]` | Architecture / Math | [solution.md](solution.md) Step 4B |
| 4 | 2026-07-13 | What is a ViT (Vision Transformer) encoder vs. a CNN encoder? How do they differ and which should I use? | `[ANSWERED]` | Terminology / Architecture | [solution.md](solution.md) Step 4A |
| 5 | 2026-07-13 | What are "perception stack detections"? What information do they contain, and what do some concrete examples look like? | `[ANSWERED]` | Terminology | [solution.md](solution.md) Step 4A |
| 6 | 2026-07-13 | In batch-based contrastive learning, how do we prevent true positives from being treated as negatives? What if two captions match the same video? | `[ANSWERED]` | Math / Architecture | [solution.md](solution.md) Step 4A |
| 7 | 2026-07-13 | What exactly are "image tokens" or "video tokens"? I understand text tokens, but image/video tokens are confusing — aren't images just RGB pixels? | `[ANSWERED]` | Terminology | [solution.md](solution.md) Step 4A |

*(Append new rows as questions come in.)*

---

## Terminology

### Q: What are "perception stack detections"? What information do they contain, and what do some concrete examples look like? `[ANSWERED]`

**A:**

The **perception stack** is the AV's real-time object-detection and tracking system — it runs continuously on raw sensor input (camera, LiDAR, radar) and outputs what it detected: where agents (pedestrians, cyclists, vehicles, etc.) are, what they are, and some metadata about each detection.

**What a single "detection" contains** — typically a structured record with:

```python
detection = {
    'timestamp': 1234567890.123,           # when was this detected
    'agent_id': 'track_42',                # persistent ID for this agent across frames
    'agent_class': 'pedestrian',           # what is it (pedestrian, cyclist, car, truck, etc.)
    'bbox_3d': {                           # 3D bounding box in world coordinates
        'center': [x, y, z],               # center position (meters, relative to vehicle)
        'size': [length, width, height],   # physical dimensions
        'orientation': yaw_angle           # heading/facing direction
    },
    'confidence': 0.87,                    # how sure is the model (0.0 to 1.0)
    'velocity': [vx, vy, vz],              # estimated velocity (m/s)
    'attributes': {
        'occluded': False,                 # partially hidden from view?
        'truncated': False,                # cut off at image edge?
        'num_frames_tracked': 12,          # how many frames has this ID existed?
    }
}
```

**Why "stack"?** The full pipeline is a stack of stages (sensor fusion, 2D detection, tracking, classification) that all output intermediate results. When we say "perception-stack detections," we're referring to the final, consolidated detection list that comes out of the tracker (so a single agent is represented as one track, not multiple detections per frame).

**Concrete examples from driving scenarios:**

**Example 1: Pedestrian at a crosswalk**
```python
{
    'timestamp': 1234567890.500,
    'agent_id': 'ped_18',
    'agent_class': 'pedestrian',
    'bbox_3d': {
        'center': [15.2, -2.1, 0.0],      # 15m ahead, 2m to the left, ground level
        'size': [0.5, 0.4, 1.7],          # person-sized
        'orientation': 1.57                # facing perpendicular to vehicle direction
    },
    'confidence': 0.92,
    'velocity': [0.3, -0.8, 0.0],         # moving left and slightly toward vehicle
    'attributes': {
        'occluded': False,
        'truncated': False,
        'num_frames_tracked': 45,          # been visible for 45 frames = ~1.5 seconds at 30fps
    }
}
```

**Example 2: Partially-occluded cyclist**
```python
{
    'timestamp': 1234567890.500,
    'agent_id': 'cyclist_7',
    'agent_class': 'cyclist',             # rider + bike detected together
    'bbox_3d': {
        'center': [22.5, 3.2, 0.0],       # 22.5m ahead, 3.2m to the right
        'size': [2.1, 0.6, 1.5],          # bike is longer and narrower than a person
        'orientation': 0.05                # heading mostly straight (parallel to vehicle)
    },
    'confidence': 0.73,                   # lower confidence because partially blocked
    'velocity': [3.2, 0.1, 0.0],          # moving forward at ~3.2 m/s (reasonable bike speed)
    'attributes': {
        'occluded': True,                 # partially hidden (e.g., by a parked car)
        'truncated': False,
        'num_frames_tracked': 8,          # just appeared recently
    }
}
```

**Example 3: Vehicle merging into lane**
```python
{
    'timestamp': 1234567890.500,
    'agent_id': 'car_102',
    'agent_class': 'car',
    'bbox_3d': {
        'center': [18.7, 4.5, 0.0],       # 18.7m ahead, 4.5m to the right (adjacent lane)
        'size': [4.7, 1.8, 1.5],          # standard car dimensions
        'orientation': -0.12               # angled slightly toward our vehicle (turning in)
    },
    'confidence': 0.96,                   # high confidence, large vehicle, clear
    'velocity': [4.2, -0.9, 0.0],         # moving forward at 4.2 m/s, steering left (-0.9 m/s leftward)
    'attributes': {
        'occluded': False,
        'truncated': False,
        'num_frames_tracked': 120,        # been visible for several seconds
    }
}
```

**Example 4: Truck partially off-road**
```python
{
    'timestamp': 1234567890.500,
    'agent_id': 'truck_5',
    'agent_class': 'truck',               # explicitly classified as truck, not generic car
    'bbox_3d': {
        'center': [35.1, -1.8, 0.0],      # 35m ahead, slightly to the left, parked
        'size': [8.5, 2.5, 2.8],          # much larger than a car
        'orientation': 0.0                 # aligned with road
    },
    'confidence': 0.98,                   # very clear, large, unoccluded
    'velocity': [0.0, 0.0, 0.0],          # stationary (parked)
    'attributes': {
        'occluded': False,
        'truncated': True,                # extends beyond the sensor's field of view at this range
        'num_frames_tracked': 300+,       # stationary object, been there many frames
    }
}
```

**How these are used for caption generation (Step 4A).**

The system doesn't store hand-written captions for every clip — instead, it **auto-generates templates** from these perception detections. For example, given the detections above at a particular timestamp, you might generate:

```
"Pedestrian in crosswalk on the left, ~15m ahead, facing perpendicular to vehicle direction"
  ↓ (from pedestrian track_18)

"Cyclist partially occluded by parked car on the right, ~22m ahead, moving forward"
  ↓ (from cyclist track_7)

"Car merging into lane from right, ~18m ahead, traveling at 4.2 m/s"
  ↓ (from car track_102)

"Large truck parked on the left shoulder ahead"
  ↓ (from truck track_5)

Combined, multi-agent caption:
"Pedestrian in crosswalk on left, partially-occluded cyclist on right, and car merging in from right lane; truck parked on shoulder ahead"
```

These auto-generated captions are **weak labels** (they can be noisy or miss nuance) but are **abundant and cost-free** to generate — you get one or more per scene. This lets you bootstrap millions of (video clip, caption) pairs to train the CLIP-style embedding with, which you then refine with a smaller set of **human-written captions** on interesting/rare scenarios (the long-tail cases from [driving_scene_search](design.md)).

**Why confidence matters:** detections with low confidence (e.g., `confidence=0.65`) are often filtered out at the planning stage (the planner won't commit a safety-critical decision based on a low-confidence detection), so they're often de-weighted or ignored in caption generation. High-confidence detections (> 0.85) are treated as reliable and heavily weighted in the caption.

**Why tracking matters:** the `agent_id` and `num_frames_tracked` let you reason about **temporal stability** — a detection that just appeared (`num_frames_tracked=1`) is less reliable than one that's been consistently tracked for many frames, because the tracker filters out noise and confirms agent identity over time.

*Pointer:* [solution.md](solution.md), Step 4A "Multimodal Video/Text Embeddings" — "Where does training data (video, text) come from?"

---

### Q: What is a ViT (Vision Transformer) encoder vs. a CNN encoder? How do they differ, and which should I use for video encoding? `[ANSWERED]`

**A:**

Both are **image encoders** — they take a single image (or in our case, a single frame from a video clip) and produce a dense vector (or a grid of vectors) summarizing the image's visual content. They differ fundamentally in their architectural building block.

| Aspect | CNN (Convolutional Neural Network) | ViT (Vision Transformer) |
|---|---|---|
| Core operation | Convolution: a small learnable filter slides over the image, computing a weighted sum at each position | Self-attention: every spatial position in the image learns to compare itself against every other position in the image |
| How it sees structure | **Local-first**: early layers see only small patches (the filter size, e.g. 3×3); nearby pixels are fused together; later layers gradually expand receptive field to see larger context | **Global-first**: the entire image is tokenized into patches (e.g. 16×16 regions), then each patch token sees **all other patches** immediately via self-attention; no need for a deep stack to capture long-range dependencies |
| Parameter efficiency | Lower: filters are "shared" across all spatial positions (one 3×3 filter reused billions of times across the image), so few parameters relative to image size | Higher: each spatial position has its own set of query/key/value weights that scale with sequence length (number of patches) |
| Inductive bias | Strong: assumes locality (nearby pixels matter more than distant ones) and translation-equivariance (the learned filters work the same way everywhere in the image); this is great for natural images | Weaker: doesn't assume locality; must learn spatial relationships from scratch via attention patterns; this requires more training data and compute but is more flexible |
| Compute cost | Lower for small/medium images; convolution is highly optimized in hardware (GPUs, TPUs) | Higher: attention is $O(n^2)$ in the number of tokens $n$, though recent variants (flash attention, sparse attention) have improved this |
| Training data requirement | Works well with moderate amounts of labeled data; implicit locality regularization helps | Wants large-scale pretraining on millions of images to learn general visual patterns without the locality shortcuts CNN gets for free |

**Concrete example: encoding a 10-second driving video at 4 fps = 40 frames, each 1920×1080 pixels.**

**CNN approach:**
1. Resize frames to e.g. 224×224 (standard size).
2. Stack all 40 frames into a 3D volume, or encode each frame separately.
3. Use a ResNet-50 or EfficientNet backbone:
   - First conv layer (7×7 kernel) sees only 7×7 patches.
   - After 4–5 layers, the receptive field has grown to ~100×100 pixels (enough to see a person, but not a whole intersection).
   - Output: a 2048-D vector per frame (or a spatial grid of 2048-D vectors, before global pooling).
4. Very fast to compute (~0.5–1 second per video on a GPU).

**ViT approach:**
1. Resize frames to 224×224.
2. Divide each frame into 16×16-pixel patches → $(224/16)^2 = 196$ patches per frame.
3. Embed each patch into a d=768-D token.
4. Stack temporal positional encodings so the model knows which frame each patch came from.
5. Feed all 40 frames × 196 patches = 7840 tokens through a ViT backbone (e.g. ViT-Base):
   - Each token, in the **first attention layer**, sees all 7840 tokens at once.
   - Global context emerges immediately, unlike CNNs where you need depth to grow the receptive field.
   - Output: fused representation, e.g. a `[CLS]` token or mean-pooled tokens.
6. Slower to compute than ResNet (~2–3 seconds per video on a GPU, or faster with optimizations).

**Why the design recommends ViT + temporal aggregator (Step 4A):**

The solution suggests "*ViT-style per-frame encoder + temporal aggregator (attention-pool or small temporal transformer)*" as the default for this problem. Here's why:

1. **Reuses pretrained general models:** Large pretrained ViT models (e.g. ViT-Base from `timm`, or domain-adapted ones from open-source datasets) are widely available and freeze-able, meaning you don't need to retrain from scratch — just plug in the pretrained weights and use them as a feature extractor.

2. **Cost-effective scaling:** Encoding billions of frames individually (even at the slightly-higher-per-frame ViT cost) and pooling them temporally is cheaper than training a full 3D-conv or ViT-on-video backbone from scratch, which would require much more compute and labeled data.

3. **Better on long-range dependencies:** In the driving domain, a cyclist's intent (are they swerving?) can depend on their trajectory *across many frames*, not just local motion — ViT's global-attention-first design is more naturally suited to this than a CNN's local-first design.

4. **Alignment with text encoder:** Most pretrained text encoders (e.g. BERT, CLIP text tower) are also Transformer-based, so using a ViT for the image side too keeps the embedding space more naturally aligned (similar scales, similar token counts, both using attention) before the final projection.

**When would you pick CNN instead?**

- If you had a large, carefully curated, in-domain labeled dataset and unlimited compute budget, a 3D-CNN or SlowFast model (which fuses spatial and temporal convolutions directly) might be better at capturing fine-grained motion.
- If latency was critical (sub-100ms per frame encoding) and you were willing to sacrifice some accuracy for speed, a lightweight CNN (MobileNet, EfficientNet-Lite) would be faster than ViT.
- If you're on embedded/on-device hardware with limited memory, a CNN's smaller parameter count (especially with pruning) might be mandatory.

**Why not use ViT-on-video directly (a video transformer, not frame-by-frame)?**

Video transformers exist (ViViT, TimeSformer, VideoMAE) and directly process the 3D spatio-temporal token grid. They're more expressive than frame-by-frame ViT + temporal pooling, but:
- They're newer and less stable across codebases.
- They're significantly more compute-expensive (thousands of tokens per clip vs. hundreds).
- For this system's scale (billions of clips), the frame-by-frame + temporal-aggregator approach trades a bit of precision for much better cost-scaling — a deliberate tradeoff named explicitly in Step 4A of the solution.

*Pointer:* [solution.md](solution.md), Step 4A "Multimodal Video/Text Embeddings" — Video Tower.

---

### Q: What's the difference between video embeddings and image embeddings, and how exactly is a video embedding calculated? `[ANSWERED]`

**A:**

**Image embedding** — a single image goes through an encoder (e.g. a ViT or CNN) that produces one dense vector summarizing the image's *spatial* content: objects, layout, appearance. There is no notion of time — the encoder has no way to know what happened before or after this frame.

**Video embedding** — a video clip is a *sequence* of $T$ frames, and a video embedding needs to summarize appearance **and** motion/temporal order. This matters a lot for this problem: a single frame of "cyclist next to the lane" looks identical whether the cyclist is riding straight or about to swerve into traffic — you can only tell the difference by looking at how the cyclist's position changes **across frames**. An image embedding structurally cannot capture that; a video embedding must.

| Aspect | Image embedding | Video embedding |
|---|---|---|
| Input | one frame | sequence of $T$ frames |
| Captures | appearance / spatial layout | appearance **+** motion **+** temporal order |
| Can distinguish | "a cyclist is present" | "cyclist riding straight" vs. "cyclist swerving" |
| Typical encoder | ViT / CNN | per-frame ViT + temporal aggregator, 3D-conv net, or a video transformer |
| Compute cost | $O(1 \text{ frame})$ | $O(T \text{ frames})$ — noticeably higher |

**How a video embedding is actually calculated.** There are a few standard approaches, roughly in order of how explicitly they model time:

**1. Frame-level encode + temporal pooling (simplest, what Step 4A of the solution uses as the default).**
- Sample frames from the clip at a fixed rate (e.g. 2–4 fps) → $T$ frames.
- Run each frame through a **shared** image encoder (same weights reused per frame) → per-frame embeddings $f_1, \dots, f_T \in \mathbb{R}^d$.
- Aggregate across time into one clip embedding $v \in \mathbb{R}^d$. The aggregation is what actually "makes it a video embedding" instead of just a bag of image embeddings:
  - **Mean/max pool:** $v = \frac{1}{T}\sum_t f_t$ — cheap, but discards ordering (a clip played backwards gives the same embedding).
  - **Attention pool (better, learns which frames matter most):**
    $$a_t = \frac{\exp(w^\top f_t)}{\sum_{t'} \exp(w^\top f_{t'})}, \qquad v = \sum_t a_t f_t$$
    where $w$ is a learned weight vector. This lets the model up-weight the exact moment a pedestrian steps off the curb rather than treating every frame equally, but is still a weighted-average — it doesn't explicitly encode *order*.
  - **Temporal transformer/RNN aggregator:** feed $f_1, \dots, f_T$ through a small self-attention block or GRU with positional encodings over time, then pool the output. This is what actually captures order (e.g. "cyclist moving left-to-right" vs. "right-to-left"), which pure pooling cannot.

**2. 3D convolution / factorized spatio-temporal convs (e.g. I3D, SlowFast).** The convolution kernel itself spans space *and* time (e.g. a $3\times3\times3$ kernel), so motion is captured directly inside the conv operation rather than bolted on afterward via pooling.

**3. Video transformers (e.g. ViViT, VideoMAE, TimeSformer)** — the approach referenced in [solution.md](solution.md) Step 4A. The clip is tokenized into spatio-temporal patches (small space×time cubes, not just space), and self-attention runs over all tokens (or is factorized into a spatial-attention step then a temporal-attention step for efficiency). A special temporal `[CLS]` token, or a final pooling layer, produces the single clip embedding. This is the most expressive option — attention can directly relate "this pixel region at frame 3" to "this pixel region at frame 8" — at higher compute cost.

**4. Two-stream fusion (older, still used as a supplement).** One stream encodes RGB frames (appearance), a second stream encodes optical flow computed between consecutive frames (explicit motion signal), and the two embeddings are fused (concat + small MLP) before the final projection. Useful when you want an explicit, interpretable motion signal in addition to a learned one.

In this design, the recommendation (Step 4A) is a ViT-style per-frame encoder + a temporal aggregator (attention-pool or a small temporal transformer) rather than full 3D convs — it reuses strong pretrained image encoders and is cheaper to scale to billions of segments than training a 3D-conv or full video-transformer backbone from scratch, at some cost in how finely it captures motion versus a dedicated video-transformer backbone.

*Pointer:* [solution.md](solution.md), Step 4A "Multimodal Video/Text Embeddings" — Video Tower.

---

## Architecture

### Q: What exactly is a "CLIP embedding," and how is such a model trained end-to-end — what does the contrastive loss look like, and what does a training row/label actually look like? `[ANSWERED]`

**A:**

**What "CLIP embedding" means.** CLIP (Contrastive Language-Image Pretraining) is the architecture Step 4A's video/text embedding is modeled after. It's a **dual-tower** (a.k.a. two-tower) model: one encoder for the visual input (image or, here, video), one encoder for text, each projecting into the **same** $d$-dimensional vector space. "A CLIP embedding" just means a vector produced by one of these towers, L2-normalized, such that semantically matching image/video and text land close together in that shared space (measured by cosine similarity).

**Training data — what a "row" looks like.** There are no hand-assigned numeric labels. A training example is simply a **matched pair**:

```
row_i = ( video_i, caption_i )

e.g.
row_1 = ( clip_00001.mp4, "pedestrian jaywalking mid-block at night in the rain" )
row_2 = ( clip_00002.mp4, "car making an unprotected left turn with oncoming traffic" )
row_3 = ( clip_00003.mp4, "cyclist riding straight in the bike lane, clear daytime"   )
```

The "label" is implicit: for row $i$, the correct match is caption $i$, and *every other caption in the same training batch* is treated as a negative for video $i$ (and vice versa). This is why it's called **self-supervised contrastive learning** — the supervision comes from the pairing itself (which video goes with which caption), not from a human-assigned class or score. For the driving-scene use case (Step 4A), these pairs come from templated captions auto-generated off perception-stack detections (cheap, large-scale, weaker quality) plus a smaller set of human-written captions (expensive, higher quality).

**Forward pass over a batch.** Take a batch of $N$ pairs $(v_1, t_1), \dots, (v_N, t_N)$:
1. Encode all videos with the video tower → $V_1, \dots, V_N \in \mathbb{R}^d$, then L2-normalize each.
2. Encode all captions with the text tower → $T_1, \dots, T_N \in \mathbb{R}^d$, then L2-normalize each.
3. Compute the full $N \times N$ **cosine similarity matrix**: $S_{ij} = V_i \cdot T_j$ (since both are unit-normalized, dot product = cosine similarity, in $[-1, 1]$).
4. The diagonal entries $S_{ii}$ are the *positive pairs* (video $i$ with its true caption $i$); every off-diagonal entry $S_{ij}, i \neq j$ is a negative — this is what makes batch size matter so much for CLIP-style training: a bigger batch means more negatives per positive, and better-calibrated embeddings.

**The loss function — symmetric InfoNCE (a.k.a. the CLIP loss).**

$$
\mathcal{L} = \frac{1}{2}\left( \underbrace{-\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^N \exp(S_{ij}/\tau)}}_{\text{video} \rightarrow \text{text direction}} \;+\; \underbrace{-\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^N \exp(S_{ji}/\tau)}}_{\text{text} \rightarrow \text{video direction}} \right)
$$

**Where:**
- $S_{ij}$ = cosine similarity between video $i$'s embedding and text $j$'s embedding (the similarity matrix from step 3 above).
- $\tau$ = a learnable (or fixed) **temperature** scalar. Dividing by $\tau < 1$ sharpens the softmax — it controls how confidently the model is pushed to separate the correct pair from the negatives.
- $N$ = batch size = number of in-batch negatives per positive is $N-1$.
- The first term is just standard **cross-entropy** treating row $i$ of $S$ as logits over $N$ classes, where the correct class is $i$ itself (video-to-text retrieval). The second term does the same over columns (text-to-video retrieval). Averaging both directions keeps the embedding space symmetric — good for both "search by text" and "search by example video."

**Concrete worked mini-example ($N=3$).** Suppose after encoding, cosine similarities come out as:

```
                caption_1   caption_2   caption_3
video_1          0.85        0.10        0.05      <- row for video_1
video_2          0.20        0.78        0.30
video_3          0.15        0.25        0.90
```

Row 1 (video_1 → text): softmax over $[0.85, 0.10, 0.05]/\tau$ should put nearly all mass on column 1 (the true match) — the loss for this row is $-\log(\text{softmax}_1)$, which shrinks as $S_{11}$ pulls further above $S_{12}, S_{13}$. Training simply pushes diagonal entries up and off-diagonal entries down across many such batches, for millions of (video, caption) pairs — no explicit numeric label is ever provided beyond "which column is the diagonal."

**Why this matters for the search system:** once trained, at *query time* we only need to run the text tower once on the query string and do a nearest-neighbor lookup against precomputed video embeddings (Step 4B) — the expensive joint (video, text) comparison from training is never needed at serving time, which is exactly why this architecture scales to billions of clips.

*Pointer:* [solution.md](solution.md), Step 4A "Multimodal Video/Text Embeddings" — Architecture: Dual/Multi-Tower Contrastive Embedding.

---

### Q: How exactly does the re-ranking step use a "cross-attention model"? What is cross-attention, and how would it work here? `[ANSWERED]`

**A:**

**Why re-ranking exists at all.** The dual-tower embedding from Step 4A (CLIP-style) is what makes ANN search over billions of segments possible (Step 4B) — but it pays for that speed with an information bottleneck: a whole video clip and a whole text query each get squashed down into a *single* fixed-size vector ($d{=}512$–$1024$) **before** they're ever compared. Two things that are actually a close match can still end up with a mediocre cosine-similarity score if that single-vector summary lost the specific detail that made them match. Cross-attention re-ranking exists to recover that lost precision, but only for a short list of candidates — it's too expensive to run over the full archive (see the cost breakdown below).

**What cross-attention is, mechanically.** Standard (self-)attention lets a sequence look at *itself*: every token computes a query, and attends to keys/values coming from **the same** sequence. **Cross-attention** is the same computation, except the query comes from one sequence and the keys/values come from a **different** sequence:

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Where:**
- $Q = X_a W_Q$ — queries projected from sequence $A$ (e.g. the text query's word tokens).
- $K = X_b W_K,\; V = X_b W_V$ — keys and values projected from sequence $B$ (e.g. the candidate video's frame/patch tokens) — a **different** sequence than the one $Q$ came from. (In self-attention, $A = B$; in cross-attention, $A \neq B$.)
- $\sqrt{d_k}$ — scaling factor (dimension of each key vector) that keeps the softmax's input from growing too large and saturating gradients.
- The output is, for **each query token**, a weighted blend of the *other* sequence's value vectors — i.e. "for this word in the query, which specific parts of the candidate video are relevant, and what do they look like?" This is exactly the same mechanism used for the encoder→decoder link in the original Transformer (the decoder's tokens cross-attend to the encoder's output) and in vision-language fusion models like ALBEF/VisualBERT.

**How it's applied here, step by step.** Unlike the dual-tower retrieval embedding — which throws away everything except one pooled vector per side — the re-ranker keeps the **full token sequences** on both sides:
1. Text side: the query's word/subword tokens → text-encoder hidden states $t_1, \dots, t_m$ (one vector per token, *not* pooled into one vector).
2. Video side: the candidate clip's frame/patch tokens → video-encoder hidden states $v_1, \dots, v_n$ (one vector per spatio-temporal patch, again *not* pooled — see [[this FAQ's video-embedding answer above]] for how those per-frame tokens are produced before pooling).
3. Stack several **cross-attention transformer layers**: text tokens attend over video tokens (query = text, key/value = video) so each word can pull in specific visual evidence — e.g. the token "swerving" learns to attend heavily to the 2-3 frames where the cyclist's trajectory actually bends, rather than the model only ever seeing one averaged whole-clip vector. Many architectures alternate this with the reverse direction (video tokens attend over text tokens) so both sides get refined jointly.
4. After a few such layers, pool the fused representation (e.g. a dedicated `[SCORE]` token, similar to BERT's `[CLS]`, or a mean-pool over the fused tokens) and pass it through a small MLP head → a single **relevance score** for this (query, candidate) pair.

```
Text tokens:   [cyclist] [swerving] [into] [lane]        (m tokens)
                    │         │        │       │
                    ▼         ▼        ▼       ▼
         ┌─────────────────────────────────────────┐
         │        Cross-Attention Layer(s)           │   Q = text tokens
         │  each text token attends over ALL video    │   K,V = video tokens
         │  tokens to pull in relevant visual detail   │
         └─────────────────────────────────────────┘
                    │         │        │       │
                    ▼         ▼        ▼       ▼
              fused, video-aware text-token representations
                              │
                              ▼
                    pool (e.g. [SCORE] token)
                              │
                              ▼
                     MLP  →  relevance score
                              │
Video tokens: [v_1] [v_2] ... [v_n]  (n spatio-temporal patch tokens, frame 1..T)
```

**How it's trained.** Given labeled or weakly-labeled triples $(query, candidate, \text{relevant?})$ — e.g. from the same auto-generated-caption + human-curated data used for the dual-tower model (Step 4A), plus **hard negatives** mined from the dual-tower's own near-miss retrievals (candidates that scored high on cosine similarity but are actually irrelevant — exactly the cases re-ranking needs to learn to fix) — train with either:
- a binary cross-entropy loss on relevant vs. not-relevant, or
- a pairwise/listwise ranking loss (e.g. a hinge loss pushing the relevant candidate's score above the irrelevant candidates' scores for the same query), which more directly optimizes for *ordering* rather than calibrated probabilities.

**Why not just use cross-attention everywhere and skip the dual-tower entirely?** This is the actual reason the two-stage retrieve-then-rerank design exists:
- The dual-tower embeddings can be **precomputed once per video, independently of any query**, then compared to a query with one cheap dot product — this is what makes ANN search over billions of vectors feasible ($O(\log N)$-ish with HNSW, not $O(N)$).
- Cross-attention **cannot** be precomputed this way — the whole point is that the model looks at the query and the candidate *together*, so a new forward pass is required for **every (query, candidate) pair**. Running it against billions of candidates per query would be computationally infeasible at interactive latency.
- The fix (Step 4B): use the cheap dual-tower + ANN search to cut billions of candidates down to a short list (e.g. top-200), and only run the expensive cross-attention re-ranker on that short list — getting most of cross-attention's precision at a small fraction of its cost. This is the standard **retrieve-then-rerank** pattern used broadly in large-scale search/recsys, not something specific to this problem.

*Pointer:* [solution.md](solution.md), Step 4B "Fast Vector Similarity Search at Scale" — "Re-ranking".

---

## Tradeoffs

*(No questions logged yet.)*

---

## Math

### Q: In batch-based contrastive learning (CLIP), how do we prevent true positives from being treated as negatives? What if two different captions both match the same video? `[ANSWERED]`

**A:**

This is a real and important issue in contrastive learning — **in-batch negatives collision problem**. The CLIP loss (from the [[CLIP embedding question above]]) assumes that within a batch of $N$ pairs, only the diagonal $(i,i)$ entries are true positives, and all off-diagonal entries $(i, j), i \neq j$ are negatives. But what if caption $j$ *also* matches video $i$? Then the loss unfairly penalizes the model for aligning them, hurting embedding quality. Here's how to avoid this:

### 1. Dataset Construction: One-to-One Pairing (Primary defense)

The strongest fix is to **ensure each video appears in exactly one training example (row) with exactly one caption**. This makes the one-positive-per-row assumption true by design:

```python
# DON'T do this (creates collision risk):
training_data = [
    ('video_A.mp4', 'pedestrian crossing street'),
    ('video_A.mp4', 'person jaywalking'),        # SAME video, DIFFERENT caption
    ('video_B.mp4', '...'),
]
# If both captions match video_A, the second one will be penalized as a negative
# for video_A's example.

# DO this instead:
training_data = [
    ('video_A.mp4', 'pedestrian crossing street'),  # pick ONE canonical caption
    ('video_A.mp4_v2.mp4', 'person jaywalking'),    # treat each (video, caption) as a distinct training instance
    ('video_B.mp4', '...'),
]
# Each row is independent; no collisions.
```

For the driving-scene system (Step 4A), this means:
- Each video clip gets **one auto-generated caption** from the perception-stack detections at a representative timestep in that clip.
- If you have multiple **human-written captions** for the same clip (e.g., an engineer describes the same clip in multiple ways), you either pick one as canonical, or you create separate synthetic "clips" (same video, different windowing) for each caption.
- A small number of high-value clips may have multiple curated captions — handle these as a separate batch (see Multi-Positive approach below) rather than mixing them with the main single-positive pipeline.

### 2. Randomized Batching: Low-Probability Collision

Even if your dataset *could* have duplicates, **randomizing which examples go into a batch together** makes collisions rare:
- If video $A$ appears $m$ times in the dataset (with different captions), the probability that two of those instances end up in the same batch of size $N$ is $\approx O(m^2 / \text{total\_dataset\_size})$.
- With careful filtering (see Deduplication below) to keep $m$ small, collisions become negligible even with randomized batching.

### 3. Hard Negative Mining: Turn Duplicates into Useful Training Signal

If you *do* have multiple captions for the same video, **flip the problem and use it as a hard negative signal**:

```python
# If you have:
video_A = ('cyclist_swerving_into_lane.mp4',)
caption_A1 = 'cyclist swerving into the lane'
caption_A2 = 'cyclist riding straight in the bike lane'

# Interpret this as:
# - (video_A, caption_A1) is a positive pair
# - (video_A, caption_A2) is a HARD NEGATIVE: a caption that sounds plausible
#   but doesn't match the actual video
# - Train the model to distinguish them

# This forces the model to learn finer distinctions than it would with
# easy negatives (completely unrelated captions).
```

In practice, if you had a clip of a cyclist actually swerving and also a caption claiming they were riding straight, the model learns that embeddings of "swerving" captions should *not* align with this video's embedding — a valuable signal, especially if one caption was a human mistake or an ambiguous scenario.

### 4. Explicit Multi-Positive Handling

Some systems explicitly handle multi-positive scenarios with a modified loss:

$$\mathcal{L}_{\text{multi-positive}} = -\frac{1}{|P_i|} \sum_{p \in P_i} \log \frac{\exp(S_{ip}/\tau)}{\sum_{j=1}^N \exp(S_{ij}/\tau)}$$

where $P_i$ is the set of all captions that match video $i$. If you have $k$ captions for video $i$, you sum over all $k$ of them as positives before the denominator (which sums over all $N$ items in the batch).

However, this requires:
- Explicit multi-positive labels in your training data (manual curation).
- Careful batch construction (ensuring all positives for a video are in the same batch, or splitting the sum across batches).
- For this system, overkill — the one-to-one design is simpler and works fine.

### 5. Deduplication: Preprocess the Dataset

Before training, scan for **near-duplicate** (video, caption) pairs:
- Compute embeddings of all captions (using a pretrained text encoder).
- Cosine-similar captions above a threshold (e.g., $> 0.95$) paired with the same or visually-similar videos get deduplicated (keep one, discard others).
- This is a one-time preprocessing cost, preventing future collisions.

### Concrete example: How the driving-scene system avoids it

In Step 4A, captions are auto-generated from perception-stack detections:

```python
# For clip_00042.mp4, at timestamp t=5.2s:
detections_t = [
    {'agent_id': 'ped_5', 'class': 'pedestrian', 'position': [12, -2], ...},
    {'agent_id': 'car_7', 'class': 'car', 'position': [25, 3], ...}
]

# Generate caption template:
caption = f"Pedestrian on left ({12}m ahead), car on right ({25}m ahead)"
# This caption is paired with clip_00042.mp4 exactly once in the training data.

# A different clip, clip_00043.mp4, might have very similar detections
# but a slightly different timestamp or scene, so it gets a different caption:
caption_2 = f"Pedestrian on left ({11.8}m ahead), car on right ({25.1}m ahead)"

# These are treated as separate training rows:
training_data = [
    ('clip_00042.mp4', 'Pedestrian on left (12m ahead), car on right (25m ahead)'),
    ('clip_00043.mp4', 'Pedestrian on left (11.8m ahead), car on right (25.1m ahead)'),
    ...
]

# Even though the captions are very similar, they're paired with different videos,
# so there's no collision in the loss.
```

### Why larger batches still matter

Even with no collisions, **batch size affects learning quality**:
- A batch of size $N$ gives you $N-1$ in-batch negatives per positive.
- Larger $N$ → more negatives to push apart → better-calibrated embeddings.
- CLIP uses batches of 32K–128K across distributed replicas, not because of collisions, but because that scale of negatives helps the model learn a well-separated embedding space.
- Trade-off: larger batches need more GPU memory and distributed training infrastructure.

### Summary

The main answer: **design the dataset so each video pairs with one canonical caption** (the fundamental assumption of CLIP-style in-batch negatives). Secondary defenses: randomized batching, hard-negative mining, deduplication, and explicit multi-positive losses for high-value cases. For the driving-scene system, the one-to-one design is natural (each clip gets one auto-generated caption) and sufficient.

*Pointer:* [solution.md](solution.md), Step 4A "Multimodal Video/Text Embeddings" — Architecture: Dual/Multi-Tower Contrastive Embedding; [[CLIP embedding question above]].

---

### Q: What exactly are "image tokens" or "video tokens"? I understand text tokens, but vision tokens are confusing — aren't images just RGB pixels? `[ANSWERED]`

**A:**

Great question — this terminology is confusing because **vision "tokens" are fundamentally different from text tokens**, even though both are called "tokens."

**Text tokens:** discrete, symbolic units
```
Text: "The cat is sleeping"
Tokenize: ["The", "cat", "is", "sleeping"]
Then embed: [token_id_1, token_id_2, token_id_3, token_id_4]
           (e.g., [262, 1234, 318, 5678])
Each token is an integer index into a fixed vocabulary (like a dictionary lookup).
```

**Image tokens:** NOT discrete — they're patches of pixels + their learned embeddings

The confusion arises because vision researchers borrowed the word "token" from NLP, but it means something different:

```
Image: 224×224 RGB pixels (224 × 224 × 3 = 150,528 individual float values)
                ▼
Patch-ify: divide into non-overlapping 16×16 patches
           224/16 = 14, so 14 × 14 = 196 patches
                ▼
Embed each patch: 
  For each 16×16 patch (~768 RGB pixels):
    - Flatten into a 1D vector: [R₁, G₁, B₁, R₂, G₂, B₂, ..., R₇₆₈, G₇₆₈, B₇₆₈]
    - Pass through a LINEAR PROJECTION (a learnable weight matrix W):
      token_embedding = W @ flattened_patch  
      (e.g., W is 768×768, so you get a 768-D output from 768-D input)
    - Result: one 768-D dense vector per patch (not an integer index!)

Now you have 196 vectors (one per patch), each vector is 768 floats.
These 196 vectors are what people call "image tokens" — but they're NOT
discrete symbols (indices), they're DENSE VECTORS (embeddings).
```

**Why is it called "tokens" at all if it's not discrete?**

Because the Transformer architecture (which powers modern ViT / CLIP models) is fundamentally a **sequence-to-sequence model**. It expects:
- A sequence of **fixed-size inputs**
- Each input can be anything (a discrete integer, a dense vector, etc.)

Text Transformers use discrete token IDs because language is naturally discrete.
Vision Transformers use patch embeddings (dense vectors) because images are naturally continuous.

The word "token" in both cases just means *"one item in a sequence that will be fed to a Transformer,"* but:
- Text token = integer ID (e.g., 262)
- Vision token = dense vector (e.g., 768-D float vector)

```
               Text Domain              Vision Domain
                   │                          │
        "The cat is sleeping"         224×224 RGB image
                   │                          │
          Tokenize (discrete)       Patchify + embed (continuous)
                   │                          │
        [262, 1234, 318, 5678]      [token₁, token₂, ..., token₁₉₆]
      (integers, indices)             (768-D vectors, each)
                   │                          │
        Lookup in embedding table    (already embedded, no lookup needed)
                   │                          │
        [emb₁, emb₂, emb₃, emb₄]   [token₁, token₂, ..., token₁₉₆]
       (all 768-D vectors)            (already 768-D vectors)
                   │                          │
                   └──────────────────┬───────┘
                                      ▼
                          Feed to Transformer
```

**Concrete walkthrough: ViT encoding a 224×224 frame from a driving video**

```python
# Raw image: shape (3, 224, 224) — 3 color channels, 224×224 pixels
image = load_frame('frame_001.png')  # shape: (3, 224, 224)

# Step 1: Divide into 16×16 patches
patch_size = 16
num_patches_per_side = 224 / 16  # = 14
num_patches_total = 14 * 14  # = 196 patches

patches = image.unfold(1, 16, 16).unfold(2, 16, 16)
# Result: shape (3, 14, 14, 16, 16) — a 14×14 grid of 16×16 patches
# Reshape to (196, 768) — 196 patches, each flattened to 768 values (3 channels × 16 × 16)

# Step 2: Project each patch to token embedding dimension (e.g., 768 → 768)
W_proj = torch.randn(768, 768)  # learnable projection matrix
patch_embeddings = (patches.reshape(196, -1)) @ W_proj  
# Result: shape (196, 768) — 196 tokens, each a 768-D dense vector

# These 196 vectors are the "image tokens" that get fed to the Transformer.

# Transformer then does:
# - Self-attention: each token attends over all 196 tokens
# - Output: 196 refined token vectors (or a pooled [CLS] token for image classification)
```

**Video tokens (the sequence dimension extends to time):**

When you have a video (a sequence of frames), the tokenization extends naturally:

```
1 frame:    14×14 patches = 196 spatial tokens
10 frames:  14×14×10 = 1960 spatio-temporal tokens

Each token still represents a small 3D cube of the video (16×16 pixels × a few frames 
or 1 frame depending on how you factorize the temporal dimension), and each is still
a 768-D embedding.

A video Transformer (like ViViT) attends over all 1960 tokens at once, learning
which spatial regions and which time frames matter for understanding the clip.
```

**Why patch embeddings instead of just feeding raw pixels?**

1. **Dimensionality reduction:** 224×224×3 = 150K pixel values → 196×768 = 150K values (similar scale, but organized nicely).
2. **Semantic grouping:** each token represents a small spatial region, which is closer to human visual intuition than individual pixels.
3. **Inductive bias:** Transformers work best with sequences of 100–10K items; raw pixels would be too large. Patches are a natural level of granularity.
4. **Computational efficiency:** Transformers scale as $O(n^2)$ in sequence length; working with 196 patches is far cheaper than 150K pixels.

**Comparison table: tokens across modalities**

| Modality | Raw input | Tokenization | Token type | Count per input |
|---|---|---|---|---|
| Text (NLP) | "cat is here" | split by spaces/subwords | **discrete integer** (vocabulary index) | ~10 |
| Image (ViT) | 224×224 RGB | split into 16×16 patches | **dense 768-D vector** (patch embedding) | 196 |
| Video (ViViT) | 8 frames, 224×224 each | 3D patches (16×16×T) | **dense 768-D vector** (spatio-temporal embedding) | ~500–2000 |
| Audio (HuBERT) | waveform | mel-spectrogram frames | **dense vector** (frame embedding) | ~1000 |

Notice: **only text uses discrete indices.** All visual/audio modalities use dense embeddings, but we call them "tokens" because they all get fed to Transformers as sequences.

**Why the confusion exists:**

The term "token" in vision is basically a historical accident — when Dosovitskiy et al. introduced Vision Transformers (ViT), they borrowed the term "token" from NLP as a shorthand for "sequence element," but the actual data type is completely different (dense vector vs. integer index). A more precise term would be **"patch embedding"** or **"visual embedding,"** but "token" stuck in the community.

**TL;DR:** Image tokens are **patches of pixels embedded as dense vectors,** not discrete symbols. They're called "tokens" because they're fed to Transformers as a sequence, just like text tokens, but they're fundamentally different data types (768-D floats vs. integers). Vision researchers use the word "token" loosely; what matters is that they're fixed-size sequence elements the Transformer can process.

*Pointer:* [solution.md](solution.md), Step 4A "Multimodal Video/Text Embeddings" — Video Tower; [[ViT vs CNN encoder question above]].

---

## Tradeoffs

*(No questions logged yet.)*

---

## Follow-up / Interview Extensions

*(No questions logged yet.)*
