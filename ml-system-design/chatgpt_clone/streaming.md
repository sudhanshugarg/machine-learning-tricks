# Understanding Async/Await and Concurrency in Python

## Overview
This document explores Python's asyncio module, which is fundamental to understanding streaming in modern applications. Streaming often requires handling multiple I/O operations concurrently (e.g., reading from multiple data sources, handling multiple client connections).

---

## 1. Basic Async/Await Concepts

### Problem: Calling Async Functions

When you define an async function, it returns a **coroutine object** when called—it doesn't execute immediately.

```python
async def task_a():
    await asyncio.sleep(0.1)

# This does NOT execute task_a!
task_a()  # Returns a coroutine object, doesn't run anything
```

To actually run an async function from a sync context (like `if __name__ == "__main__"`), you need `asyncio.run()`:

```python
if __name__ == "__main__":
    asyncio.run(task_a())  # This actually executes task_a
```

### Key Rule
- **Sync functions** can be called from anywhere (sync or async contexts)
- **Async functions** can only be awaited from async contexts
- `asyncio.run()` is the bridge that lets you start an async function from a sync context

---

## 2. Sequential vs. Concurrent Execution

### Sequential Execution (Default)
Using `await` statements one after another executes tasks **sequentially**:

```python
async def regular():
    start = time.time()

    await task_a()  # Wait 0.1s
    await task_b()  # Then wait 0.1s

    print(f"Duration: {time.time() - start:.4f}s")  # ~0.2s
```

The `await` keyword is **blocking** — it waits for the task to complete before moving to the next line.

### Concurrent Execution
To run tasks **concurrently** (at the same time), use `asyncio.gather()`:

```python
async def regular():
    start = time.time()

    await asyncio.gather(task_a(), task_b())  # Both run at same time

    print(f"Duration: {time.time() - start:.4f}s")  # ~0.1s (not 0.2s)
```

**Key Insight:** Even though Python is single-threaded, `gather()` allows the event loop to manage multiple concurrent tasks through **task switching**.

---

## 3. Core Asyncio Concepts

### Event Loop
- The **scheduler** that runs async code
- Continuously loops and decides which coroutine to execute next
- Switches between coroutines when one is waiting
- Created and managed by `asyncio.run()`

Think of it as a coordinator juggling multiple tasks:
```
Event Loop:
  ├─ Task A: waiting on I/O
  ├─ Task B: ready to run → executes
  ├─ Task C: waiting on I/O
  └─ [Back to Task A: I/O done → executes]
```

### Coroutine
- A function defined with `async def`
- When called, returns a **coroutine object** (not the result)
- Must be `await`ed or passed to functions like `gather()` to execute
- Example:
  ```python
  coro = task_a()  # Returns coroutine object, doesn't execute
  await coro       # Actually executes
  ```

### Future
- A **placeholder** for a result that will be available later
- Lower-level primitive; coroutines are built on top of Futures
- Usually you don't interact with Futures directly; they're internal
- When you `await` something, internally it's waiting on a Future

---

## 4. How the Event Loop Achieves Concurrency

### Single-Threaded Concurrency through Task Switching

Python's event loop is **single-threaded**, but achieves concurrency through **cooperative multitasking**:

#### With Non-Blocking I/O (`asyncio.sleep()`):
```
Timeline:
0.0s: Task A starts → await asyncio.sleep(0.1) → yields control ✓
0.0s: Task B starts → await asyncio.sleep(0.1) → yields control ✓
      Event loop waits for both...
0.1s: Both tasks wake up → Resume and complete
Total: ~0.1s (CONCURRENT)
```

#### With Blocking I/O (`time.sleep()`):
```
Timeline:
0.0s: Task A starts → time.sleep(0.1) → BLOCKS event loop ✗
0.1s: Task A done
0.1s: Task B starts → time.sleep(0.1) → BLOCKS event loop ✗
0.2s: Task B done
Total: 0.2s (SEQUENTIAL)
```

**The difference:**
- `asyncio.sleep(0.1)`: "I'm waiting, you handle other tasks"
- `time.sleep(0.1)`: "I'm blocking you, you can't do anything"

### Real-World Analogy
Imagine a cashier:
- **Async approach:** "Customer A, wait over there. Let me help Customer B while you wait." (concurrent)
- **Blocking approach:** "Customer A, stand right here. I need to sleep and you're stuck." (sequential)

---

## 5. When Does the Event Loop Switch Tasks?

**Important Rule:** The event loop can **ONLY** switch at `await` points.

The event loop switches when:
1. A coroutine encounters an `await` that yields control
2. That awaited operation completes and another task becomes ready

The event loop **CANNOT** switch:
- In the middle of regular Python code execution
- During CPU-bound work without `await`

### Example:
```python
async def task_a():
    await asyncio.sleep(0)  # ← Event loop CAN switch here

async def task_b():
    for i in range(1000000):  # ← Event loop is STUCK here
        x = i * i              # No await = no switch possible
```

**Key Takeaway:** Tasks **voluntarily** yield control via `await`. If you have CPU-bound work without any `await`, the event loop is blocked.

To allow switching during long computations, insert `await asyncio.sleep(0)`:
```python
async def task_with_cpu_work():
    for i in range(1000000):
        x = i * i
        if i % 10000 == 0:
            await asyncio.sleep(0)  # Let other tasks run
```

---

## 6. Practical Example

```python
import time
import asyncio

async def task_a():
    await asyncio.sleep(0.1)
    print("Task A done")

async def task_b():
    await asyncio.sleep(0.1)
    print("Task B done")

async def main():
    start = time.time()

    # Sequential: 0.2s
    # await task_a()
    # await task_b()

    # Concurrent: 0.1s
    await asyncio.gather(task_a(), task_b())

    print(f"Duration: {time.time() - start:.4f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Connection to Streaming

Streaming is about handling **continuous, non-blocking I/O**:

- **ChatGPT streaming:** Receive tokens one at a time without blocking the client
- **Data streaming:** Process data from multiple sources concurrently
- **Server streaming:** Handle multiple client connections simultaneously

Async/await patterns are essential because:
1. **Non-blocking:** Don't block on I/O (network calls, file reads, etc.)
2. **Concurrent:** Handle multiple streams simultaneously with a single thread
3. **Efficient:** Use minimal resources compared to multi-threading

Example: Streaming tokens from an API without blocking the UI:
```python
async def stream_tokens(prompt):
    async for token in api.stream(prompt):  # Non-blocking iteration
        yield token  # Emit token immediately
        await asyncio.sleep(0)  # Allow other tasks to run
```

---

## 8. Generators: The Foundation of Streaming

### What is a Generator?

A **generator** is a function that produces a sequence of values **lazily** using the `yield` keyword. Instead of computing and returning all values at once, it produces them one at a time.

```python
# Regular function: Computes all values upfront
def regular_function():
    result = []
    for i in range(5):
        result.append(i * 2)
    return result  # Returns entire list at once

# Generator: Produces values on-demand
def generator_function():
    for i in range(5):
        yield i * 2  # Pauses here, returns one value at a time
```

### Key Characteristics of Generators

**1. Lazy Evaluation**
- Values are NOT computed until requested
- Generator object is created immediately, but no computation happens yet

```python
gen = generator_function()  # Nothing happens yet
print(type(gen))           # <class 'generator'>

for value in gen:
    print(value)           # Now computation happens, one value at a time
```

**2. Memory Efficient**
- Only one value in memory at a time
- Ideal for large datasets that don't fit in memory

```python
# Regular: Loads all 1 million items into memory
def load_all_data():
    data = []
    for i in range(1000000):
        data.append(i)
    return data  # ~40MB+ memory

# Generator: Constant memory footprint
def stream_data():
    for i in range(1000000):
        yield i  # Only one item in memory at a time
```

**3. Stateful**
- Generators maintain their state between yields
- Can resume from where they left off

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

gen = countdown(3)
print(next(gen))  # 3
print(next(gen))  # 2
print(next(gen))  # 1
# StopIteration raised after this
```

### Generator Expressions

Just like list comprehensions, there are **generator expressions** using parentheses:

```python
# List comprehension (eager - loads all into memory)
squares_list = [x**2 for x in range(1000000)]  # ~40MB

# Generator expression (lazy - on-demand)
squares_gen = (x**2 for x in range(1000000))   # Minimal memory
```

---

## 9. Async Generators: Streaming with Concurrency

### The Power of Async Generators

An **async generator** combines generators with async/await. It produces values asynchronously, allowing concurrent I/O while streaming:

```python
async def async_stream():
    """
    This is what ChatGPT streaming uses!
    Produces values asynchronously without blocking.
    """
    tokens = ["Hello", "from", "async", "generator"]
    for token in tokens:
        await asyncio.sleep(0.1)  # Simulate API delay (non-blocking)
        yield token

# Consume with 'async for'
async def main():
    async for token in async_stream():
        print(token)  # Prints one token at a time
```

### Why Async Generators are Essential for Streaming

| Aspect | Sync Generator | Async Generator |
|--------|---|---|
| **Declaration** | `def` with `yield` | `async def` with `yield` |
| **Consumption** | `for` loop | `async for` loop |
| **I/O Operations** | Blocks on I/O | Non-blocking I/O |
| **Concurrent Work** | Can't do other work while waiting | Can do other work while waiting |
| **Use Case** | In-memory data streaming | Network/I/O streaming (ChatGPT, APIs) |

### Example: Non-Blocking Token Streaming

```python
async def chatgpt_streaming(prompt):
    """Simulates ChatGPT token streaming"""
    # This is essentially what happens with streaming:
    # 1. Make API call (non-blocking)
    # 2. Receive tokens one by one
    # 3. Yield each token immediately (don't wait for full response)

    tokens = ["The", "answer", "to", "life", "is", "42"]
    for token in tokens:
        await asyncio.sleep(0.1)  # Simulate network delay
        yield token

async def main():
    # Start consuming tokens
    async for token in chatgpt_streaming("What is the answer?"):
        print(token, end=" ", flush=True)
        # UI updates with each token immediately
        # No need to wait for entire response!
```

**Why this matters:**
- **Without streaming:** Wait for entire response (0.6s), then display
- **With async generator:** Display first token at 0.1s, second at 0.2s, etc.
- **User experience:** Feels fast and responsive, even though same total time

---

## 10. Building Data Pipelines with Generators

Generators are perfect for building composable data pipelines:

```python
def read_file(filename):
    """Generator: Read file line by line"""
    with open(filename) as f:
        for line in f:
            yield line.strip()

def filter_numbers(lines):
    """Generator: Filter lines containing numbers"""
    for line in lines:
        if any(c.isdigit() for c in line):
            yield line

def transform(lines):
    """Generator: Transform each line"""
    for line in lines:
        yield line.upper()

# Pipeline: data flows through generators
pipeline = transform(filter_numbers(read_file("data.txt")))
for processed_line in pipeline:
    print(processed_line)
```

**Benefits:**
- Each step processes one item at a time
- Memory efficient (no intermediate lists)
- Easy to extend and compose
- Lazy evaluation (only processes items that are consumed)

---

## 11. Generators vs Async Generators in Streaming

**When to use sync generators:**
- Processing data already in memory
- Reading local files in chunks
- Building data transformation pipelines

**When to use async generators:**
- Streaming from network APIs (ChatGPT, etc.)
- Handling multiple concurrent I/O sources
- Real-time data feeds
- Websocket streaming
- Anything that involves waiting for I/O

### Real-World Example: ChatGPT Streaming

```python
async def stream_chatgpt_response(prompt):
    """
    This mimics how ChatGPT streaming works:
    1. Send prompt to API (non-blocking await)
    2. Receive response as a stream of tokens
    3. Yield each token as it arrives
    4. Client can display tokens in real-time
    """
    # In reality, this is an HTTP request with streaming response
    response_stream = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    async for chunk in response_stream:
        token = chunk.choices[0].delta.get("content", "")
        if token:
            yield token

# Usage
async def display_response(prompt):
    async for token in stream_chatgpt_response(prompt):
        print(token, end="", flush=True)  # Display immediately
```

---

## Summary: Generators and Streaming

**Generators enable streaming because they:**
1. **Produce values lazily** - Don't compute everything upfront
2. **Are memory efficient** - Only one item in memory at a time
3. **Support composition** - Build pipelines by chaining generators
4. **Work with async/await** - Async generators enable non-blocking streaming

**The streaming pattern:**
- Request → Yield first token → Yield second token → ... → Done
- Each token displayed immediately instead of waiting for complete response
- User feels like the system is responsive and fast

---

## Next Steps
- Run the `generators.py` file to see all these concepts in action
- Explore how to build a ChatGPT clone with async generator streaming
- Learn about backpressure and flow control in streaming systems
- Understand how FastAPI and other frameworks use async generators for streaming responses
