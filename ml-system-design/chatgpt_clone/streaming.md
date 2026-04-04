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

## Next Steps
- Understand how streaming APIs use async generators
- Explore how to build a ChatGPT clone with streaming responses
- Learn about backpressure and flow control in streaming systems
