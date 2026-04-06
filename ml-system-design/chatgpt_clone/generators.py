"""
Understanding Generators and Streaming

Generators are functions that produce values lazily using 'yield'.
They are the foundation of efficient streaming systems.
"""

# ============================================================================
# 1. REGULAR FUNCTION vs GENERATOR
# ============================================================================

def regular_function():
    """Regular function: Compute and return all values at once"""
    result = []
    for i in range(5):
        result.append(i * 2)
    return result  # Returns entire list at once


def generator_function():
    """Generator function: Produce values one at a time"""
    for i in range(5):
        yield i * 2  # Pauses here, returns one value


# ============================================================================
# 2. LAZY EVALUATION
# ============================================================================

def demo_lazy_evaluation():
    """
    Generators don't compute values until you ask for them.
    This is called "lazy evaluation".
    """
    print("\n=== REGULAR FUNCTION (EAGER) ===")
    result = regular_function()
    print(f"Function returned: {result}")
    print(f"Type: {type(result)}")  # list

    print("\n=== GENERATOR FUNCTION (LAZY) ===")
    gen = generator_function()
    print(f"Generator created: {gen}")
    print(f"Type: {type(gen)}")  # generator object

    # Values are produced one at a time
    print("\nIterating through generator:")
    for value in gen:
        print(f"  Received: {value}")


# ============================================================================
# 3. MEMORY EFFICIENCY
# ============================================================================

def naive_large_data():
    """Regular function: Loads entire dataset into memory"""
    data = []
    for i in range(1000000):
        data.append(i)  # Stores all 1 million items in memory
    return data


def streaming_large_data():
    """Generator: Produces data on-demand, constant memory"""
    for i in range(1000000):
        yield i  # Only one item in memory at a time


def demo_memory_efficiency():
    print("\n=== MEMORY COMPARISON ===")

    # Regular function: loads everything
    print("Regular function: Loading 1M items...")
    # result = naive_large_data()  # Would use ~40MB+ of memory

    # Generator: stream on-demand
    print("Generator: Creating stream...")
    gen = streaming_large_data()  # Uses negligible memory

    print("Processing first 5 items:")
    for i, value in enumerate(gen):
        if i >= 5:
            break
        print(f"  Item {i}: {value}")
    # Rest of data is never loaded into memory!


# ============================================================================
# 4. STATEFUL ITERATION
# ============================================================================

def countdown_generator(n):
    """Generators maintain state between yields"""
    print(f"[GENERATOR] Starting countdown from {n}")
    while n > 0:
        yield n
        n -= 1
    print(f"[GENERATOR] Countdown complete")


def demo_stateful_iteration():
    print("\n=== STATEFUL ITERATION ===")
    gen = countdown_generator(3)

    print("Getting first value:")
    print(f"  {next(gen)}")

    print("Getting second value:")
    print(f"  {next(gen)}")

    print("Getting third value:")
    print(f"  {next(gen)}")

    print("Getting fourth value (raises StopIteration):")
    try:
        print(f"  {next(gen)}")
    except StopIteration:
        print("  StopIteration raised - generator exhausted")


# ============================================================================
# 5. GENERATOR EXPRESSIONS
# ============================================================================

def demo_generator_expressions():
    print("\n=== GENERATOR EXPRESSIONS ===")

    # List comprehension (eager - loads all into memory)
    squares_list = [x**2 for x in range(5)]
    print(f"List: {squares_list}")
    print(f"Type: {type(squares_list)}")

    # Generator expression (lazy - produces on-demand)
    squares_gen = (x**2 for x in range(5))
    print(f"\nGenerator: {squares_gen}")
    print(f"Type: {type(squares_gen)}")

    print("Consuming generator:")
    for sq in squares_gen:
        print(f"  {sq}")


# ============================================================================
# 6. STREAMING ANALOGY
# ============================================================================

def stream_api_response():
    """
    Simulates an API that streams tokens like ChatGPT.
    In reality, the API sends tokens over the network.
    Here, we simulate it with a generator.
    """
    tokens = ["The", "answer", "is", "42"]
    for token in tokens:
        yield token  # Send one token at a time


def demo_streaming_use_case():
    print("\n=== STREAMING API RESPONSE ===")
    print("Receiving response from API:")

    for token in stream_api_response():
        print(f"  Token: '{token}'")
        # In a real scenario, you'd display this to the user immediately
        # without waiting for the entire response


# ============================================================================
# 7. ASYNC GENERATORS (Essential for Streaming)
# ============================================================================

import asyncio


async def async_stream_api_response():
    """
    Async generator: Produces values asynchronously.
    This is what ChatGPT streaming uses!
    """
    tokens = ["Hello", "from", "async", "generator"]
    for token in tokens:
        await asyncio.sleep(0.1)  # Simulate API delay
        yield token


async def demo_async_generators():
    print("\n=== ASYNC GENERATORS ===")
    print("Receiving async response:")

    async for token in async_stream_api_response():
        print(f"  Token: '{token}'")


# ============================================================================
# 8. GENERATOR vs ASYNC GENERATOR - STREAMING COMPARISON
# ============================================================================

async def demo_streaming_vs_async():
    print("\n=== SYNC vs ASYNC GENERATORS ===")

    print("\nSync Generator (blocks on I/O):")
    print("  - Can't do other work while waiting")
    print("  - Good for in-memory streaming")

    print("\nAsync Generator (non-blocking):")
    print("  - Can do other work while waiting for data")
    print("  - Essential for I/O-bound streaming (network, files, etc.)")
    print("  - Use with 'async for' to consume")


# ============================================================================
# 9. PRACTICAL EXAMPLE: Building a Data Pipeline
# ============================================================================

def read_large_file(filename, chunk_size=10):
    """Generator: Read file in chunks without loading everything"""
    with open(filename, 'w') as f:
        # Create a sample file
        for i in range(100):
            f.write(f"line {i}\n")

    # Now read it back in chunks
    with open(filename, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line.strip())
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def process_numbers(data_stream):
    """Generator: Process data from another generator"""
    for chunk in data_stream:
        # Process each chunk
        processed = [int(x.split()[1]) * 2 for x in chunk]
        yield processed


def demo_pipeline():
    print("\n=== GENERATOR PIPELINE ===")
    # Create a pipeline of generators
    # Each generator feeds into the next

    filename = "/tmp/sample.txt"
    pipeline = process_numbers(read_large_file(filename))

    print("Processing data through pipeline:")
    for i, result in enumerate(pipeline):
        print(f"  Batch {i}: {result}")
        if i >= 2:  # Just show first few
            break


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo_lazy_evaluation()
    demo_memory_efficiency()
    demo_stateful_iteration()
    demo_generator_expressions()
    demo_streaming_use_case()

    # Async demo
    print("\n" + "="*70)
    asyncio.run(demo_async_generators())
    asyncio.run(demo_streaming_vs_async())

    demo_pipeline()

    print("\n" + "="*70)
    print("\nKey Takeaways:")
    print("1. Generators produce values lazily with 'yield'")
    print("2. Memory efficient - only one value in memory at a time")
    print("3. Stateful - maintain position between yields")
    print("4. Async generators enable non-blocking streaming")
    print("5. Perfect for building data pipelines and streaming APIs")
