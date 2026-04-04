import time
import asyncio

async def regular():
    # add time
    start = time.time()
    print(f"Start time: {start}")
    await asyncio.gather(task_a(), task_b(), task_c())
    end = time.time()
    duration = end - start
    print(f"End time: {end}")
    print(f"Duration: {duration:.4f} seconds")

    start = end
    print(f"Start time: {start}")
    await asyncio.gather(task_b(), task_c(), task_a())
    end = time.time()
    duration = end - start
    print(f"End time: {end}")
    print(f"Duration: {duration:.4f} seconds")

    start = end
    print(f"Start time: {start}")
    await asyncio.gather(task_c(), task_a(), task_b())
    end = time.time()
    duration = end - start
    print(f"End time: {end}")
    print(f"Duration: {duration:.4f} seconds")

async def task_a():
    await asyncio.sleep(0.1)

async def task_b():
    await asyncio.sleep(0.2)

async def task_c():
    time.sleep(0.25)
    # await asyncio.sleep(0.25)

if __name__ == "__main__":
    asyncio.run(regular())