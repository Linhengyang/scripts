
import os
import itertools
from concurrent.futures import ProcessPoolExecutor

def process_fn(i):
    print(f'执行 {i} in PID {os.getpid()}')
    return f'返回 {i} in PID {os.getpid()}'


futures = []
with ProcessPoolExecutor(3) as e:
    for i in itertools.batched(range(10), 3):
        future = e.submit(process_fn, i)
        futures.append(future)

# 按提交顺序收集
for future in futures:
    print(future.result())