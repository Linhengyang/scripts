
import os
import itertools
from concurrent.futures import ProcessPoolExecutor
import time
import random

def process_fn(i):
    time.sleep(random.random())
    print(f'执行 {i} in PID {os.getpid()}')
    return f'返回 {i} in PID {os.getpid()}'


futures = []
with ProcessPoolExecutor(3) as e:
    for i in itertools.batched(range(10), 3):
        future = e.submit(process_fn, i)
        futures.append(future)

# 按提交顺序收集
# for future in futures:
#     print(future.result())


# 按完成顺序收集
from concurrent.futures import as_completed
for future in as_completed(futures):
    print(future.result())

# 执行 (6, 7, 8) in PID 16211
# 执行 (0, 1, 2) in PID 16209
# 执行 (9,) in PID 16211
# 执行 (3, 4, 5) in PID 16210
# 返回 (9,) in PID 16211
# 返回 (6, 7, 8) in PID 16211
# 返回 (3, 4, 5) in PID 16210
# 返回 (0, 1, 2) in PID 16209