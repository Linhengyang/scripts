# customized design pattern for async process

import asyncio
import typing as t

# 异步的生产者-消费这模式，是通过一个queue连接，但解耦了 生产数据 和 消费数据 两个步骤之间的依赖关系，使得
# 生产和消费两个异步task可以自顾自运行

# 同步的 generator --> 异步put的queue
async def wrap_sync_as_async(gen:t.Generator):
    for item in gen:
        yield item

async def async_queue_get(yield_generator:t.Generator, queue:asyncio.Queue):
    '''
    异步读取 generator 的导出结果, 并放入 queue
    '''
    async for product in wrap_sync_as_async(yield_generator):
        await queue.put(product) # 异步地放入 queue: 当 queue 满的时候，要等待空间
    
    await queue.put(None) # 生产结束的信号



# 单消费者: queue 中的 output 全部由一个消费任务消费
async def async_queue_process(
        queue:asyncio.Queue,
        executor,
        process_fc:t.Callable,
        collector:t.Callable|None=None,
        *args):
    '''
    异步处理 queue 中的导出结果.
    处理函数 process_fc 是同步的, 且可并行. 它的第一个位置参数必须是队列中的 product
    executor 是线程池concurrent.futures.ThreadPoolExecutor/进程池concurrent.futures.ProcessPoolExecutor

    `result = await loop.run_in_executor(executor, process_fc, product, *args)` 
    对于进程池 executor, 上面代码有两次跨进程通信, 分别是
        1. product从主线程pickle后拷贝到子进程executor
        2. result从子进程pickle后拷贝到主线程
    所以要求 1. process_fc必须是可pickle的; 2. product必须是可pickle的; 3. result必须是可pickle的
    
    考虑到collector只在主线程里执行, 而主线程是顺序的, collector可以是普通的容器(但必须要包一层异步以适配await)
    '''
    loop = asyncio.get_running_loop() # 在异步协程内部获取事件循环.
    # 事件循环由 asyncio.run自动创建关闭, 不再使用 get_event_loop

    while True:
        product = await queue.get()

        if product is None: # 收到结束信号
            await queue.put(None) # 把结束信号放回去，以通知多个消费者
            break
        # 这里 collector 也可以用一个asyncio.Queue来接收. 这是因为
        # 主线程 await 得到子进程 process_fc(product, *args) 的result, loop会执行一次跨进程拷贝
        result = await loop.run_in_executor(executor, process_fc, product, *args)

        # 以一个in-place状态改变的方式，在主线程异步聚合异步消费的结果
        if collector:
            await collector(result)
    



# 主协程
async def pipeline_producer_consumer(
        producer:t.Generator,
        process_fc:t.Callable,
        executor,
        max_queue_size:int,
        num_consumers:int,
        collector:t.Callable|None,
        *args):
    # 创建队列
    queue = asyncio.Queue(max_queue_size)

    producer_task = asyncio.create_task(async_queue_get(producer, queue))
    
    consumer_tasks = [
        asyncio.create_task(async_queue_process(queue, executor, process_fc, collector, *args))
        for _ in range(num_consumers)
    ]
    
    await asyncio.gather(producer_task, *consumer_tasks)

    if collector:
        await collector(None)


# 以上三个公版异步函数, 只有 process_fc 在进程池/线程池, read到queue, await拿到process_fc的结果,
# 以及收集结果 result 到 collector, 都是在主线程执行的.

# 若要在 executor 中执行 collector, 那么需要根据线程池/进程池的区别, 有不同的线程安全/内存贡献设计
# 若executor是线程池, 那么collector的设计如下:
# shared_container = []
# lock = asyncio.Lock()
# async def collector(result):
#     async with lock: // 锁保证线程安全
#         shared_container.append(result)

# 若executor是进程池, 那么collector的设计如下:
# from multiprocessing import Manager, get_context
# manager = get_context("spawn").Manager() 或者是 Manager()
# shared_container = manager.list() // 由一个进程管理的跨进程共享
# async def collector(result):
#     shared_container.append(result)