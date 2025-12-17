import os

print(f'Process {os.getpid()} start...')


pid = os.fork() # 从这里开始, os 创建一个 新的进程(同时也创建了一个新的Python解释器)
# 从此处开始的代码, 将由两个 解释器分别执行一遍
if pid == 0:
    print( f'I am child process {os.getpid()} and my parent is {os.getppid()}.' )
else:
    print( f'I {os.getpid()} just created a child process {pid}.' )

print('endline') # 可以看到这行会被两个进程分别执行一次