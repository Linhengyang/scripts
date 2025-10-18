# torch script
import time
import numpy as np
import torch
import torch.nn as nn


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class benchmark:
    def __init__(self, desciption='Done'):
        self.description = desciption

    def __enter__(self):
        self.timer = Timer()
        return self
    
    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')



# toy network
def get_toyNet():
    class toynet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layer1 = nn.Linear(512, 256)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(256, 128)
            self.layer3 = nn.Linear(128, 2)

        def forward(self, X):
            # X shape: (batch_size, num_seq, 512)
            
            # test dynamic control flow
            if X.sum() > 0:
                s = -torch.ones_like(X, device=X.device)
            else:
                s = torch.ones_like(X, device=X.device)

            Y = self.layer1(X + s)

            for i in range(5):
                X = X * 1/2
            

            if torch.randn(1) > 0:
                X = X[:, :, :256]
            else:
                X = X[:, :, 256:]

            Z = self.layer2(Y + X)
            Z = self.relu(Z)

            while Z.sum() > 0:
                Z = Z - 1/16
            
            return self.layer3(Z)
            
    return toynet()

x = torch.randn(size=(1, 2, 512))



# eager mode
net = get_toyNet()

print( net(x) )




# script
script_net = torch.jit.script(get_toyNet())

print( script_net(x) )

with benchmark('without torchscript'):
    for i in range(1000):
        net(x)


with benchmark('with torchscript'):
    for i in range(1000):
        script_net(x)


# 对于 动态模型, 存储的是参数
torch.save(net.state_dict(), 'tmp/net.params')

# 对于 静态模型，存储的是模型，序列化
script_net.save('tmp/net')





# 读取 动态模型
new_net = get_toyNet()
new_net.load_state_dict( torch.load('tmp/net.params') )


print(f'loaded dynamic net output {new_net(x)}')


# 读取 静态模型
static_net = torch.jit.load('tmp/net')

print(f'loaded script net output {static_net(x)}')
