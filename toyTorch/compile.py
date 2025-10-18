# torch compile
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

# sequential
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



device = torch.device("cuda")
x = torch.randn(size=(1, 1, 512), device=device)


net = get_toyNet()
net.to(device)

net.train()
print( f'net(x) at first: {net(x)}' )




# eager mode
with benchmark('eager: warm up'):
    for i in range(5):
        net(x)

with benchmark('eager: 100 loop'):
    for i in range(100):
        net(x)












# compile
with benchmark('compile:'):
    compile_net = torch.compile(net)

print( f'compile_net(x) at first: {compile_net(x)}' )

print( f'type of compile net: {type(compile_net)}' )


with benchmark('compile: warm up'):
    for i in range(5):
        compile_net(x)

with benchmark('compile: 100 loop'):
    loss_fn = nn.CrossEntropyLoss()
    for i in range(100):
        y_hat = compile_net(x) # (1, 1, 2)
        y_hat = y_hat.permute(0,2,1) # (1, 2, 1)
        y = torch.randint(0, 2, (1,1), device=device) # (1, 1)
        l = loss_fn(y_hat, y)

        l.sum().backward()



print( f'net(x) after train: {net(x)}' )
print( f'compile_net(x) after train: {compile_net(x)}' )

# net.state_dict()