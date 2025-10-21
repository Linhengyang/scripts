# 自定义 optimizer: 自定义优化器必须继承自 torch.optim.Optimizer, 并正确实现 __init__ 和 step 方法,
# 才能保证优化器与 pytorch 的训练流程比如 .zero_grad() / .step() / state save or load / 正确协作

# __init__规范:
# 1. 必须调用父类的 __init__:  super().__init__(params, defaults)
# 这里 params 必须是:
#   1. 一个 param tensor 的可迭代对象, 比如 model.parameters()
#   2. 一个字典列表, 每个字典包含 'params' key 和其他超参数 key, 比如
#      [{'params': model.base.parameters(), 'lr': 1e-4}, {'params': model.fc.parameters(), 'lr': 1e-3}]
#   总结: params 必须是参数tensor的可迭代对象, 或者 参数tensor nested with other hyper-params 的可迭代对象
# 这里 defaults 必须是 字典, 包含该优化器的默认超参数, 比如 lr / weight_decay / betas 等

# 2. 必须对所有超参数作 合法性检查, 抛出 value error 而不是 assert

# 3. 对于有状态的优化器(状态比如动量等), 不要在 __init__ 中初始化状态, 而是在 step 中 lazy 初始化


# step规范:
# 1. 必须使用 torch.no_grad() 装饰: 防止在 grad 计算中构建计算图
# 2. 支持可选的 closure 参数: closure 应该是一个 None or 无参的返回 loss 的函数. 当执行 closure not None 的 step 时,
#    应该打开 torch.enable_grad(), 使用 closure 返回 带梯度的 loss(以用于 L-BFGS 等二阶优化器):
#    if closure is not None:
#       with torch.enable_grad():
#           loss = closure()
#    大部分情况下一阶优化器即可, 所以保留该 closure 接口即可
# 3. 更新 grad tensor 时, 遍历 self.param_groups, 即 __init__ 里的 参数 params 统一形式为 list of params_group, each group is a dict
#    for group in self.param_groups:
#       for p in group['params']:
#           if p.grad is not None:
                # ... to update p
# 4. 使用 self.state[p] 来存储每个 p 的状态(比如动量/历史梯度等):
#    self.state 是特殊管理的 defaultdict 类, 可以以 tensor p 为键(实质是以 参数p 的身份为键, 参数p在模型生命周期内不会被改变, 不依赖 p.data 和 hash(p))
#    lazy initialization: 首次遇到参数 p 才创建参数 p 的 state buffer. state buffer 应该与 p 同设备同dtype
#    self.state[p]['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#    zeros_like 自动保证 zero 的 dtype 看齐 p, memory_format 参数

# 5. 对 p 的操作, in-place 优先, 避免额外内存分配, 比如:
#    p.add_(grad, alpha=-lr) 而不是 p = p - lr * grad

# 6. 一般情况下都返回 None(closure = None). 但是当 closure not None 时, 返回调用 closure 得到的 loss




import torch
import torch.nn as nn
from torch.optim import Optimizer



# basic SGD optimizer
# vanilla SGD: velocity = grad, weight <- weight - lr * velocity
# SGD with momentum and weight_decay:
# 1. grad-modification: grad_ = grad + wd * weight (梯度grad本身不改变, 但参与更新weight的grad_修正一个weight比例值)
#    (note: SGD 通过改梯度值的方式来decay weight, AdamW是直接缩放weight. 两种方式对于SGD是等价的)
# 2. velocity-update: velocity <- momentum * velocity + grad (历史velocity折扣后加上当前梯度grad, 复合出一个更新方向)
# 3. weight-update: weight <- weight - lr * velocity (weight在复合的velocity方向更新 lr)

class MySGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0):
        if lr < 0: raise ValueError(f'learning rate must be positive. now {lr}')
        defaults = {'lr':lr, 'momentum':momentum, 'weight_decay':weight_decay}
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    if group['weight_decay'] != 0: # 如果有 weight_decay, SGD 的做法是加到 梯度上
                        # 不希望污染 梯度自身, 因为可能其他地方还要用梯度, 所以用 .add 方法   .add 会分配新内存返回新的对象, 不改变 grad 和 p
                        grad = grad.add(p, alpha=group['weight_decay']) # grad_ = grad + wd * weight
                    
                    state = self.state[p] # 拿出 p 对应的状态
                    if 'momentum_buffer' not in state: # 延迟初始化: 当要用时检查发现尚未初始化, 那么初始化
                        state['momentum_buffer'] = torch.zeros_like(p) # 延迟初始化为 zero
                    velocity = state['momentum_buffer']
                    velocity.mul_(group['momentum']).add_(grad) # 状态 是要保存的, 所以用 in-place 操作避免新内存分配, 节省内存
                    p.add_(velocity, alpha=-group['lr']) # 修改 p 的值是优化器的职责, 所以用 in-place 操作
        
        return loss
    


class MyAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if lr < 0: raise ValueError(f'learning rate must be positive. now {lr}')
        if betas[0] < 0 or betas[0] > 1: raise ValueError(f'beta1 must between 0-1. now {betas[0]}')
        if betas[1] < 0 or betas[1] > 1: raise ValueError(f'beta2 must between 0-1. now {betas[1]}')
        if eps <= 0: raise ValueError(f'eps must be positive. now {eps}')
        if weight_decay < 0: raise ValueError(f'weight_decay must 0 or positive. now {weight_decay}')

        defaults = {'lr':lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay, 'amsgrad':amsgrad}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            eps = group['eps']
            amsgrad = group['amsgrad']

            for p in group['params']: # 取出权重 weight as p
                if p.grad is None:
                    continue

                grad = p.grad
                # 原始 Adam 的 WD 是和梯度耦合的: weight_decay 是加在 梯度上的: 先用 WD 修改梯度, 再用修改后的 梯度去更新 velocity 和 梯度方差
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # Adam中, 涉及三个状态值: velocity(momentum_buffer) & 梯度方差(variance_buffer) & 优化迭代次数(step)
                # if amsgrad = True, 还需要一个额外状态值: (历史最大梯度方差)max_variance_buffer
                state = self.state[p]
                # lazy initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['variance_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['step'] = 0
                    if amsgrad:
                        state['max_variance_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # update step
                state['step'] += 1
                step = state['step']

                # update velocity: velocity <- b1 * velocit + (1-b1)*grad
                velocity = state['momentum_buffer']
                velocity.mul_(beta1).add_(grad, alpha=1-beta1)

                # update variance: variance <- b2 * variance + (1-b2)*grad^2
                variance = state['variance_buffer']
                variance.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # update weights: p <- p - lr * velocity_ / (variance_.sqrt() + eps)
                # if amsgrad=True: p <- p - lr * velocity_ / (max_variance_.sqrt() + eps)
                #   here: velocity_ <- velocity/(1-beta1^t), variance_ <- variance/(1-beta2^t)
                # at last have: p <- p - (lr/bias_correction1) * velocity / denom,
                #   here denom = variance.sqrt()/bias_correction2.sqrt() + eps, or max_variance.sqrt()/bias_correction2.sqrt() + eps

                bias_correction1 = 1-beta1**step
                bias_correction2 = 1-beta2**step

                step_size = lr / bias_correction1
                if amsgrad:
                    # update max_variance: only when variance > max_variance
                    max_variance = state['max_variance_buffer']
                    # 对比 max_variance 和 variance 的每一个元素, 取大者, 并将结果输出到 max_variance (即更新 max_variance)
                    torch.maximum(max_variance, variance, out=max_variance)
                    denom = (max_variance.sqrt() / (bias_correction2**0.5)).add_(eps)
                else:
                    denom = (variance.sqrt() / (bias_correction2**0.5)).add_(eps)
                
                # p <- p - step_size * velocity / denom
                p.addcdiv_(velocity, denom, value=-step_size)
    
        return loss
    



# AdamW 和 Adam 的区别在于: AdamW 的 weight_decay 与 grad 解耦, 即
#   AdamW 在用 真实 grad 更新完 weight 之后, 再按比例衰减 weight: weight <- weight * (1 - lr*wd)
#   Adam 是先调整 grad: grad_ = grad + wd*weight, 以达到更新时衰减的效果: weight <- weight - lr * grad_ = (1 - lr*wd)weight - lr*grad
# Adam 不好的地方在于: grad_ 而不是 grad 参与了所有 velocity / variance 的计算. 含有weight衰减量的 grad_ 在计算 variance 时, 
# 会贡献使得 variance 很大, 即 denom 很大, 从而 p 的更新 p <- p - step_size * velocity / denom 很慢: 即 p 的 decay 被延缓了, 衰减速率被下降
# 也就是说, 预设定的 weight_decay 被打折扣了.   ----> 不行, 所以解耦 decay 和 grad: 直接按比例衰减 weight 自身, 而不是加到 grad 里.

# AdamW 已经替代了 Adam, 成为了标准做法!

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if lr < 0: raise ValueError(f'learning rate must be positive. now {lr}')
        if betas[0] < 0 or betas[0] > 1: raise ValueError(f'beta1 must between 0-1. now {betas[0]}')
        if betas[1] < 0 or betas[1] > 1: raise ValueError(f'beta2 must between 0-1. now {betas[1]}')
        if eps <= 0: raise ValueError(f'eps must be positive. now {eps}')
        if weight_decay < 0: raise ValueError(f'weight_decay must 0 or positive. now {weight_decay}')

        defaults = {'lr':lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay, 'amsgrad':amsgrad}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            eps = group['eps']
            amsgrad = group['amsgrad']

            for p in group['params']: # 取出权重 weight as p
                if p.grad is None:
                    continue

                grad = p.grad
                # AdamW 的 WD 和梯度解耦

                # Adam中, 涉及三个状态值: velocity(momentum_buffer) & 梯度方差(variance_buffer) & 优化迭代次数(step)
                # if amsgrad = True, 还需要一个额外状态值: (历史最大梯度方差)max_variance_buffer
                state = self.state[p]
                # lazy initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['variance_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['step'] = 0
                    if amsgrad:
                        state['max_variance_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # update step
                state['step'] += 1
                step = state['step']

                # update velocity: velocity <- b1 * velocit + (1-b1)*grad
                velocity = state['momentum_buffer']
                velocity.mul_(beta1).add_(grad, alpha=1-beta1)

                # update variance: variance <- b2 * variance + (1-b2)*grad^2
                variance = state['variance_buffer']
                variance.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # update weights: p <- p - lr * velocity_ / (variance_.sqrt() + eps)
                # if amsgrad=True: p <- p - lr * velocity_ / (max_variance_.sqrt() + eps)
                #   here: velocity_ <- velocity/(1-beta1^t), variance_ <- variance/(1-beta2^t)
                # at last have: p <- p - (lr/bias_correction1) * velocity / denom,
                #   here denom = variance.sqrt()/bias_correction2.sqrt() + eps, or max_variance.sqrt()/bias_correction2.sqrt() + eps

                bias_correction1 = 1-beta1**step
                bias_correction2 = 1-beta2**step

                step_size = lr / bias_correction1
                if amsgrad:
                    # update max_variance: only when variance > max_variance
                    max_variance = state['max_variance_buffer']
                    # 对比 max_variance 和 variance 的每一个元素, 取大者, 并将结果输出到 max_variance (即更新 max_variance)
                    torch.maximum(max_variance, variance, out=max_variance)
                    denom = (max_variance.sqrt() / (bias_correction2**0.5)).add_(eps)
                else:
                    denom = (variance.sqrt() / (bias_correction2**0.5)).add_(eps)
                
                # p <- p - step_size * velocity / denom
                p.addcdiv_(velocity, denom, value=-step_size)

                # AdamW 的 WD 和梯度解耦
                if wd != 0:
                    p.mul_(1 - lr*wd)
        
        return loss
    




if __name__ == "__main__":
    model1 = nn.Linear(10, 1)
    nn.init.zeros_(model1.weight)
    nn.init.zeros_(model1.bias)
    Opt1 = AdamW(model1.parameters(), lr=1e-3, weight_decay=1e-4)
    Opt1.zero_grad()

    loss_fn = nn.MSELoss()
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    model2 = nn.Linear(10, 1)
    nn.init.zeros_(model2.weight)
    nn.init.zeros_(model2.bias)
    Opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-4)
    Opt2.zero_grad()

    l1 = loss_fn(model1(x), y)
    l2 = loss_fn(model2(x), y)
    l1.backward()
    l2.backward()

    Opt1.step()
    Opt2.step()

    print(model1.weight, model1.bias)
    print('===========================')
    print(model2.weight, model2.bias)