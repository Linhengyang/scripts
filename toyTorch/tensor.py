# tensor.py
import torch

def tensor_basic():
    # create tensors
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])
    x = torch.ones(4, 8) # 4 x 8 matrix of 1
    x = torch.zeros(4, 8) # 4 x 8 matrix of 0
    x = torch.randn(4, 8) # 4 x 8 matrix of iid Normal(0, 1) samples

    # allocate but don't initialize values
    x = torch.empty(4, 8) # 4 x 8 matrix of uninitialized values
    torch.nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2) # in-place value-asign with truncated Normal(0, 1) bwt [-2, 2]
    print(x)



def get_memory_usage(x: torch.Tensor):
    # .numel() 方法返回 基本元素数量
    # .element_size() 方法返回 单个基本元素 的所占 字节数量, 比如 FP32 --> 4字节, FP16 --> 2字节
    return x.numel() * x.element_size()



def precision_fp32():
    # 32 bits: 1 for sign + 8 for magnitude + 23 for mantisa
    # maximum_value = 11111110 for magnitude + 11..11 for mantisa
    # min_normal_value = 00000001 for magnitude + 00..01 for mantisa
    # 正规数区域 精度很高
    # subnormal_area: 00000000 for magnitude + mantisa to approximate 0
    # 非正规数区域 精度下降
    pass



def precision_fp16():
    # 16 bits: 1 for sign + 5 for magnitude + 10 for mantisa
    # maximum_value = 11110 for magnitude + 11..11 for mantisa = 65504
    x = torch.tensor([65505], dtype=torch.float16)
    assert x == torch.tensor([65504.], dtype=torch.float16) # 截断到 65504
    # min_normal_value = 00001 for magnitude + 00..01 for mantisa = 2⁻¹⁴ = 6.104 × 10⁻⁵
    # min_value = 00000 for magnitude + 00..01 for mantisa = 2^-24 = 5.96×10⁻⁸
    x = torch.tensor([1e-8], dtype=torch.float16)
    assert x == 0 # 下溢 underflow

    assert x.element_size() == 2



def precision_bf16():
    # 16 bits: 1 for sign + 8 for magnitude + 7 for mantisa
    # maximum_value = 11111110 for magnitude + 11..11 for mantisa = (1+1-1/2^7)*2^255 = 3.4e38
    # min_normal_value = 00000001 for magnitude + 00..01 for mantisa
    # min_value = 00000000 for magnitude + 00..01 for mantisa = 2^(-261) = 1.18e-38
    x = torch.tensor([1e-8], dtype=torch.bfloat16)
    assert x != 0 # 和 fp16 不同, 1e-8 在 bf16 不会下溢

    assert x.element_size() == 2



def get_info_floats():
    fp32_info = torch.finfo(torch.float32)
    fp16_info = torch.finfo(torch.float16)
    bf16_info = torch.finfo(torch.bfloat16)

    print(fp32_info, fp16_info, bf16_info)



def precision_fp8():
    # E4M3: 1 for sign + 4 for magnitude + 3 for mantisa
    # maximum_value in IEEE标准 = 1110 for magnitude + 111 for mantisa = (1+7/8)*2^7 = 15*2^4 = 240
    # 但实际上, 英伟达 征用了 INF 的指数 1111, 把 尾数 111 留给 INF, 从而 尾数 110 是最大值. bias 仍然是 7.
    # maximum_value in FP8 = 1111 for magnitude + 110 for mantisa = (1+3/4)*2^8 = 448
    
    # E5M2: 1 for sign + 5 for magnitude + 2 for mantisa
    # maximum_value in FP8 = 11110 for magnitude + 11 for mantisa = (1+3/4)*2^15 = 57344
    # 也就是说 E5M2 用的是标准的 IEEE 规范(即指数全1留给INF)
    pass


def tensors_on_gpu():
    # default tensors are stored in CPU memory
    x = torch.zeros(32, 32)
    assert x.device == torch.device('cpu')

    # move tensor data from CPU 内存 到 GPU 显存(HBM/DRAM) through PCIe/NVLink 等

    # 检测是否有 GPU
    if not torch.cuda.is_available():
        return
    
    # cuda device property: 一个硬件 device 的规格说明书
    # 从驱动层查询到 GPU 硬件信息, 并以python友好的方式返回
    property = torch.cuda.get_device_properties(0)

    # 返回已经占用的 显存容量
    memory_allocated = torch.cuda.memory_allocated()

    # 复制 tensor 到 cuda device 0
    y = x.to('cuda:0')

    new_memory_allocated = torch.cuda.memory_allocated()
    assert new_memory_allocated - memory_allocated == 32*32*4



# tensor.unfold(dimension, size, step) 方法
# 一个 tensor的存储: 1 meta-data, 一维连续的内存存储
# 2 shape, 定义 tensor 的维度大小, 比如 (4, 3, 2) 代表在维度 (0, 1, 2) 的大小: 
#   维度0上有4个元素, 
#   给定维度0, 每个元素在维度1上有3个子元素
#   给定维度0和维度1, 每个子元素在维度2上有2个子子元素。这里子子元素就是scalar.

# 这里有一个“给定维度0”的说法, 原因是如果不考虑维度0, 维度1总长度有4*3=12。但实际上, tensor在维度1上不存在多于3的长度, 因为多余3的长度, 是以 n*3+m 决定的
# 同样的, 不考虑维度0和1, 维度2总长度有24. 但实际上, tensor在维度2上不存在多余2的长度, 因为多余2的长度, 是以 p*6 + q*3 + k 决定的.

# 倒序: 
#   维度2有两个scalar元素的位置, 组成一个长度为2的组合元素.
#   维度1上有3个组合元素的位置, 组成一个长度为3的组组合元素.
#   维度0上有4个位置, 4个组组合组成当前tensor



# 3 strides, 定义 tensor 的索引寻址偏移, 比如(6, 2, 1) 代表在维度 (0, 1, 2)的寻址偏移量: 
#   给定维度1和2, 在维度0上移动一个单位, 需要在meta-data中偏移6。维度0上最多可移动4-1=3个单位
#   给定维度0和2, 在维度1上移动一个单位, 需要在meta-data中偏移2。维度1上最多可移动3-1=2个单位
#   给定维度0和1, 在维度2上移动一个单位, 需要在meta-data中偏移1。维度2上最多可移动2-1=1个单位

#   一般来说, strides[-1] 要保持 等于 1, 保证 tensor 在最后一个维度上的偏移 等价于 在 meta-data 中的偏移


## 
# 利用 strides 和 在meta-data上的总偏移 num_steps 确定 index 位置:
#   num_steps % strides[0] 得到的是 dimension 0 的 index, 余数 num_steps // strides[0] = num_step_expt_dim0
#   num_step_expt_dim0 % strides[1] 得到的是 dimension 1 的 index, 余数 num_step_expt_dim0 // strides[1] = num_step_expt_dim01
#   。。。
#   num_step_expt_ % strides[-1] 得到的是 dimension -1 (最后一个维度) 的 index


## 对于 non-overlapping 的 tensor data, shape 和 strides 之间存在一定的数字关系
# 1. strides[-1] 永远等于 1
# 2. strides[-2] 等于 shape[-1], 因为 strides[-2] 代表了 在倒数第二维度偏移一个单位, 需要 meta-data 上的偏移量。
#    在 non-overlapping 的前提下, 这个量就代表了 在倒数第一维度最多允许的偏移量，即 倒数第一维度的长度.
# 3. strides[-3] 等于 shape[-2] * shape[-1], 因为 strides[-3] 代表了 在倒数第三维度偏移一个单位, 需要 meta-data 上的偏移量。
#    在 non-overlapping 的前提下, 这个量代表偏移 最多倒数第二维度长度次偏移, 而每一次倒数第二维度上的偏移，都代表了 倒数第一维度长度次偏移。
#    故 得到了 shape[-2] * shape[-1] 的结果。
# 4. 其他剩余维度的关系类似: 在 non-overlapping 的前提下, strides 和 shape 之间存在累乘的关系


# 但是在通用的视角下, 基于 meta-data 之上的 shape 和 stride 不存在数字关系: shape 是 shape, stride 是 stride,
# shape 代表 各个 dimension, 在给定 前面低维 dimension 之后, 在当前维度 可以最多有多少次偏移量, 即为当前维度的 length-1
# 比如 tensor X 的shape 是 (d0, d1, d2, d3, d4), 代表在 dim0 可以最多有 d0-1 次偏移(d0个位置); 选定 dim0 后, 在 dim1 可以最多有 d1-1 次偏移(d1个位置)

# stride 代表 各个dimension, 在当前维度 偏移一个单位（其他维度不偏移），需要在 meta-data 上等效移动 多少个单位, 即为当前维度的 stride
# 比如 tensor X 的stride 是 (s0, s1, s2, s3, s4), 代表在 dim0 一次偏移(其他dim不动), 等价于在 meta-data 上偏移 s0 个单位
# 可以看出 stride 只需要满足 都是 正整数 即可. 各维度上的  stride 数字之间 没有任何 限制关系: 即
# 给定 meta-data, 给定 shape 和 stride(相同长度), shape 和 stride 除了正整数外无其他要求, 即可从 meta-data 中寻址, 构建满足的 tensor





# unfold(dimension, size, step) 方法 不改变 tensor 的 meta-data, 通过在维度 dimension 上作 宽度为 size 的滑动窗口, 并以 step 为跳跃步长滑动
# 对于一个 shape 为 (d_0, d_1, ..., d_i, ..., d_n), stride 为 (s_0, s_1, ..., s_i, ..., s_n). 现在要 unfold dimension i
# unfold(dimension = d_i, size = window_size, step = step), 那么:
# 对于 d_i 前面的所有维度来说, 即 d_0, ..., d_i-1, 没有变化。即本来是 d_0, ..., d_i-1 各维度 index 都确定后, 得到一个 (d_i,...,d_n) 的tensor, 
# 只是现在这个 (d_i, ..., d_n) 的 tensor 要被 unfold 成另一个形状了。
# 所以 前面维度的shape d_0, ..., d_i-1 和 前面维度的stride s_0, ..., s_i-1 都没有变化。

# unfold 的关键在于, 原来shape 为 (d_i,...,d_n), stride 为 (s_i,...,s_n) 的 tensor, 要如何作 窗口宽度为 window_size, 滑动跨度为 step 的 unfold 操作?





# 1、 以 d_i 为总长度（滑动过程中窗口不越界） 作 窗口宽度为 window_size、滑动步长为 step 的滑动，那么总共会产生 
# [ (d_i - window_size)/step ] + 1 = num_windows 个窗口
#       (显然 d_i 维度中最后几个维度可能不会被滑动窗口覆盖, 故可能在新tensor中消失)

# 原 tensor 在第 i 维上, stride 是 s_i, 长度是 d_i 个 shape为(d_i+1,...d_n)/stride为(s_i+1,...s_n) 的 原sub-tensor

# 新 tensor 在第 i 维上, stride 是 s_i * step
#       (原来 d_i 维, 每偏移一个单位, 在meta-data上偏移s_i; 现在是 num_windows 维, window 偏移到下一个 window, 跳过了 step 个原单位, 故新的stride = s_i*step)

# 长度是 num_windows 个 
#   "由 window_size 个 原sub-tensor, 即 shape为(window_size, d_i+1,...d_n)/stride为(s_i, s_i+1,...s_n) 的tensor permute 的 新sub-tensor"


# 有点绕, 首先总结 新 tensor 在第 i 维上, stride 是 s_i * step, 长度是 num_windows 个 新 sub-tensor.

# 其次, 考虑不 permute 的unfold方法, 那么当 unfold 在 第i维, 过程应该是:
# shape(d_i,...,d_n) / stride(s_i,...,s_n)  ------------unfold on dimension i------------> 
# shape(num_windows, window_size, d_i+1,...d_n) / stride(s_i*step, s_i, s_i+1,...s_n)
# 这个过程很好理解, 即 d_i 维被拆分成了两个维度, 长度分别是 num_windows, window_size，stride 分别是 s_i*step, s_i

# 但是 torch.unfold 对 unfold 的结果做了一个 额外的 permute:
# shape(num_windows, window_size, d_i+1,...d_n) / stride(s_i*step, s_i, s_i+1,...s_n)  ------------permute------------>
# shape(num_windows, d_i+1,...d_n, window_size) / stride(s_i*step, s_i+1,...s_n, s_i)
# 即把 新增的 窗口内维度 permute 到最后一维度

# torch.unfold 总结:
# unfold on dimension i with window size as size, step as step:  
# shape(d0,..., d_i-1, d_i,  d_i+1,...,d_n) / stride(s_0,..., s_i-1, s_i,  s_i+1,...,s_n)   ------vanilla unfold on dim i------>
# shape(d0,..., d_i-1, num_windows, window_size, d_i+1,...d_n) / stride(s_0,..., s_i-1, s_i*step, s_i, s_i+1,...s_n)  ------permute last dim------>
# shape(d0,..., d_i-1, num_windows, d_i+1,...d_n, window_size) / stride(s_0,..., s_i-1, s_i*step, s_i+1,...s_n, s_i)


def tensor_storage():
    # tensor 是指针: 指向 allocated memory of meta-data
    # meta-data:
    x = torch.tensor([
        [0., 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])
    # dtype/stride/shape
    print(f'stride as {x.stride()}')
    print(f'shape as {x.shape}')

    # for this tensor, skip 1 in dim 0 <--> skip 4 in storage
    assert x.stride(0) == 4
    # for this tensor, skip 1 in dim 1 <--> skip 1 in storage
    assert x.stride(1) == 1
    # how to index an element? 如何定位一个元素, 比如给定位置 row 1, col 2
    # 答: row / col 是 dim 0 / 1 上的 偏移量, 乘以 stride 0 / 1, 就得到在 storage 上的总偏移量
    r, c = 1, 2
    index = r * x.stride(0) + c * x.stride(1)
    assert index == 6

    # 对于大部分 tensor storage 上的 operation, 本质是 确定不同的 shape/stride 以 "改变" tensor 的view, storage 本身没有变化, 也没有 copy storage

    # 辅助函数: 根据 tensor 的原始存储 data 指针, 确定两个 tensor 是否处于 同一内存storage
    # 举例: 比如说 y 是 x 的另一种 view but share same storage, 那么 .storage() 方法会返回 原始storage
    def same_storage(x: torch.Tensor, y: torch.Tensor):
        return x.storage().data_ptr() == y.storage().data_ptr()

    # permute/transpose/view/冒号slice/等, 都是 "通过改变 shape/stride" 以改变视图, 没有改变 storage meta-data
    # permute/transpose
    y = x.T
    assert same_storage(x, y)
    
    # view/
    y = x.view(2, 4, 2)
    assert same_storage(x, y)

    # 整数index/冒号slice 等 basic slicing 
    y = y[:, 1:, 1:]
    assert same_storage(x, y)

    # ! advanced slicing 会触发 copy 创建新的 storage, 从而有新的 data_ptr
    # gather/index_select/mask_select/张量或列表索引 等都是 advanced slicing

    
    # 检查 对 x 的 mutation 也会发生在 y 上: 因为 x 和 y share storage
    x[0][0] = 100
    assert y[0][0][0] == 100

    # re-view 操作 虽说不改变 storage, 但可能使得 tensor 不再 连续(storage 必然是连续的)
    # 比如 permute/transpose/basic slicing等
    y = x[:, 1]
    assert same_storage(x, y)
    assert not y.is_contiguous() # y 虽然和 x 仍然share storage, 但 y 自身不再连续

    # y 的 storage 不再连续之后, view 成其他形状时, 除非极少数情况下 stride/shape 符合要求, 不然大概率都是不行的

    # 强制 storage 连续化, 会触发 copy
    y = x[:, 1].contiguous()
    assert same_storage(x, y)
    


def tensor_elementwise():
    # tensor 的运算之一: 逐元素计算, 并返回 output tensor as new
    x = torch.tensor([1, 4, 9])

    # torch.equal 和 == 操作符一样, 关注数值上的相等性
    assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))
    assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))
    assert torch.equal(x.rsqrt(), torch.tensor([1, 1/2, 1/3])) # x_i --> 1/sqrt(x_i)
    assert torch.equal(x/0.5, torch.tensor([2, 8, 18]))
    # operations on matrix
    x = torch.ones(3, 3).triu() # 即 torch.triu(), 取上三角, 参数 diagonal 表达相对对角线的上移步数
    

def tensor_matmul():
    # 矩阵乘法是 tensor 最关键的能力
    x = torch.ones(4, 8, 16, 32)
    w = torch.ones(32, 2)

    y = x @ w
    assert y.shape == torch.Size([4, 8, 16, 2])

import torch
from jaxtyping import Float
import einops
def tensor_einops():
    # Einstein Operations: 来自 爱因斯坦 summation notation: 命名维度,  并用维度的名字来定义操作
    
    # 维度命名方法: jaxtyping
    x : Float[torch.Tensor, f"batch seq heads hidden"] = torch.ones(2, 2, 1, 3)
    

    # einops 之 einsum: 通过指定 维度变换, 指定 矩阵乘法
    x: Float[torch.Tensor, f"batch seq1 hidden"] = torch.ones(2, 3, 4)
    y: Float[torch.Tensor, f"batch seq2 hidden"] = torch.ones(2, 3, 4)
    # 操作: 计算 x @ y.T

    # old way:
    z = x @ y.transpose(-2, -1) # [B, seq1, h] @ [B, h, seq2] --> [B, seq1, seq2]

    # einsum way: 在 output dims 里没有申明 name 的dim被 summed over, 故此操作叫 einsum
    z = einops.einsum(x, y, 'batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2')
    # 可以用 ... 代表 broadcast 的维度
    z = einops.einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")

    # torch 包装了 einsum, 不过 dim-naming 方式不同
    z = torch.einsum('bsd,bld->bsl', x, y)



    # einops 之 reduce: 用 sum/mean/min/max 等操作规约 a single dim
    x: Float[torch.Tensor, f"batch seq hidden"] = torch.ones(2, 3, 4)

    # old way:
    z = x.sum(dim=-1) # reduce sum on last dim

    # einops way:
    z = einops.reduce(x, '... hidden -> ...', 'mean')



    # einops 之 rearrange: 即 reshape, 比如这里把 last dim 8 拆成 2*4
    x: Float[torch.Tensor, f"batch seq total_hidden"] = torch.ones(2, 3, 8)
    # total_hidden = flattend of  num_heads * dim_head, num_heads =2, dim_head = 4
    w: Float[torch.Tensor, f"dim_head out_dim"] = torch.ones(4, 4)

    # old way:
    x = x.reshape(2, 3, 2, 4)
    x = x @ w
    x = x.reshape(2, 3, 8)

    # einops way: 代表 last 本来是 heads * hidden1, 现在要 拆开 且 heads = 2
    x = einops.rearrange(x, '... (heads hidden1) -> ... heads hidden1', heads=2)
    x = einops.einsum(x, w, '... hidden1, ... hidden2 -> ... hidden2')
    # 代表 last two dim heads & hidden2 flatten 成一维
    x = einops.rearrange(x, '... heads hidden2 -> ... (heads hidden2)')

    