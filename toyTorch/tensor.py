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
    assert x != 0

    assert x.element_size() == 2



