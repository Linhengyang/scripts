



def create_mask_on_last_dim(last_dim_size: int, mask_lens: torch.Tensor, mask_flag:bool = True) -> torch.Tensor:
    '''
    input:
        last_dim_size: int, dim size of a tensor's last dim
        mask_lens: tensor with shape as (...,), dtype as int
        mask_flag: 指定 mask 部分是 True 还是 False
    return:
        mask tensor: (..., last_dim_size), dtype as bool
        mask 是一个 True/False tensor, 它的 shape 等于 mask_lens.shape + last_dim_size
    
    explain:
        mask_lens 中的单个元素用位置(...,)确定后, 在原 tensor 中有唯一的长度为 last_dim_size 的 1-D tensor 与其对应,
        mask_lens 元素值代表了这个 1-D tensor 从index 0(包括)开始, 有多少个是要 mask 的

        mask_lens 元素值可以为 0, 代表原 tensor 在该位置的长度为 d_size 的1-D tensor 所有都不需要mask.
    '''
    # index_1dtensor: [0, 1, ..., last_dim_size-1]
    index_1dtensor = torch.arange(last_dim_size, device=mask_lens.device, dtype=mask_lens.dtype)

    # 伸展 mask_lens 的last dim, 即变成单个元素的 1d tensor. index_1dtensor 和 单个元素的1dtensor broadcast对比
    # [0, 1, ..., last_dim_size-1] with shape as (last_dim_size, ) <?< [ ...,[valid_len_1], [valid_len_2], ..., ] with shape as (..., 1)
    mask = index_1dtensor < mask_lens.unsqueeze(-1)
    # braodcast 对比: [0, 1, ..., n_logits-1] 逐一 和 valid_len_i 对比, 得到 T/F mask shape as (..., n_logits)
    # 此时 mask 部分是 True, 非 mask 部分是 False. 符合 mask_flag 为 True 的情况.

    # 如果 mask_flag 为 False, 那么 mask 部分应该是False, 非 mask 部分为 True: 取反
    if not mask_flag:
        mask = ~mask

    return mask