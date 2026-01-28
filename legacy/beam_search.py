import torch
from torch import nn
import math
import re
from operator import itemgetter
import typing as t
from src.core.interface.infra_easy import easyPredictor
from src.core.evaluation.evaluate import bleu
from src.utils.text.text_preprocess import preprocess_space
from src.utils.text.string_segment import sentence_segment_greedy
from src.core.data.assemble import truncate_pad



def greedy_predict(
        net, tgt_vocab, num_steps, src_inputs, length_factor, device,
        *args, **kwargs):
    '''
    net:
    tgt_vocab: 生成序列的vocab. 需要用到其 序列生成的 begin-of-sequence tag 和 停止生成的end-of-sequence tag, 以及 vocab_size
    num_steps: 模型 net 能捕捉的最大 序列关系 步长. 正整数
    src_inputs: source 序列相关信息, 应该为 (src, src_valid_lens)
        src shape: (1, num_steps)int64, timesteps: 1 -> num_steps
        src_valid_lens shape: (1,)int32
    length_factor: 生成序列的长度奖励, 长度越长奖励越大
    device: 手动管理设备. 应该和 net 的设备一致

    output: @ CPU
        predicted tokens(eos not included), along with its score
    '''
    src_inputs = [tensor.to(device) for tensor in src_inputs]
    net.to(device)

    net.eval()
    with torch.no_grad():
        # encoder / decoder.init_state in infer mode:
        # encoder input: src_inputs = (src, src_valid_lens): shapes  [(1, num_stepss), (1,)]
        # init_state output: src_enc_info = (src_enc, src_valid_lens): [(1, num_stepss, d_dim), (1,)]
        src_enc_info = net.decoder.init_state(net.encoder(*src_inputs))

        # decoder in infer mode:
        # input: tgt_query shape: (1, 1)int64, 对于第i次infer, tgt_query 的timestep是 i-1 (i = 1, 2, ..., num_steps), output的timestep 是 i
        #        src_enc_info = (src_enc, src_valid_lens): [(1, num_stepss, d_dim), (1,)]
        #        input KV_Caches: 
        #           Dict with keys: block_ind,
        #           values: 对于第 1 次infer, KV_Caches 为 空
        #                   对于第 i > 1 次infer, KV_Caches 是 tensors shape as (1, i-1, d_dim), i-1 维包含 timestep 0 到 i-2
        tgt_query = torch.tensor( [tgt_vocab['<bos>'],], dtype=torch.int64, device=device).unsqueeze(0)

        output_tokenIDs, KV_Caches, raw_pred_score = [], {}, 0

        for _ in range(1, num_steps+1):
            # 对于第 i 次 infer:
            # output: tgt_next_hat shape: (1, 1, vocab_size)tensor of logits, 对于第i次infer, timestep 是 i;
            #         output KV_Caches: dict of 
            #              keys as block_indices
            #              values as (1, i, d_dim) tensor, i 维 包含 timestep 0-i-1, 实际上就是 input KV_Caches 添加 timestep i-1 的 tgt_query
            Y_hat, KV_Caches = net.decoder(tgt_query, src_enc_info, KV_Caches)

            pred_tokenIDX = Y_hat.argmax(dim=-1).item() # .item() 把值从 device 移到cpu上

            # greedy predict 是可以提前结束的, 不一定要执行 num_steps 次
            if pred_tokenIDX == tgt_vocab['<eos>']:
                break

            output_tokenIDs.append(pred_tokenIDX)
            raw_pred_score += torch.log( nn.Softmax(dim=-1)(Y_hat).max(dim=-1).values ).item()
            # raw_pred_score = log(estimated probability of selected one)
            
        long_award = math.pow(len(output_tokenIDs), -length_factor)
    
    # return [pred_tokn1, .., pred_toknN], score
    return tgt_vocab.to_tokens(output_tokenIDs), raw_pred_score*long_award



# greedy search 贪心搜索: 第 t(1=1,2,...num_steps)次infer, 从 Seq(0至t-1), net on Seq(0至t-1) 出 Tok(t) 的 vocab_size 个选择, 
# 选择其中概率最大的作为 Tok(t)


# beam search 束搜索: 第 t(1=1,2,...num_steps)次infer, 不仅有 Seq(0至t-1), 还有 condProbs(0至t-1), 
# condProbs的元素是 Tok(i) 在使用 net Seq(0至i-1) 生成时的概率. 这里 condProbs(0) = 1. 
# 这个概率是条件概率:
#   在 net Tok(t) 时, net on Seq(0至t-1) 出 Tok(t) 的 vocab_size 个选择, softmax 后就是 vocab_size 个 Tok(t)|Seq(0至t-1) 的条件概率

# beam search 时, 每次选择都保留 k 个结果, 即有 Seq(0至t-1)_1, ..., Seq(0至t-1)_k, 和对应的 condProbs(0至t-1)_1,..., condProbs(0至t-1)_k

# 在寻找 Tok(t) 时, net on Seq(0至t-1)_1, ..., Seq(0至t-1)_k, 得到 k * vocab_size 个选择, 即一张 shape 为 (k, vocab_size) 的 token logits mat.
# token logits mat 的 第 j 行, 是 Seq(0至t-1)_j 作为前置序列, 生产出来的给 Tok(t) 的 vocab_size 个选择的 logits, 
# softmax 后即为 vocab_size 个 Tok(t)|Seq(0至t-1)_j 的条件概率

# 求出这 k * vocab_size 个选择各自的条件概率, 即
#   softmax(dim=-1) on (k, vocab_size) 的 token logits mat --> token condProb mat, shape 为 (k, vocab_size)

# 那么 Seq(0至t-1)_j + Tok(t) 的序列总概率, 即为 Tok(t)|Seq(0至t-1)_j 的条件概率 * 连乘 condProbs(0至t-1)_j.
# 对于 Seq(0至t-1)_j 为前置序列, 有 vocab_size 个序列总概率
# 对于 k 条 前置序列 Seq(0至t-1)_1, ..., Seq(0至t-1)_k, 得到 k * vocab_size 个序列总概率, 
# 是一张 shape 为 (k, vocab_size) 的矩阵 seq prob mat: 元素为 序列总概率, 行坐标代表前置序列, 列坐标代表 Tok(t)
# 取 seq prob mat 的 前 k 个最大值, top1, top2,...topk. 对于 j = 1,2..k, topj 的行坐标代表的前置序列, append topj 的列坐标代表的 Tok(t),
# 得到了 序列总概率最大的 k 个 Seq(0至t), 分别是 Seq(0至t)_1,...Seq(0至t)_k

# 对于第 t 步predict (1=1,2,...num_steps)
def beam_search_single_step(net, vocab_size, src_enc_info, k, parrallel,
                            k_seq_mat, k_cond_prob_mat, k_KV_Caches=None
                            ):
    # beam_size == k
    # k_seq_mat: (k, t)int64, 包含 timestep 0 至 t-1, k 条 Seq(0至t-1), 每条作为行. 对于第1步predict, k_seq_mat 是 k 条  timestep=0的<bos>
    # k_cond_prob_mat: (k, t), 包含 timestep 0 至 t-1, k 条 Cond Prob序列(0至t-1), 每条作为行. timestep=0, Cond Prob=1

    # k_KV_Caches: list of dicts, 考虑每个 dict, 
    #   对于 第 1 次predict, 为空 {}
    #   对于第 t > 1 次predict, keys 是 block_inds, values 是 tensors shape as (1, t-1, d_dim), t-1 维包含 timestep 0 到 t-2
    # k_KV_Caches 从原理上来说, 没有必要输入, 因为beam search记录了k条 timestep 0-t-1 的历史结果在 k_seq_mat,
    # 所以理论上, k 条 seq 对应的 decoder block caches 都可以用 seq 里的tokens 逐一重新算一遍. 但这样会造成极大的算力浪费.

    # 计算 cond_prob_tokn_t_mat shape(k, vocab_size), 即 token t 的 k 个条件概率
    # 只有 net 设定为 training 模式, 才能作 并行前向计算. 此时也不需要 KV_Caches
    if parrallel and net.training:
        with torch.no_grad():
            logits, _ = net.decoder(k_seq_mat, src_enc_info) # (k, num_steps, vocab_size)tensor of logits,  timestep 从 1 到 t;
            logits_tokn_t = logits[:, -1, :].squeeze(1) # (k, 1, vocab_size) --> (k, vocab_size)
            cond_prob_tokn_t_mat = nn.Softmax(dim=-1)(logits_tokn_t) # (k, vocab_size)

    elif isinstance(k_KV_Caches, list) and isinstance(k_KV_Caches[0], dict):
        # 对应关系, k_seq_mat <行 对应 行> k_cond_prob_mat <行 对应 index> k_KV_Caches

        cond_prob_tokn_t_mat = torch.zeros((k, vocab_size), device=k_seq_mat.device) # (k, vocab_size)
        for i in range(k):
            # 对于 第 i 个序列, i = 0, 1, 2, ...k-1

            # k_seq_mat[i:i+1, -1:] 取 i 行 最后一个元素, shape (1, 1). k_KV_Caches[i] 在第一步时, 可以为 {}
            logits, k_KV_Caches[i] = net.decoder(k_seq_mat[i:i+1, -1:], src_enc_info, k_KV_Caches[i]) # logits(1, 1, vocab_size)
            cond_prob_tokn_t_mat[i, :] = logits[0][0] # logits[0][0] 以 (vocab_size,) 的 1d tensor 填入 cond_prob_tokn_t_mat 第 i 行
        
        cond_prob_tokn_t_mat = nn.Softmax(dim=-1)(cond_prob_tokn_t_mat) # softmax on dim -1. shape (k, vocab_size)
    
    else:
        raise ValueError(f'k_KV_Caches must be a list of dict')
    
    # beam search
    prob_seqs = k_cond_prob_mat.prod(dim=1, keepdim=True) * cond_prob_tokn_t_mat # (k, vocab_size)

    topk = torch.topk(prob_seqs.flatten(), k)

    # 找到下一组 k 候选
    row_inds = torch.div(topk.indices, vocab_size, rounding_mode='floor') # (k, )
    col_inds = topk.indices % vocab_size # (k, )

    # 下一组 k 候选的 条件概率矩阵
    k_cond_prob_tokn_t = cond_prob_tokn_t_mat[row_inds, col_inds]
    # (k, ), 从 timestep t 的 cond_prob_tokn_t_mat (k, vocab_size) 中选出的 top k的 条件概率

    # append selected top k's 条件概率 to k_cond_prob_mat
    next_k_cond_prob_mat = torch.cat([k_cond_prob_mat[row_inds], k_cond_prob_tokn_t.reshape(-1, 1)], dim=1) #(beam_size, t)

    # 下一组 k 候选的 token 序列矩阵
    # append selected top k's token ID to k seqs
    next_k_seq_mat = torch.cat([k_seq_mat[row_inds], col_inds.reshape(-1,1)], dim=1) #(k, t+1)

    # # 下一组 k 候选的 k_KV_Caches
    k_KV_Caches = list( itemgetter(*row_inds)(k_KV_Caches) )

    return next_k_seq_mat.type(torch.int64), next_k_cond_prob_mat, k_KV_Caches





# 从 k 个 <bos>开始, 执行 t 步 beam search (1=1,2,...num_steps).
# 由于最终选择的标准和 总长度 相关, 有两种寻找方案：
#   记录 t 步中所有的中间结果, 在 k*t 个结果中, 计算分数, 选择最大的
#   只记录 完成 t步后的 k 个最终结果. 在 k 个结果中, 计算分数, 选择最大的
def beam_predict(
        net, tgt_vocab, num_steps, src_inputs, length_factor, device,
        beam_size, parrallel, *args, **kwargs):
    '''
    net:
    tgt_vocab: 生成序列的vocab. 需要用到其 序列生成的 begin-of-sequence tag 和 停止生成的end-of-sequence tag, 以及 vocab_size
    num_steps: 模型 net 能捕捉的最大 序列关系 步长. 正整数
    src_inputs: source 序列相关信息, 应该为 (src, src_valid_lens)
        src shape: (1, num_steps)int64, timesteps: 1 -> num_steps
        src_valid_lens shape: (1,)int32
    length_factor: 生成序列的长度奖励, 长度越长奖励越大
    device: 手动管理设备. 应该和 net 的设备一致
    beam_size: 束搜索 的 束宽
    parrallel: 是否并行计算

    output: @ CPU
    predicted tokens(eos not included), along with its score
    '''
    src_inputs = [tensor.to(device) for tensor in src_inputs]
    net.to(device)

    src_enc_info = net.decoder.init_state(net.encoder(*src_inputs))# src_enc_seqs, src_valid_lens: (1, num_steps, d_dim), (1, )

    # 初始的 k_seq_mat 和 k_cond_prob_mat
    if parrallel and net.training:
        # enc_info = (src_enc_seqs.repeat(beam_size, 1, 1), src_valid_lens.repeat(beam_size))
        # dec_X = torch.tensor( [tgt_vocab['<bos>'],], dtype=torch.int64, device=device).unsqueeze(0)
        # #first pred
        # logits, _ = net.decoder(dec_X, (src_enc_seqs, src_valid_lens)) #logits shape(batch_size=1, num_steps=1, tgt_vocab_size)
        # topk_1 = torch.topk(logits.flatten(), beam_size)
        # pred_token_mat_1 = topk_1.indices.unsqueeze(1).type(torch.int64) #(beam_size, 1)
        # cond_prob_mat_1 = nn.Softmax(dim=-1)(topk_1.values).unsqueeze(1) #(beam_size, 1)
        # all_pred_token_mat, all_cond_prob_mat = [pred_token_mat_1], [cond_prob_mat_1]
        # k_pred_token_mat, k_cond_prob_mat = pred_token_mat_1, cond_prob_mat_1
        raise NotImplementedError(f'parrallel mode not implemented')
    else:
        # init k_seq_mat:  (k, 1)int64, all tgt bos
        k_seq_mat = torch.tensor( [ tgt_vocab['<bos>'], ]*beam_size, dtype=torch.int64, device=device).reshape(beam_size, 1)
        # init k_cond_prob_mat:  (k, 1)float, all 1.
        k_cond_prob_mat = torch.tensor( [1.,]*beam_size,  device=device).reshape(beam_size, 1)
        # init k_KV_Caches:  (k, )list of dict, all {}.
        k_KV_Caches = [{},]*beam_size
        # _transformer block在forward的时候,会将KV_Caches的tensor生成在tgt_query相同的device上

    # beam predict 不可以提前结束的, 必须要执行 num_steps 次留下 k 个结果, 并在 其中选择分数最高的
    for _ in range(1, num_steps+1):
        # 对于第 t 次 infer:
        # k_seq_mat: (k, t)int64, 包含 timestep 0 至 t-1, k 条 Seq(0至t-1), 每条作为行. 对于第1步predict, k_seq_mat 是 k 条  timestep=0的<bos>
        # k_cond_prob_mat: (k, t), 包含 timestep 0 至 t-1, k 条 Cond Prob序列(0至t-1), 每条作为行. timestep=0, Cond Prob=1

        # k_KV_Caches: list of dicts, 考虑每个 dict, 
        #   对于 第 1 次predict, 为空 {}
        #   对于第 t > 1 次predict, keys 是 block_inds, values 是 tensors shape as (1, t-1, d_dim), i-1 维包含 timestep 0 到 i-2

        k_seq_mat, k_cond_prob_mat, k_KV_Caches = beam_search_single_step(net, len(tgt_vocab), src_enc_info,
                                                                          beam_size, parrallel,
                                                                          k_seq_mat, k_cond_prob_mat, k_KV_Caches)
        
    # k_seq_mat: (k, num_steps+1)int64, 包含 timestep 0 至 num_steps
    # k_cond_prob_mat: (k, num_steps+1), 包含 timestep 0 至 num_steps, k 条 Cond Prob序列(0至num_steps)

    # 对于 k_seq_mat 中的每一行, 找到 <eos>. 其前面的 tokens 才是 effective output. 如果没有 <eos>, 那么所有 都是 effective output.
    # 在 train data 里, eos 包含在 valid area 里. 这里评价结果的方法中, 不包含 eos, 所以这里用 effective 与 valid 区分开
    eos_bool_mat = k_seq_mat == tgt_vocab['<eos>'] # (k, num_steps+1)bool: 等于 tgt eos 的地方为 True, 其余为 False
    eos_exist_arr = eos_bool_mat.any(dim=1) # (k,), 每行序列中是否有 eos 得到的 bool array

    eos_indices = eos_bool_mat.int().argmax(dim=1) # (k,) 每行而言, eos处在哪个位置. 不存在 eos 的行, 该值是 0.

    # 由于开头有 bos 存在, 故这个值-1 等于 该行 effective token个数（不包括bos和eos）
    effective_lens = eos_indices - 1 # (k,)
    effective_lens[~eos_exist_arr] = num_steps # (k,) 对于 不存在 eos 的行, effective token 数量就是 num_steps

    # 计算每行序列的 score = sum of {log conditional probs on effective area} / (effective_length^length_factor)
    k_seq_mat = k_seq_mat[:, 1:] # (k, num_steps) 去掉了 bos
    log_cond_probs = torch.log(k_cond_prob_mat[:, 1:]) # (k, num_steps) 去掉了 bos

    effective_area_mask = torch.arange(0, num_steps, device=device).unsqueeze(0) < effective_lens.unsqueeze(1)
    # (k, num_steps), True for effective token, False for in-effective token
    
    scores = (log_cond_probs*effective_area_mask).sum(dim=1) * torch.pow(effective_lens.to(torch.float), -length_factor) # (k, ) * (k, )
    # 选择分数最大的
    max_ind = scores.argmax().item() # .item() 把 max_ind 在cpu上
    # 取 k_seq_mat 分数最高的 行, 以及该行 前 effective_lens[mx_ind] 部分(即该行的 effective 部分), .tolist() 移动数据到 cpu 上
    output_tokenIDs = k_seq_mat[max_ind, :effective_lens[max_ind]].tolist()

    # return [pred_tokn1, .., pred_toknN], score
    return tgt_vocab.to_tokens(output_tokenIDs), scores[max_ind].item()













class sentenceTranslator(easyPredictor):
    def __init__(self,
                 src_vocab, tgt_vocab, # 两个语言的词汇表
                 net, num_steps, # 模型, 步长
                 search_mode='greedy', bleu_k=2, device=None, beam_size=3, length_factor=0.75, # 生成的模式及参数
                 ):
        
        super().__init__()

        # load src/tgt language vocab
        self._src_vocab, self._tgt_vocab = src_vocab, tgt_vocab

        # set device
        if device is not None and torch.cuda.is_available():
            self.device = device
        else:
            self.device = torch.device('cpu')
        print(f"use device {self.device} to infer")

        # set predict function/model/num_steps
        self.net = net
        self.num_steps = num_steps
        
        if search_mode == 'greedy':
            self.pred_fn = greedy_predict
            self.beam_size = None
        elif search_mode == 'beam':
            self.pred_fn = beam_predict
            self.beam_size = beam_size
        else:
            raise NotImplementedError(
                f'search_mode {search_mode} not implemented. Should be one of "greedy" or "beam"'
                )
        
        # set evaluate function
        self.eval_fn = bleu
        self.bleu_k = bleu_k
        self.length_factor = length_factor # length_factor 越大, 输出越偏好长序列





    def predict(self, src_sentence, need_lower=True, separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        # 从 text 转换成 input data

        # 和训练相同的预处理
        self.src_sentence = preprocess_space(src_sentence, need_lower, separate_puncs, normalize_whitespace=True)

        # 和训练相同的tokenize
        # 根据 src EOW_token 是否为 '', 确定 src glossary
        if self._src_vocab.to_tokens(self._src_vocab.eow):
            src_glossary = {'tokens':self._src_vocab.tokens, 'EOW_token':self._src_vocab.to_tokens(self._src_vocab.eow)}
        else:
            src_glossary = None

        source, _ = sentence_segment_greedy(
            self.src_sentence,
            glossary = src_glossary,
            UNK_token = self._src_vocab.to_tokens(self._src_vocab.unk),
            flatten = True,
            need_preprocess = False # 已经有预处理了
            )

        # 和训练相同的tokenize: output src_array:(1, num_steps)int64, valid_length:(1,)int32
        source = self._src_vocab[ source ] + [ self._src_vocab['<eos>'] ] # add src_eos. map to integer, list of integer
        source = truncate_pad(source, self.num_steps, self._src_vocab['<pad>']) # truncate/pad to (num_steps,) list of integer

        src_array = torch.tensor( source, dtype=torch.int64).unsqueeze(0) # (num_steps,) --> (1, num_steps), tensor of int64
        src_valid_len = (src_array != self._src_vocab['<pad>']).type(torch.int32).sum(1) # (1,), tensor of int32


        ## infer
        # net:,
        # tgt_vocab:,
        # num_steps:,
        # src_inputs: src_array, src_valid_len,
        # length_factor:,
        # device:,
        # other
        output_tokens, self._pred_scores = self.pred_fn(
            self.net, self._tgt_vocab, self.num_steps, (src_array, src_valid_len), self.length_factor, self.device,
            self.beam_size, parrallel=False)
        
        tgt_EOW_token = self._tgt_vocab.to_tokens(self._tgt_vocab.eow)
        if tgt_EOW_token: # 如果 target eow token 不是空字符
            self.pred_sentence = re.sub(tgt_EOW_token, ' ', ''.join(output_tokens))
        else: # 如果 target eow token 是空字符
            self.pred_sentence = ' '.join(output_tokens)
        
        return self.pred_sentence



    def evaluate(self, tgt_sentence, need_lower=True, separate_puncs='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        assert hasattr(self, 'pred_sentence'), f'predicted target sentence not found'

        # 和训练相同的预处理
        self.tgt_sentence = preprocess_space(tgt_sentence, need_lower, separate_puncs, normalize_whitespace=True)
        
        return self.eval_fn(self.pred_sentence, self.tgt_sentence, self.bleu_k)



    @property
    def pred_scores(self):
        return self._pred_scores