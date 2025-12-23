# Tokenizer.py
# new version of Tokenizer inspired by GPT tokenizer(tiktokenizer)

from abc import ABC
import typing as t


class Tokenizer(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def encode(self, string: str) -> t.List[int]:
        raise NotImplementedError
    
    def decode(self, indices: t.List[int]) -> str:
        raise NotImplementedError



class CharacterTokenizer(Tokenizer):
    '''
    A tokenizer mapping between unicode characters and its unicode number as tokens
    encode: char --> integer between 0 and 0x10FFFF
    decode: integer between 0 and 0x10FFFF --> char
    '''
    def encode(self, string:str) -> t.List[int]:
        return list( map(ord, string) )
    
    def decode(self, indices: t.List[int]) -> str:
        # filter valid unicode index
        indices = [ i  for i in indices if 0<= i <= 0x10FFFF ]
        return "".join(map(chr, indices))
    


class ByteTokenizer(Tokenizer):
    '''
    A tokenizer splits string into bytes, and use integers between 0 and 255 as tokens
    encode: string --> integer(between 0 and 255) sequence
    decode: integer(between 0 and 255) sequence --utf-8--> string with possible replacement
    '''
    def encode(self, string: str) -> t.List[int]:
        string_bytes = string.encode("utf-8") # 返回 utf-8 规范编码的 字节byte 序列. 所以返回的 string_bytes 是一个序列
        # 可以求 len, 返回 字节数量; 可以索引 index，返回各个 字节的整数值(0-255)
        # 英文 1 字节，欧洲字符 2字节，中日韩字符 3字节，罕见字符 4字节
        indices = list( map(int, string_bytes) ) # list of integers btw 0-255
        return indices
    
    def decode(self, indices: t.List[int]) -> str:
        # filter valid unicode index
        try:
            string_bytes = bytes(indices) # bytes 其中一种使用方式: 输入 list of integers, 要求每个 integer 0-255
            return string_bytes.decode('utf-8', errors='replace')
        except ValueError:
            print(f'input indices {indices} has wrong values. must be 0-255')






import regex as re
import os
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from ..system.math import check_monotonic


GPT4_TOKENIZER_REGEX = \
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


ENDOFTEXT = '<|endoftext|>'
FIM_PREFIX = '<|fim_prefix|>'
FIM_MIDDLE = '<|fim_middle|>'
FIM_SUFFIX = '<|fim_suffix|>'
ENDOFPROMPT = '<|endofprompt|>'




def get_pair_counts(tokens:t.List[int|str], p_counts:t.Dict[tuple[int|str, int|str], int]|None = None):
    '''
    p_counts 如果是 None: init a p_counts and return
        给定 tokens list, 计算它的 pair-tokens counts, 返回一个 pair counts dict
    p_counts 如果不是 None: in-place update p_counts
        给定 tokens list, 计算它的 pair-tokens counts, 更新 到 输入的 p_counts 里
    输入的 tokens 可以是 list of int, 也可以是 list of str. 因为 token 可以由 int / str 两种方式表示.
    '''
    if p_counts is None:
        p_counts = {}
    if len(tokens) == 1: # 如果只剩下 一个 token, 那么就无需 count pair / 更新 p_counts 
        return p_counts
    
    for pair in zip(tokens, tokens[1:]):
        p_counts[pair] = p_counts.get(pair, 0) + 1

    return p_counts





def merge_pair(tokens:t.List[int|str], pair:tuple[int|str, int|str], new_token:int|str) -> t.List[int|str]:
    new_tokens = [] # 返回一个 new_tokens, 而不是对 tokens 作 in-place 更新
    i = 0
    if len(tokens) == 1: # 如果只剩下 一个 token, 那么就没有 merge pair 的必要
        return tokens
    
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append( new_token )
            i += 2
        else:
            new_tokens.append( tokens[i] )
            i += 1
    return new_tokens





# 缓存这个函数, 因为预计会用完全相同的输入，调用很多次这个函数。缓存函数调用结果，下次遇到相同输入时，避免计算直接返回缓存结果
@functools.lru_cache(maxsize=128)
def _special_token_regex(tokens: frozenset[str]) -> "re.Pattern[str]":
    inner = "|".join(re.escape(token) for token in tokens)
    # 返回诸如 "(<pad>|<eos>|<bos>)" 之类的 special tokens 或运算符 拼接的 compiled regex
    # compiled regex 可以直接调用 .search(text) 方法
    return re.compile(f"({inner})")



def raise_run_out_corpus_error(num_occured_merges:int, num_specials:int) -> t.NoReturn:
    '''
    如果经过 tokens 累积更新 p_counts, p_counts 仍然是 {}, 说明 corpus 已经全部 merge 到一起, 无可 merge。
    不提前结束 merge 循环, 而是报错, 提示 更换参数 explicit_n_vocab 或 语料 corpus。
    原因是参数 explicit_n_vocab 语意为「明确的 vocab_size」,
    所以不应引入不确定性：当 explicit_n_vocab 和 corpus 冲突时, raise error而不是根据 corpus 确定 explicit_n_vocab。
    '''
    raise RuntimeError(
        f'run out of corpus(all merged together) after {num_occured_merges} merges.\n'
        f'the maximum valid `explicit_n_vocab` for this corpus is {256+num_occured_merges+num_specials}.\n'
        f're-init the tokenizer with lower `explicit_n_vocab` in between {256+num_specials}'
        f'(zero-merge) & {256+num_occured_merges+num_specials}(exactly-ran-out-of current corpus), '
        f'or with enlarged corpus.'
        )






def raise_disallowed_special_token(token: str) -> t.NoReturn:
    raise ValueError(
        f'disallowed specials {token!r} found in text.\n'
        f'expand `allowed_special` if you want to encode the disallowed marks into special tokens.\n'
        f'narrow `disallowed_special` if you want to encode the disallowed marks as plain.\n'
        f'you can expand `allowed_special` to "all" or narrow `disallowed_special` to (),'
        f'both will ignore specials and tokenize the text as plain'
        )






def encode_to_ints(s:str, encoding='utf-8') -> t.List[int]:
    return list( s.encode(encoding) )



class baseBBPETokenizer(Tokenizer):
    '''
    merge_ranks 是 dict of [token_L, token_R] ---> merged_token
    其中 merged_token 是从 256 开始编号, 即 rank + 256, rank = 0, ..., num_merges-1
    故 rank(等同 merge_rank)是0开始的、merged_token 相对 256 的偏移量
    '''
    def __init__(
            self,
            name: str,
            buffer_dir: str,
            pat_str: str = GPT4_TOKENIZER_REGEX,
            merge_ranks: dict[tuple[int, int], int] = {},
            special_marks: list[str] = [ENDOFTEXT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, ENDOFPROMPT],
            explicit_n_vocab: int | None = None,
            **kwargs):

        self.name = name
        assert os.path.isdir(buffer_dir)
        self._buffer_dir = buffer_dir
        self.pat_str = pat_str
        self._merge_ranks = merge_ranks
        self._special_marks = special_marks
        # special marks 必须都能被 pat_str 切开，不然可能会导致 merge-regenerate-special_mark in BPE, 导致混淆
        assert all([ len(re.findall(pat_str, mark)) > 1 for mark in special_marks ]) # special_marks 可以为 []

        if merge_ranks: # 如果输入了非空的 merge_ranks
            # merge_ranks 的 RANK 应该是 256 至 MAX_RANK 的连续正整数: 递增 且 len(merge_ranks) = MAX_RANK - 255 且 首元素 = 256
            ranks_seq = list( merge_ranks.values() )
            assert check_monotonic(ranks_seq, mode='increase', strict=True) and len(ranks_seq) == ranks_seq[-1]-255 and ranks_seq[0] == 256

            # 可以直接 构建 vocab: token_ID --> bytes
            self._build_vocab()
            
            # 可以直接 注册 special_tokens，因为已经有 merge_ranks，无需 BPE train
            self._register_special_tokens()

            # 总的 vocab_size, 即 explicit_n_vocab 也随之 确定。若输入了 explicit_n_vocab，检查是否和 merge+special 匹配
            if explicit_n_vocab:
                # 总 vocab size 应该等于 256 + merge tokens size + n_special_tokens
                assert explicit_n_vocab == 256 + len(merge_ranks) + len(special_marks)

            self.explicit_n_vocab = 256 + len(merge_ranks) + len(special_marks)

        else: # 如果 没有输入非空的 merge_ranks
            # 那么需要 run BPE train process to build merges_ranks forrest. corpus text 将随后输入
            # 如果输入了 explicit_n_vocab, 只要 valid, 那么总 explicit_n_vocab将在这里确定
            if isinstance(explicit_n_vocab, int):
                assert explicit_n_vocab >= 256 + len(special_marks), \
                    f'pretrained merge_ranks forrest empty.\ninput explicit_n_vocab (shall be at least greater ' + \
                    f'than 256+{len(special_marks)}(num_special_marks)={256+len(special_marks)}.\n' + \
                    f'e.g, GPT2 tokenizer has explicit_n_vocab as 50257, with 1 special marks and 50000 merges.'
                
                self.explicit_n_vocab = explicit_n_vocab



    def _build_vocab(self):
        # vocab: token_ID --> bytes
        assert hasattr(self, "_merge_ranks")
        self._vocab = {i: bytes([i]) for i in range(256)} # initialize 0 - 255 --> bytes
        for (L_int, R_int), merged_int in self._merge_ranks.items():
            self._vocab[merged_int] = self._vocab[L_int] + self._vocab[R_int] # two bytes concatenate



    def _register_special_tokens(self):
        assert hasattr(self, "_merge_ranks") and hasattr(self, "_special_marks")
        # vocab key 是 int, value 是 bytes
        # inverse special tokens key 是 int, value 是 str
        # special tokens key 是 str, value 是 int
        # 不统一，是因为 special tokens 要保持 str, 以便作 正则分割操作
        self.special_tokens: dict[str, int] = {} # speical mark str --> special token id

        # special tokens 的 token ID 应该 紧接着 merge_ranks 的 MAX RANK，即 MAX_RANK + 1 开始
        # 这样 所有tokens的 mapping value 应该是 0 至 explicit_n_vocab-1 = 256 + len(merge_ranks) + len(special_marks) - 1 的连续正整数
        # 所以 注册 special_tokens 的工作应该在 获得 有效的 merge_ranks 之后
        if self._merge_ranks:
            SPECIAL_RANK_START = max( self._merge_ranks.values() ) + 1 # special starts from MAX_MERGE_RANK + 1
        else:
            import warnings
            warnings.warn(
                f'merge_ranks is Empty. now register special tokens right after 0 - 255 bytes.\n'
                f'run valid BPE train process to build merge_ranks forrest before register special tokens')
            SPECIAL_RANK_START = 256 # special starts from 255 + 1

        for i, sp_mark in enumerate(self._special_marks):
            self.special_tokens[sp_mark] = SPECIAL_RANK_START + i
        
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()} # special token id --> special mark str



    def _prepare_train(self, num_merges:int|None = None, *args, **kwargs):
        '''
        num_merges 是希望 tokenizer 最后达到的 merge_ranks 的大小. 它与 explicit_n_vocab(如果有)的关系是:
        explicit_n_vocab = 256 + num_merges + num_special_marks. 如果冲突, 以 explicit_n_vocab 为准

        真正在这里为训练轮次准备的 变量是 num_train_epochs
        '''
        # if explicit_n_vocab not exist, must input num_merges here
        if not hasattr(self, 'explicit_n_vocab'):
            assert isinstance(num_merges, int) and num_merges >= 0, f'num merges not set. must input num_merges >= 0.'
            self._num_merges = num_merges

        # if explicit_n_vocab already set, and num_merges input here. warning if not match
        elif isinstance(num_merges, int) and self.explicit_n_vocab - 256 - len(self._special_marks) != num_merges:
            # warning. used pre-set _num_merges
            import warnings
            warnings.warn(
                f'input `num_merges`{num_merges} is not consistent with `explicit_n_vocab` from initialization.\n'
                f'merge times derived from `explicit_n_vocab` shall be `explicit_n_vocab`-num_specials-256 = '
                f'{self.explicit_n_vocab-256-len(self._special_marks)}.\n'
                f'ignore `num_merges`, use `explicit_n_vocab` to run BPE.')
            self._num_merges = self.explicit_n_vocab-256-len(self._special_marks)

        # if explicit_n_vocab already set, and match with num_merges or num_merges is None
        else:
            self._num_merges = self.explicit_n_vocab-256-len(self._special_marks)
        
        assert self._num_merges > len(self._merge_ranks)-256, f'current size of merge_ranks - 256 must be smaller to num_merges'

        self._num_train_epochs = self._num_merges - len(self._merge_ranks)
        


    def _update_tokenizer(self, occur_most_pair:tuple[int, int], new_token:int, occurence:int|None):
        # rank = i: 0, ..., _num_mergs-1
        self._merge_ranks[occur_most_pair] = new_token
        self._vocab[new_token] = self._vocab[occur_most_pair[0]] + self._vocab[occur_most_pair[1]]

        if occurence: # occurence 不会是0
            print(f'merge process {len(self._merge_ranks)}/{self._num_merges}: {occur_most_pair} -> {new_token}'
                  f'[{self._vocab[new_token]}] had {occurence} occurences')



    def _init_tokens(self, corpora:t.List[str]|str, *args, **kwargs) -> t.Generator:
        if isinstance(corpora, str):
            corpora = [corpora]
        
        corpus = ENDOFTEXT.join( corpora )
        chunks_str: t.List[str] = re.findall(self.pat_str, corpus) # pre-split to list of string

        with ThreadPoolExecutor(max_workers=8) as e:
             yield_tokens = e.map(encode_to_ints, chunks_str) # a generator
        
        return yield_tokens


    def _clear(self):
        self._merge_ranks: dict[tuple[int, int], int] = {} # 初始化 _merge_ranks
        self._vocab: dict[int, bytes] = {i:bytes([i]) for i in range(256)} # 初始化 _vocab


    def bpe_single_merge(self, rank:int, tokens_generator:t.Generator, verbose:bool=False):
        # rank 从 0 开始
        agg_p_counts:t.Dict[tuple[int, int], int] = {}
        stored_tokens = []
        for tokens in tokens_generator:
            # 对于最多只有 1个token 的 tokens(即all merged together)
            # 它从本次 merge开始，不会再贡献影响任何后续 p_counts, 也没有更新的必要
            if len(tokens) <= 1:
                continue
            get_pair_counts(tokens, agg_p_counts)
            stored_tokens.append( tokens )
        
        if not agg_p_counts:
            self.save(self._buffer_dir) # 在raise error前先保存已经train好的tokenizer防止前功尽弃
            raise_run_out_corpus_error(rank, len(self._special_marks))
        
        occur_most_pair: tuple[int, int] = max(agg_p_counts, key=agg_p_counts.get)
        occurence: int|None = agg_p_counts[occur_most_pair] if verbose else None
        new_token: int = rank + 256
        del agg_p_counts

        yield occur_most_pair, occurence, new_token # first yield

        # yield remain as new tokens_generator
        for tokens in stored_tokens:
            yield merge_pair(tokens, occur_most_pair, new_token)
    

    def train_bpe(self, corpora:t.List[str]|str, num_merges:int|None = None, verbose:bool=False, *args, **kwargs):
        # baseTokenizer 因为没有中间结果可以缓存, 故续训（load merge_ranks 之后再输入corpus train），是没办法校对的
        # 所以 baseTokenizer 只能从头开始 train
        self._clear()
        self._prepare_train(num_merges)
        yield_tokens:t.Generator = self._init_tokens(corpora)
        
        # <merge循环有前后依赖，所以不能并行>
        # <从init_merge_ranks_size续训num_train_epochs轮次, rank从init_merge_ranks_size到total_size-1>
        for i in range(self._num_train_epochs):
            # rank := i + init_merge_ranks_size = i + 0 = i: 0, ..., _num_mergs-1
            yield_output = self.bpe_single_merge(i, yield_tokens, verbose)
            occur_most_pair, occurence, new_token = next(yield_output) # first yield
            
            self._update_tokenizer(occur_most_pair, new_token, occurence)
            
            yield_tokens = yield_output # continue to yield tokens. update yield_tokens

        # set down others
        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()



    def _encode_chunk(self, tokens:t.List[int]) -> t.List[int]:
        '''
        对 chunk(tokens) 作持续的 merge, 直至 无可merge
        '''
        while len(tokens) > 1: # 在循环体内不断更新 tokens
            p_counts: t.Dict[tuple[int, int], int] = get_pair_counts(tokens)
            # 取出 合并 rank 最小的 pair. 当无可合并时, 即 pairs 中无一存在于 _merge_ranks, 那么 min_rank_pair 不存在于 _merge_ranks, 该跳出合并循环
            min_rank_pair: tuple[int, int] = min(p_counts, key=lambda p: self._merge_ranks.get(p, float('inf')))
            if min_rank_pair not in self._merge_ranks:
                break
            # 更新 tokens
            tokens = merge_pair(tokens, min_rank_pair, self._merge_ranks[min_rank_pair])
        
        return tokens

    
    def encode_ordinary(self, text:str) -> t.List[int]:
        '''
        encoding text that ignores any special tokens
        无视任何 special tokens / marks, 纯粹从 utf-8 编码后的字节流 作持续不断的 merge, 以 tokenize
        '''
        # encode 方法必须先检查有无 special_tokens. special_tokens 可以空, 也可以不在 encode 中用到.
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated
        # raw
        chunks_str: t.List[str] = re.findall(self.pat_str, text) # pre-split to list of string
        # initial tokens: 可多线程加速
        chunks_tokens: t.List[list[int]] = [list( chunk.encode('utf-8') ) for chunk in chunks_str] # list of int(0..255)_list

        encoded_output: t.List[int] = []
        for tokens in chunks_tokens: # tokens: list of 0-255(int)
            encoded_output.extend( self._encode_chunk(tokens) )
        
        return encoded_output


    def encode_special(
            self,
            text:str,
            allowed_special:t.Literal["all"] | t.AbstractSet[str]
            ) -> t.List[int]:
        '''
        encoding text that first mapping registered and allowed special marks to special_token_ID, then tokenize the rest
        输入的 allowed_special 和 注册的 special_tokens 取交集.
            交集内的 special 若出现在 text 中, 则mapping as 注册的 special token ID
            text 其他部分采用 encode_ordinary 方式 编码
        '''
        # 要求本 tokenizer 已完成 specials 注册. special_tokens is dict of {str: int}
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated
        if allowed_special == 'all':
            specials = self.special_tokens
        else:
            # allowed_special 交集 registered_special  如果 allowed_special 为空, 那么 specials 也为 空
            specials = {k:v for k, v in self.special_tokens.items() if k in allowed_special} # dict of str: int

        if not specials: # 如果 specials 为 空
            return self.encode_ordinary(text)
        
        special_pat = '(' + '|'.join(re.escape(k) for k in specials) + ')'
        chunks = re.split(special_pat, text) # special tokens 从 text 中分离出来: list of str

        tokens = []
        for chunk in chunks:
            if chunk in specials: # 如果是 special mark, 直接匹配 注册的 special token ID
                tokens.append( self.special_tokens[chunk] )
            else: # 如果是 plain text, 使用 encode ordinary 编码成 tokens
                tokens.extend( self.encode_ordinary(chunk) )
        
        return tokens
    
    
    def encode(self, text:str,
               allowed_special:t.Literal["all"] | t.AbstractSet[str] = set(),
               disallowed_special:t.Literal["all"] | t.Collection[str] = "all"
               ) -> t.List[int]:
        '''
        按理来说, special tokens 是拿来控制 LLM 的, 不应该出现在 text 中。这里 额外处理special的逻辑与 OpenAI tiktoken 保持一致。
        即：如果在 text 中检测到 special tokens, 则 raise error。
        
        通过参数 allowed_special/disallowed_special 来 控制 special tokens 的粒度。allow 和 disallow 的区别在于是否 raise error.

        第一步 确定 disallowed specials, 以此 判断本次 encode 要不要 raise error: 若 text 中出现了 disallowed specials, 则 raise error; 否则进入第二步
        第二步 用 encode_special 方法来 encode text: 即 allowed specials 和 registered specials 的交集会被 map to special token ID, 其余全部正常encode
        
        1. 确定 disallowed specials。若 text 里包含 disallowed specials, 则 raise error。不包含则进入下一步
            如何确定 disallowed specials?
                1. input arg disallowed_special = all: 意味着 该tokenizer 注册的 special tokens 减去 arg allowed_special, 就是 disallowed specials
                (此时若 arg allowed_special = all, 则 disallowed_special 为 空，即 没有 disallow 的 special.)
                2. input arg disallowed_special = (): 意味着 disallowed_special 为 空，即 没有 disallow 的 special.
                3. input arg disallowed_special = set of str marks: 意味着 disallowed_special 是一个 valid 集合, 检测该集合的 marks 是否出现即可.
        
        2. 若在 第1步没有 raise error, 则采用 encode with special on text. 参数 allowed_special 确定了 map to special token ID 的 special marks范围.
           这里 allowed_special 即使和上文确定的 disallowed_special 有交集也无所谓的, 因为已经保证了 text 中不存在 disallowed_special.
        '''
        assert hasattr(self, 'special_tokens') # 有 / 无 special_tokens 属性, 可以区分 tokenizer 的 merge_ranks-empty / merge_ranks not-generated

        if allowed_special == "all":
            allowed_special = set( self.special_tokens ) # dict of {str: int} ---> set of str

        if disallowed_special == "all":
            disallowed_special = set( self.special_tokens ) - allowed_special # set - set ---> set of str

        if disallowed_special: # 如果到这里, disallowed_special 非空, 那么要对 text 作检测，保证其不出现 disallowed_special, 不然 raise error
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special) # set --> frozenset

            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())
        
        return self.encode_special(text, allowed_special)

    
    def decode(self, tokens:t.List[int], errors: str = "replace") -> str:
        assert hasattr(self, '_vocab') # special_tokens / invers_special_tokens 会随着 _vocab build 而生成

        parts = []
        for idx in tokens:
            if idx in self._vocab:
                parts.append( self._vocab[idx] ) # append bytes
            elif idx in self.inverse_special_tokens:
                parts.append( self.inverse_special_tokens[idx].encode('utf-8') ) # append bytes
            else:
                raise ValueError(f'invalid index {idx} out-of-vocab')
        concat_bytes = b''.join( parts )
        # 容错：错误的字节序列使用一个 replacement char 来替代
        return concat_bytes.decode('utf-8', errors=errors)
    

    def save(self, f_path):
        '''
        保存 name
        保存 pat_str
        保存 special_marks
        保存 merge_ranks (只需要保存 keys 即可，因为 values 是从 256 的递增序列)
        ---> 即保存了一个 tokenizer 的全部信息
        '''
        assert f_path.endswith(".tok")
        with open(f_path, 'w') as f:
            # write the name of the tokenizer as version
            f.write(f"{self.name}\n")
            # write the split pattern
            f.write(f"{self.pat_str}\n")
            # write the special_marks: first line number of special marks, then each line with a special mark
            f.write(f"{len(self._special_marks)}\n")
            for mark in self._special_marks:
                f.write(f"{mark}\n")
            # write the merge_ranks' keys
            for L, R in self._merge_ranks:
                f.write(f"{L} {R}\n")

    
    def load(self, f_path):
        '''
        读取 name: line 1
        读取 pat_str: line 2
        读取 num_of_special_marks: line 3
        读取接下来 num_of_special_marks 行 --> special marks
        按序读取剩下所有行: pair tokens, 依次存入 merge_ranks[pair tokens] = 256 ++
        构建 register special tokens / vocab
        '''
        assert f_path.endswith(".tok")
        # read .tok file
        self._special_marks = []
        self._merge_ranks = {}
        with open(f_path, 'r', encoding='utf-8') as f:
            # line 1: the name of the tokenizer as version
            self.name = f.readline().strip()
            # line 2: the split pattern
            self.pat_str = f.readline().strip()
            # line 3: the num_special
            num_special = int(f.readline().strip())
            # at line 4, next num_special lines
            for _ in range(num_special):
                self._special_marks.append( f.readline().strip() )
            # all remained lines as pair merged, if exists: split and store them in order
            for i, line in enumerate(f):
                L, R = map(int, line.split())
                self._merge_ranks[(L, R)] = 256 + i # i 从 0 开始, rank 从 256 开始
        
        # 构建 vocab: token_ID --> bytes
        self._build_vocab()

        # 注册 special_tokens
        self._register_special_tokens()

    
    def view(self, tmpsave_dir):
        # _vocab: int(0 至 MAX_MERGE_RANK) --> bytes
        # _merge_ranks: (int, int) --> merged_int(256 至 MAX_MERGE_RANK)
        # special_tokens: (str, int)
        assert hasattr(self, "_vocab") and hasattr(self, "_merge_ranks") and hasattr(self, "special_tokens")
        reverse_merge = {v:k for k, v in self._merge_ranks.items()} # merged_int --> (int, int)

        from .text_preprocess import render_bytes
        import os

        with open(os.path.join(tmpsave_dir, f'tmp_{self.name}.vocab'), 'w', encoding='utf-8') as f:
            # 首先打印 special marks:
            for mark, idx in self.special_tokens.items():
                f.write(f"[{mark}] {idx}\n")

            for idx, token in self._vocab.items():
                s = render_bytes(token)
                if idx < 256:
                    f.write(f"[{s}] {idx}\n")
                else:
                    L, R = reverse_merge[idx]
                    s_L, s_R = render_bytes(self._vocab[L]), render_bytes(self._vocab[R])
                    f.write(f"[{s_L}] {L} , [{s_R}] {R} -> [{s}] {idx}\n")


    @property
    def vocab_size(self) -> int:
        return self.explicit_n_vocab
    

    @property
    def eot_token(self) -> int:
        return self.special_tokens[ENDOFTEXT]


    @functools.cached_property
    def special_marks_set(self) -> set[str]:
        return set(self.special_tokens.keys())
    

    def is_special_token(self, token: int) -> bool:
        assert isinstance(token, int)
        return token in self.inverse_special_tokens
    












import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from collections import defaultdict
from multiprocessing import get_context, Manager
from ..file.parquet_io import yield_parquet_batch, concate_parquet_files
from ..file.folder_op import clean_folder
from ...design.stream_outline import stream_parallel_process_with_pending




def count_pair_batch(tokens_offsets_border):
    '''
    对一个 batch 统计 pair-counts: 返回一个shape为(N, 3)的np.ndarray for pair-counts.
    3列分别是 L, R, counts. 其中 L, R 作为pair, dtype是uint16 确定. counts dtype uint64

    Args:
        tokens_offsets_border: tokens_flat: uint16, offsets: int64, b_order: int
    '''
    (tokens_flat, offsets), b_order = tokens_offsets_border

    mask = np.full(shape=(len(tokens_flat),), fill_value=True)
    chunk_ends_ = (offsets-1)[1:] # 每个chunk末尾token在flat中的index
    chunk_starts_ = offsets[:-1] # 每个chunk开头token在flat中的index
    # ends_ == starts_ 的，说明chunk长度为1, 不需要统计paircounts. 略过
    # 把这些 id 指向的位置设为 False
    _where_equal_ = chunk_ends_ == chunk_starts_
    mask[ chunk_ends_[_where_equal_] ] = False
    
    mask_cp = mask.copy()

    # 提取所有非chunk end的tokens
    mask[chunk_ends_] = False
    L_tokens_flat = tokens_flat[mask] # 保持dtype=uint16, 可以为空
    
    # 提取所有非chunk start的tokens
    mask_cp[chunk_starts_] = False
    R_tokens_flat = tokens_flat[mask_cp] # 保持dtype=uint16, 可以为空

    # 构建L_tokens_flat, R_tokens_flat作为列构建的(N, 2)的pairs 2darray
    pairs = np.stack([L_tokens_flat.astype(np.uint16), R_tokens_flat.astype(np.uint16)], axis=1) # 可以为空

    # 构建新的dtype, 使得每一行L_token,R_token作为一个元素
    pair_dtype = np.dtype([('L', np.uint16), ('R', np.uint16)])
    structured = pairs.view(pair_dtype).squeeze()

    # 聚合计算频数
    uniq_pairs, counts = np.unique(structured, return_counts=True)

    # (N, 3)array as (L, R, counts), dtype分别是(uint16, uint16, uint64)
    pcounts = (uniq_pairs['L'], uniq_pairs['R'], counts.astype(np.uint64))

    return pcounts, b_order # pcounts:tuple of 3 arrays (L,R,counts), dtype(uint16, uint16, uint64)



# merge_pair 如果是一个 C/C++ 封装的接口，那么传参 tokens 时, 会遍历整个list并拷贝。
# 解决办法是把 token data 用numpy.array.data 的方式传入. 这种方式只传数组指针, 不拷贝.
# 为了统一接口, 在这里先提供一份 tokens_batch（np.ndarray）以及其他输入输出的版本
def merge_pair_batch_memcontiguous(
        tokens_offsets_border: object,
        pair_L:np.uint16,
        pair_R:np.uint16,
        new_token:np.uint16,
        ) -> tuple[np.ndarray, np.ndarray]:
    # tokens_flat:np.ndarray, # np.ndarray of uint16
    # offsets:np.ndarray, # np.ndarray of int64
    # e.g, tokens_flat: [1, 2, 3, 4, 5], offsets: [0, 1, 1, 3, 5] --> [1], [], [2, 3], [4,5]
    # tokens_lens: [1, 0, 2, 2]
    (tokens_flat, offsets), b_order = tokens_offsets_border
    tokens_lens = [j-i for i, j in zip(offsets, offsets[1:])]

    output_tokens_lens = list(tokens_lens) # 复制 tokens_lens
    output_tokens_flat = np.zeros_like(tokens_flat, dtype=np.uint16) # output tokens flat 只会比 token flat 短

    num_merges = 0
    for i in range( len(tokens_lens) ): # len(offsets)-1 == len(tokens_lens)
        # 从 tokens_flat 中slice出 tokens
        tokens = tokens_flat[offsets[i]:offsets[i+1]]
        # 遍历 tokens, 看里面是否出现了 pair0 pair1
        len_tokens, j = tokens_lens[i], 0
        while j < len_tokens:
            if j < len_tokens-1 and tokens[j] == pair_L and tokens[j+1] == pair_R:
                output_tokens_lens[i] -= 1 # 如果出现pair, 该tokens要发生一次merge, 长度-1
                output_tokens_flat[offsets[i]+j-num_merges] = new_token
                j += 2
                num_merges += 1
            else:
                output_tokens_flat[offsets[i]+j-num_merges] = tokens[j]
                j += 1
    
    output_offsets = np.array([0]+output_tokens_lens, dtype=np.int64).cumsum()

    return (output_tokens_flat[:output_offsets[-1]], output_offsets), b_order





def merge_pair_batch_parallel(
        tokens_offsets_border: object,
        pair_L:np.uint16,
        pair_R:np.uint16,
        new_token:np.uint16,
        ) -> tuple[np.ndarray, np.ndarray]:
    # tokens_flat:np.ndarray, # np.ndarray of uint16
    # offsets:np.ndarray, # np.ndarray of int64
    # e.g, tokens_flat: [1, 2, 3, 4, 5], offsets: [0, 1, 1, 3, 5] --> [1], [], [2, 3], [4,5]
    # tokens_lens: [1, 0, 2, 2]
    (tokens_flat, offsets), b_order = tokens_offsets_border
    tokens_lens = [j-i for i, j in zip(offsets, offsets[1:])]

    output_tokens_lens = list(tokens_lens) # 复制 tokens_lens
    output_tokens_flat = np.zeros_like(tokens_flat, dtype=np.uint16) # output tokens flat 只会比 token flat 短
    output_filter = np.zeros_like(tokens_flat, dtype=np.bool) # 从 output_tokens_flat 中 filter 出 output 的 mask
    
    # can parallel-program to speed upon loop i: thread-secure
    for i in range( len(tokens_lens) ): # len(offsets)-1 == len(tokens_lens)
        # 从 tokens_flat 中slice出 tokens
        tokens = tokens_flat[offsets[i]:offsets[i+1]]
        # 遍历 tokens, 看里面是否出现了 pair0 pair1
        len_tokens, j = tokens_lens[i], 0
        while j < len_tokens:
            if j < len_tokens-1 and tokens[j] == pair_L and tokens[j+1] == pair_R:
                output_tokens_lens[i] -= 1 # 如果出现pair, 该tokens要发生一次merge, 长度-1
                output_tokens_flat[offsets[i]+j] = new_token
                output_filter[offsets[i]+j] = True
                j += 2
            else:
                output_tokens_flat[offsets[i]+j] = tokens[j]
                output_filter[offsets[i]+j] = True
                j += 1
    
    output_offsets = np.array([0]+output_tokens_lens, dtype=np.int64).cumsum()

    return (output_tokens_flat[output_filter], output_offsets), b_order







def raise_continue_num_merges_conflict(num_merged, num_total_merges, continue_num_merges):
    raise ValueError(
        f'continue_num_merges plus loaded merge_ranks size must not exceed num_merges derived from `explicit_n_vocab`.\n'
        f'merge times derived from `explicit_n_vocab` shall be `explicit_n_vocab`-num_specials-256 = {num_total_merges}.\n'
        f'now loaded merge_ranks size {num_merged} + `continue_num_merges`{continue_num_merges} = '
        f'{num_merged+continue_num_merges} which exceeds {num_total_merges}.'
        )



class bufferBBPETokenizer(baseBBPETokenizer):
    '''
    get_pair_counts 和 merge_pair 两个 work on tokens(list of integers) 的原子操作, 在超长的 chunks of tokens 上并发执行
    的收益非常低。由于 tokens 一般被切得很小, 故 这两个原子操作的计算密度不大, 而超长的 chunks of tokens 的并发数量太大，并发
    带来的开销完全抵消了其提升。
    真正的瓶颈在于 stored_tokens(暂存的tokens以merge top pair) 导致的内存瓶颈。超长的 stored_tokens 是 list of tokens, 长
    度非常长，单机内存很可能放不下。 应该首先处理这个内存瓶颈。-----> buffer it

    由于有中间结果可以和 merge_ranks 相互校对, 所以buffer之后存在一个"续训"的概念: 从init/load到的tok, 检查它的merge_ranks状
    态是否符合buffer_tokens文件。若符合, 则从buffer_tokens续train
    
    最大的中间结果: 全局 pair-counts, 一张形状为(N, 3)的表格. 每行三个元素分别是L_token, R_token, counts
    首先分析一行占用空间:
    考虑一个大小为 x GB的parquet语料文件, pq压缩率大概在2-10, 那么算它是一个 10x GB左右的文本文件.
    也即最多 10x G个token(单字节一个token). 从而pair-counts的counts(出现频数)最多就是10x G= x 100亿 左右. uint32(最大值43亿)
    很可能存不下, 所以存储 counts 统一用 uint64就好. 正好聚合计算后, 一般都用 uint64 表示聚合计算的结果.
    tokens ID而言, uint16值范围是0-65536, 足够覆盖小的tokenizer了。2个bytes最多覆盖65535(uint16上限).
        经典例子, GPT2的词表大小(不包含special marks)是5W+256=50256. 
    vocab_size(不包含special marks)不超过65535的tok, 其token用2个字节(uint16)就可以表达.
    总结一下:
        一个counts占据 8(uint64)字节.
        两个tokens占据 4(uint16, for 6W 词表大小以下) 或 8(uint32, for 6W 词表大小以上)字节.
    小词表 --> 每行12字节.   大词表 --> 每行16字节
        
    然后分析总行数N for merge epoch e:
    考虑 e 时刻状态下的语料: 长度为L(个tokens), 词表大小为V(e=V-256), 它生成的 pair-tokens counts总行数, 有两个上限1: L, 上限2: V^2
    两个上限共同起作用:
        初期V^2小, L大, N 主要由 V^2 限制
        随着merge进行, L逐渐降低(以衰减的速率), V^2逐渐增大(以2次的速率增大), 后期 N 主要由 L 限制
    
    对于小词表而言, pair-count的 V^2 上限大概是 6W^2 = 36亿行, 每行12字节 ----> 40GB
    '''
    token_dtype = pa.uint16()

    p_counts_schema = pa.schema([
        pa.field('L', token_dtype),
        pa.field('R', token_dtype),
        pa.field('counts', pa.uint64()),
        ])
    
    
    tokens_schema = pa.schema([
        pa.field( 'tokens', pa.large_list(pa.field('token', token_dtype)) ),
        ])


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @classmethod
    def text_to_tokens_pa_table(cls, pre_split_pat, text):
        if not text.endswith(ENDOFTEXT):
            text = text + ENDOFTEXT
        chunks_str = re.findall(pre_split_pat, text) # list of tokens(string)
        
        # list of list of integers(every list of integers as tokens)
        chunks_tokens = [encode_to_ints(chunk) for chunk in chunks_str]
        
        # 创建 pa table
        batch_table = pa.Table.from_pydict({cls.tokens_schema[0].name: chunks_tokens}, cls.tokens_schema)
        return batch_table


    def _set_config(self, buffer_size, fc_count_pair_batch:t.Callable, fc_merge_pair_batch:t.Callable):

        os.makedirs(self._buffer_dir, exist_ok=True)
        
        buffer_tokens_dir = os.path.join(self._buffer_dir, 'tokens')
        os.makedirs(buffer_tokens_dir, exist_ok=True)
        
        buffer_pcounts_dir = os.path.join(self._buffer_dir, 'p_counts')
        os.makedirs(buffer_pcounts_dir, exist_ok=True)

        self._buffer_tokens_dir = buffer_tokens_dir
        self._buffer_pcounts_dir = buffer_pcounts_dir

        self._buffer_size = buffer_size
        self._func_merge_pair_batch = fc_merge_pair_batch
        self._func_count_pair_batch = fc_count_pair_batch


    def _init_tokens(self, corpora:str|t.List[str], text_colnames:t.List[None|str], extra_save_dir:str|None):
        '''
        如果 extra_save_dir 是path str, 且目录不为空, 那么copy extra_save_dir里的文件到 _buffer_dir/tokens/0
        如果 extra_save_dir 是None,
        生成 _buffer_dir/tokens/0 目录, 并在其中生成 corpora 的 byte-tokens .pq文件们

        corpora: list of string or parquet file paths
        text_colnames: corresponding column names for parquet files. place None for string
            e.g.,
            corpora: ['aaabbc', '../data/raw/train.pq', '../data/raw/valid.pq']
            text_colnames: [None, 'text', 'text']
        
        the resulting _buffer_dir/tokens/0 & extra_save_dir(if any) would be like as:
        [   /buffer_dir/tokens/0/{name1}.pq
            /buffer_dir/tokens/0/{name2}.pq
                      ...
            /buffer_dir/tokens/0/{name_}.pq   ]

        step1: save string corpus as .pq file with column 'text' under _buffer_dir
        step2: byte-level tokenize all parquet corpus files into .pq files with column 'tokens'
               under _buffer_dir/tokens/0 & extra_save_dir(if any)

        .pq file generating:
        从 corpus 中读取 batch of text, join/split 成 chunks 之后, 把每个 chunk encode 成 tokens(list of integers)
        把 tokens 写入到位于 buffer_dir 的 Parquet 文件。per tokens-chunk per row

        tokens-chunk 是 ListArray 类型 --> list of uint16 named 'token' --> list_of_ints_type
            变长的 List Array, field 是 token, dtype 为 uint16(token的最大vocab size 十几万即可, 目前这个小tok uint16足够)
        
        Parquet table 的 schema:
            列名: tokens
            column type: list_of_ints_type, which is pa.list_(pa.field('token', pa.uint16()))
        '''
        # create and clean starting directory: _buffer_dir/tokens/0
        tokens_dir_0 = os.path.join(self._buffer_tokens_dir, '0')
        os.makedirs( tokens_dir_0, exist_ok=True)
        clean_folder( tokens_dir_0 )

        # 如果输入了 extra_save_dir 且 extra_save_dir 不为空, 就从 extra_save_dir 拷贝到 _buffer_dir/tokens/0
        if extra_save_dir and os.listdir(extra_save_dir):
            if len(os.listdir(extra_save_dir)) == len(corpora):
                import shutil
                shutil.copytree(extra_save_dir, tokens_dir_0, dirs_exist_ok=True)
                return
            else:
                raise ValueError(
                    f'extra_save_dir {extra_save_dir} conflicts with {len(corpora)} corpora'
                    )

        # 如果没有输入 extra_save_dir 亦或 extra_save_dir 为空. 从 corpora 生产 init tokens pq files
        assert len(corpora) == len(text_colnames), f"length of corpora must match with length of columns"
        corpus_col_pair = []
        for i, corpus in enumerate(corpora):
            if not corpus.endswith('.parquet'):
                corpus_pq_schema = pa.schema([ pa.field('text', pa.string()) ])
                # overwrite if exists
                corpus_pq = os.path.join(self._buffer_dir, f'corpus_{i}.parquet')
                with pq.ParquetWriter(corpus_pq, corpus_pq_schema) as writer:
                    writer.write_table(
                        pa.Table.from_pydict({"text":[corpus]}, corpus_pq_schema)
                        )
                corpus_col_pair.append( (corpus_pq, 'text') )
            elif os.path.exists(corpus) and text_colnames[i]:
                corpus_col_pair.append( (corpus, text_colnames[i]) )
            else:
                raise FileNotFoundError(
                    f'corpus parquet file {corpus} column for text {text_colnames[i]} not exiting')

        # create corresponding tokens .pq files
        init_tokens_pqs = []
        for corpus_pq, _ in corpus_col_pair:
            init_tokens_pq = os.path.join(self._buffer_tokens_dir, '0', os.path.basename(corpus_pq))
            if init_tokens_pq in init_tokens_pqs:
                raise FileExistsError(
                    f'input corpus parquet files cannot have duplicated file basename.\n'
                    f'here {init_tokens_pq} duplicated.\n'
                    f'change the basename of input corpus file of parquet format, because the program '
                    f'generated same name for previous corpus of string format.')
            else:
                init_tokens_pqs.append(init_tokens_pq)
        
        for (corpus_pq, text_col), init_tokens_pq in zip(corpus_col_pair, init_tokens_pqs):
            corpus_batch_iter = yield_parquet_batch(corpus_pq, 8192*32, [text_col])
            with pq.ParquetWriter(init_tokens_pq, self.tokens_schema) as writer:
                for k, batch in enumerate(corpus_batch_iter):
                    text = ENDOFTEXT.join( batch[text_col].to_pylist() )
                    # 创建 pa table
                    batch_table = self.text_to_tokens_pa_table(self.pat_str, text)
                    writer.write_table(batch_table)
        
        # clean corpus parquet file from string input
        clean_folder(self._buffer_dir, method='only_file')

        # copy init_tokens file to extra_save_dir if it is an empty dir
        if extra_save_dir and os.path.isdir(extra_save_dir) and os.listdir(extra_save_dir) == []:
            import shutil
            shutil.copytree(tokens_dir_0, extra_save_dir, dirs_exist_ok=True)


    def _prepare_train(self, num_merges, executor, *args, **kwargs) -> str:
        '''
        检查是否满足训练条件, 返回与当前_merge_ranks匹配的 本次训练的开始点 _buffer_tokens_dir/?
        对于 size = S 的 _merge_ranks 来说, 匹配的本次训练开始点是 _buffer_tokens_dir/S
        '''
        # 检查 num_merges 和 explicit_n_vocabs / merge_ranks_size 的冲突. 确定 num_train_epochs
        super()._prepare_train(num_merges)
        # 确保 _buffer_tokens_dir 不为空, 在里面选取一个作为 训练起始点
        assert os.listdir(self._buffer_tokens_dir), f'empty buffer dir for tokens {self._buffer_tokens_dir}'

        # 由于 bufferTokenizer 存在一个续train的概念, 要检查 _buffer_dir_tokens 和 merge_ranks 的状态
        # 对于 merge_ranks size = 0：buffer_dir_tokens/0 不为空，即可开始训练
        if len(self._merge_ranks) == 0:
            tokens_dir_0 = os.path.join(self._buffer_tokens_dir, '0')
            assert os.listdir(tokens_dir_0)
            return tokens_dir_0
        
        # 对于 merge_ranks size = s > 0：训练 next token, 即 merge_ranks s+1 的素材是 buffer_dir_tokens/s
        latest = max([int(f) for f in os.listdir(self._buffer_tokens_dir)])
        tokens_dir_latest = os.path.join(self._buffer_tokens_dir, f'{latest}')

        if latest == len(self._merge_ranks):
            return tokens_dir_latest
        
        elif latest + 1 == len(self._merge_ranks): # merge_ranks 不会是空
            # 取出 merge_ranks 的最后一对
            to_merge_pair, new_token = max( self._merge_ranks.items(), key=lambda item: item[1] )
            return self._next_tokens_dir(tokens_dir_latest, to_merge_pair, new_token, executor)
        
        else:
            raise RuntimeError(
                f"current merge_ranks size {len(self._merge_ranks)} not match with buffer_tokens_dir "
                f"{self._buffer_tokens_dir}'s latest buffer tokens {tokens_dir_latest}, merge_ranks "
                f"size shall be equal to latest dir, or latest dir + 1")


    @classmethod
    def yield_tokens_offsets_order(cls, yield_batch: t.Generator):
        for i, batch in enumerate(yield_batch):
            # batch['tokens'] --> pa.LargeListArray
            # pa.LargeListArray .values --> 得到原类型array的数据; .offsets --> 得到 int64array 的偏移量
            tokens_flat = batch[cls.tokens_schema[0].name].values.to_numpy()
            offsets = batch[cls.tokens_schema[0].name].offsets.to_numpy()

            yield (tokens_flat, offsets), i




    @classmethod
    def _write_pcounts_batch(cls, pcounts_order, save_dir, src_tokens_fname, collector:list):
        '''
        buffer the pair-counts for the batch with 'order'
        '''
        b_pcounts, b_order = pcounts_order # b_pcounts: tuple of 3arrays(L,R,counts), (uint16, uint16, uint64)dtypes

        data = {
            cls.p_counts_schema[0].name: b_pcounts[0], # L: L_tokens
            cls.p_counts_schema[1].name: b_pcounts[1], # R: R_tokens
            cls.p_counts_schema[2].name: b_pcounts[2], # counts: counts
            }

        table = pa.Table.from_pydict(data, cls.p_counts_schema)

        if not table: # 如果是空table, 直接返回None不用写入
            return None
        
        save_path = os.path.join(save_dir, f'{src_tokens_fname}-part-{b_order:06d}.parquet')
        pq.write_table(table, save_path)

        del b_pcounts # 已经落盘了就可以删了
        collector.append(save_path)
    


    def _write_pcounts(self, tokens_pq, executor) -> list:
        # 从 tokens_pq 中解析出 rank 和 tokens_pq 名字
        rank, tokens_fname = tokens_pq.split('/')[-2:]

        # 创建 本次 rank 的 buffer_pcounts_dir
        pcounts_save_dir = os.path.join(self._buffer_pcounts_dir, rank)
        os.makedirs(pcounts_save_dir, exist_ok=True) 

        pcounts_paths = []
        yield_tokens:t.Generator = yield_parquet_batch(tokens_pq, self._buffer_size)
        # data_gen: (tokens_flat, offsets), i
        data_gen:t.Generator = self.yield_tokens_offsets_order(yield_tokens)

        stream_parallel_process_with_pending(
            executor,
            data_gen, # data_gen: (tokens_flat, offsets), i
            process_fn = self._func_count_pair_batch, # return (pcounts, b_order)
            result_handler = self._write_pcounts_batch, 
            max_pending = 8,
            result_handler_args = (pcounts_save_dir, tokens_fname, pcounts_paths)
        )

        return pcounts_paths
    

    @classmethod
    def _aggregate_pcounts(cls, pcounts_files):

        tables = [pq.read_table(f) for f in pcounts_files]
        concatenation = pa.concat_tables(tables)

        L, R, counts = cls.p_counts_schema[0].name, cls.p_counts_schema[1].name, cls.p_counts_schema[2].name
        
        # group sum 会提升精度防止溢出, counts_sum dtype=uint64
        agg_pcounts = concatenation.group_by([L, R]).aggregate([(counts, 'sum')]) # counts 列 --> counts_sum 列

        return agg_pcounts, L, R, '_'.join([counts, 'sum'])


    @staticmethod
    def get_occur_most_info(agg_pcounts, L_col, R_col, agg_pcounts_col):

        max_occur = pc.max(agg_pcounts[agg_pcounts_col]).as_py()
        filter_mask = pc.equal(agg_pcounts[agg_pcounts_col], max_occur)

        _row = agg_pcounts.filter(filter_mask).slice(0, 1)

        occur_most_pair: tuple[int, int] = (_row[L_col][0].as_py(), _row[R_col][0].as_py())

        return occur_most_pair, max_occur



    @classmethod
    def _write_merge_batch(cls, tokens_offsets_border, merge_func, L, R, new_token, save_dir, src_fname):
        # tokens_offsets: tuple of tokens_flat: uint16 ndarray, offsets: int64 ndarray
        # b_order: 批次batch的序号
        # merge_fun: 执行 merge pair的函数. L, R, new_token: uint16
        # save_dir: 保存的目录, src_fname: 批次batch的来源文件名

        # valid 指已经剔除掉 merge 之后长度为 1 的chunk of tokens
        (valid_merged_tokens_flat, valid_merged_offsets), b_order = merge_func(
            tokens_offsets_border,
            L, R, new_token)
        
        merged_tokens = pa.ListArray.from_arrays(valid_merged_offsets, valid_merged_tokens_flat)
        batch = pa.RecordBatch.from_pydict({cls.tokens_schema[0].name: merged_tokens}, cls.tokens_schema)
        table = pa.Table.from_batches([batch], schema=cls.tokens_schema)
        
        merged_save_path = os.path.join(save_dir, f'{src_fname}-part-{b_order:06d}.parquet')
        pq.write_table(table, merged_save_path)

        return merged_save_path

    
    @classmethod
    def _merge_tokens_save(
            cls,
            save_dir,
            to_merge_pair,
            new_token,
            tokens_pq,
            fc_merge:t.Callable,
            buffer_size,
            executor
            ):
        '''
        given tokens parquet file `tokens_pq`,
        merge `to_merge_pair` tokens to `new_token` inside every tokens chunk,
        then save result tokens chunks into a same-file-name parquet file to `save_dir`

        读(yield_tokens)
        算(_thrd_process_tokens_batch)
        写(writer.write_batch to save_dir)
        batches(buffer_size确定batch大小)
        '''
        src_fname = os.path.basename(tokens_pq)
        L, R = map(cls.token_dtype.to_pandas_dtype(), to_merge_pair)
        new_token = cls.token_dtype.to_pandas_dtype()(new_token)

        # 遍历读取当前 tokens_pq
        yield_tokens:t.Generator = yield_parquet_batch(tokens_pq, buffer_size)
        # data_gen yields: (tokens_flat:uint16 array, offsets:int64 array), i
        data_gen:t.Generator = cls.yield_tokens_offsets_order(yield_tokens)

        merged_save_path_collector = []
        # ParquetWriter 并发写入同一不安全, 可能会造成文件损坏。故这里采用多进程分片写入-统一合并处理
        stream_parallel_process_with_pending(
            executor,
            data_gen, # data_gen: (tokens_flat, offsets), i
            process_fn = cls._write_merge_batch, # return merged_save_path
            result_handler = lambda merged_save_path: merged_save_path_collector.append(merged_save_path), 
            max_pending = 8,
            process_args = (fc_merge, L, R, new_token, save_dir, src_fname)
            )
        
        concate_parquet_files(merged_save_path_collector, os.path.join(save_dir, src_fname), clean=True)


    def _next_tokens_dir(
            self,
            tokens_dir_this:str,
            occur_most_pair:t.Tuple,
            new_token:int,
            executor
            ) -> str:
        # 在计算获得 tokens parquet for next merge_rank 时, 当前 tokens_pq 已经提炼出了 merge_info
        # 并更新了 tokenizer._merge_rank，使得其 + 1。所以 cur_dir_for_tokens_pq = len(_merge_rank) - 1
        # next_dir_for_tokens_pq = cur_dir_for_tokens_pq + 1 = len(cur_merge_rank)
        next_rank = len(self._merge_ranks)
        assert next_rank == int( os.path.basename(tokens_dir_this) ) + 1
        
        tokens_dir_next = os.path.join(self._buffer_tokens_dir, f'{next_rank}')
        os.makedirs( tokens_dir_next, exist_ok=True)

        for f in os.listdir(tokens_dir_this):
            tokens_pq = os.path.join(tokens_dir_this, f)

            self._merge_tokens_save(
                save_dir = tokens_dir_next,
                to_merge_pair = occur_most_pair, 
                new_token = new_token,
                tokens_pq = tokens_pq,
                fc_merge = self._func_merge_pair_batch,
                buffer_size = self._buffer_size,
                executor = executor
                )
        
        return tokens_dir_next # update



    def _get_merge_info(self, tokens_dir, executor):
        # 从当前 tokens_dir, 统计得到 next merge pair 和 new token
        num_merged_epochs = len(self._merge_ranks)
        assert num_merged_epochs == int(os.path.basename(tokens_dir))

        # compute pair_counts for every batch of tokens parquet into p_counts parquet, and collect them
        b_pcounts_pqs = []
        for f in os.listdir(tokens_dir):
            tokens_pq = os.path.join(tokens_dir, f)

            if not os.path.exists(tokens_pq):
                raise FileNotFoundError(
                    f'buffer parquet {tokens_pq} for merge epoch {num_merged_epochs+1} not found'
                    )
            b_pcounts_pqs.extend( self._write_pcounts(tokens_pq, executor) )
        
        if not any(b_pcounts_pqs):
            # 在raise error前先保存已经train好的tokenizer
            self.save(os.path.join(self._buffer_dir, f'cache_{self.name}.tok'))

            raise_run_out_corpus_error(num_merged_epochs, len(self._special_marks))

        # aggregate pair counts to agg_p_counts(pa.Tabel)
        # obtain the pair with most occurrence
        occur_most_pair, max_occurence = self.get_occur_most_info(*self._aggregate_pcounts(b_pcounts_pqs))

        return occur_most_pair, max_occurence


    def _train_loop(self, tokens_dir_start:str, start:int, end:int, executor, keep_window:int, verbose:bool):
        '''
        merge_rank 遍历 start --> end(不含) 地 BPE train 循环.
        tokens_dir_start 是起始 tokens 文件夹, buffer_dir_tokens/tokens/start
        
        merge_rank 是此次merge相对第1次merge的偏移, 比如第1次merge的merge_rank=0
        merge_rank + 256是此次merge生成的new_token. 
        此次merge完成后, merge_ranks 的size = merge_rank of final merge + 1
        由此, 在 train_loop 开始时, merge_rank = start, merge_ranks size = start+1
        在 train_loop 结束时, merge_rank = end-1, merge_ranks size = end
        '''

        tokens_dir_this = tokens_dir_start
        for rank in range(start, end):
            print(f'merge rank {rank} / {start} to {end-1}')
            try:
                top_pair, max_occurence = self._get_merge_info(tokens_dir_this, executor)
                new_token, occurence = rank + 256, max_occurence if verbose else None

                # update tokenizer: len(self._merge_rank) += 1
                self._update_tokenizer(top_pair, new_token, occurence)

                # rank = i = len(self._merge_rank) - 1 == self._num_merges - 1:
                if rank == end - 1:
                    break

                tokens_dir_this = self._next_tokens_dir(tokens_dir_this, top_pair, new_token, executor)
                
            finally:
                # keep the init and `keep_window` tokens/p_counts parquet file
                to_remove = rank - keep_window
                if to_remove > 0:
                    clean_folder( os.path.join(self._buffer_tokens_dir, f'{to_remove}') )
                    clean_folder( os.path.join(self._buffer_pcounts_dir, f'{to_remove}') )



    def train_bpe(self,
                  num_merges:int|None = None, # global num merges fir the tokenizer
                  *,
                  corpora:t.List[str]|str|None,
                  colnames:t.List[str|None]|None = None,
                  backup_init_tokens_dir:str|None = None, # backup the init tokens files of corpus
                  buffer_size:int = 172470436, # max num of token-chunks in memory for each core. 0.16GB
                  keep_window:int = 3, # max reserved tokens_pq file in disk
                  fc_count_pair_batch:t.Callable = count_pair_batch,
                  fc_merge_pair_batch:t.Callable = merge_pair_batch_memcontiguous,
                  verbose:bool = False
                  ):
        
        assert keep_window >= 0
        if isinstance(corpora, str):
            corpora = [corpora]
            colnames = [None]

        self._set_config(buffer_size, fc_count_pair_batch, fc_merge_pair_batch)

        # corpora 为 t.List[str], 模式是 从头train
        # backup_init_tokens_dir如果是空文件夹，那么生成init tokens后在这里保存一份
        # backup_init_tokens_dir如果有和corpora等长的文件个数，那么从backup_init_tokens_dir读取init tokens
        if corpora is not None:
            self._clear()
            self._init_tokens(corpora, colnames, backup_init_tokens_dir)
        # corpora 为 None 时, 模式是 续train.
        # 续train需要满足的条件会由 _prepair_train 检查或满足. 
        else:
            self._build_vocab()

        with ProcessPoolExecutor(os.cpu_count()) as executor:
            # _prepare_train 检查 num_merges 和 explicit_n_vocabs / merge_ranks_size 的冲突
            # 确定 num_train_epochs, 检查 buffer_dir_tokens 和 merge_ranks 是否匹配. 返回匹配的训练起点文件夹
            tokens_dir_start = self._prepare_train(num_merges, executor)
            
            start, end = len(self._merge_ranks), len(self._merge_ranks) + self._num_train_epochs

            self._train_loop(tokens_dir_start, start, end, executor, keep_window, verbose)

        # set down others
        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()



    def extend_bpe(self, *args, **kwargs):
        assert hasattr(self, '_merge_ranks'), f'tokenizer not load.'
        if not self._merge_ranks:
            raise RuntimeError(f'merge_ranks empty. should run `train_bpe`.')
        #TODO    
        raise NotImplementedError(f'tokenizer extend not implemented')












import multiprocessing as mp
import atexit
from multiprocessing.util import Finalize

import pair_count_merge


def _worker_init(block_size: int):
    # 子进程启动时, 执行 cython 包里的 initialize
    pair_count_merge.initialize(block_size)

    # 注册 进程退出时的清理程序
    Finalize(None, pair_count_merge.close, exitpriority=10)
    atexit.register(pair_count_merge.close)





class boostBBPETokenizer(bufferBBPETokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train_bpe(self,
                  num_merges:int|None = None,               # global num merges for the tokenizer
                  *,
                  corpora:t.List[str]|str|None,             # None 代表从 buffer_dir 续train; 可以是直接输入文本str; 可以是parquet路径; 可以混合
                  colnames:t.List[str|None]|None = None,    # 和 corpora 对应, 指定 parquet 文件里 文本 列名
                  backup_init_tokens_dir:str|None = None,   # backup the init tokens files of corpus
                  buffer_size:int = 172470436,              # max num of token-chunks in memory for each core. 0.16GB
                  keep_window:int = 3,                      # max reserved tokens_pq file in disk
                  verbose:bool = False
                  ):
        
        assert keep_window >= 0
        if isinstance(corpora, str):
            corpora = [corpora]
            colnames = [None]
        
        self._set_config(
            buffer_size = buffer_size,
            fc_count_pair_batch = pair_count_merge.count_pair_batch,
            fc_merge_pair_batch = pair_count_merge.merge_pair_batch)

        # corpora 为 t.List[str], 模式是 从头train
        # backup_init_tokens_dir如果是空文件夹，那么生成init tokens后在这里保存一份
        # backup_init_tokens_dir如果有和corpora等长的文件个数，那么从backup_init_tokens_dir读取init tokens
        if corpora is not None:
            self._clear()
            self._init_tokens(corpora, colnames, backup_init_tokens_dir)
        # corpora 为 None 时, 模式是 续train.
        # 续train需要满足的条件会由 _prepair_train 检查或满足. 
        else:
            self._build_vocab()

        ctx = mp.get_context('fork') # spawn方法使得 跨平台一致

        # 测算设定 block_size = 40 * buffer_size, 就使得最大块的内存需求落在同一个 block. 避免多次申请block.
        # 根据本机64GB内存，8核, 每核内存8GB, 分6.4GB内存给计算, 那么 buffer_size=0.16GB
        # 反推最佳 buffer_size = 0.16GB, 这样1个进程有1个内存block,占用6.4GB. 8核总共占据51.2GB.
        memblock_size = 40 * self._buffer_size

        with ProcessPoolExecutor(
            max_workers=8,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(memblock_size,)
        ) as executor:
            # 检查 num_merges 和 explicit_n_vocabs / merge_ranks_size 和 buffer_dir_tokens 是否匹配
            # 确定 _num_train_epochs, 返回匹配的训练起点文件夹 tokens_dir_start
            tokens_dir_start = self._prepare_train(num_merges, executor)
            
            start, end = len(self._merge_ranks), len(self._merge_ranks) + self._num_train_epochs

            self._train_loop(tokens_dir_start, start, end, executor, keep_window, verbose)

        # set down others
        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()












from ...design.async_outline import async_queue_get, async_queue_process, pipeline_producer_consumer
import asyncio






class asyncBBPETokenizer(boostBBPETokenizer):
    '''
    异步的生产者-消费这模式, 是通过一个queue连接, 但解耦了 生产数据 和 消费数据 两个步骤之间的依赖关系，使得
    生产和消费两个异步task可以自顾自运行.
    '''
    _MAX_QUEUE_SIZE = 10
    _NUM_COMPUTERS = 8
    _NUM_WRITERS = 1

    
    def _write_pcounts(self, tokens_pq, executor) -> list:
        # 从 tokens_pq 中解析出 rank 和 tokens_pq 名字
        rank, tokens_fname = tokens_pq.split('/')[-2:]

        # 创建 本次 rank 的 buffer_pcounts_dir
        pcounts_save_dir = os.path.join(self._buffer_pcounts_dir, rank)
        os.makedirs(pcounts_save_dir, exist_ok=True) 

        yield_tokens:t.Generator = yield_parquet_batch(tokens_pq, self._buffer_size)
        # data_gen: (tokens_flat, offsets), i
        data_gen:t.Generator = self.yield_tokens_offsets_order(yield_tokens)
        
        async def main():
            # 一个in-place改变状态的收集函数.
            pcounts_paths = []
            async def collector(pcounts_order):
                # 把落盘写到 collector 步骤中. 安全因为它 await 拿到process_fc 计算结果, 以及collector都在主线程中
                if pcounts_order is not None:
                    self._write_pcounts_batch(pcounts_order, pcounts_save_dir, tokens_fname, pcounts_paths)
                else:
                    pcounts_paths.append(None)
            
            await pipeline_producer_consumer(
                data_gen, # yield (tokens_flat, offsets), b_order
                self._func_count_pair_batch, # arg1:(tokens_flat, offsets), b_order;arg2: token_dtype --> (pcounts, b_order)
                executor,
                self._MAX_QUEUE_SIZE,
                1,
                collector, # arg: (pcounts, b_order) --> in-place change pcounts_paths
                )
            
            return pcounts_paths
        
        pcounts_paths = asyncio.run(main())
        return [path for path in pcounts_paths if path] # collector会收集None作为结束信号


    @classmethod
    def _write_merge_batch(cls, merged_tokens_offsets_order, save_dir, src_fname):
        (merged_tokens, merged_offsets), b_order = merged_tokens_offsets_order

        merged_tokens = pa.ListArray.from_arrays(merged_offsets, merged_tokens)
        batch = pa.RecordBatch.from_pydict({cls.tokens_schema[0].name: merged_tokens}, cls.tokens_schema)
        table = pa.Table.from_batches([batch], schema=cls.tokens_schema)

        merged_save_path = os.path.join(save_dir, f'{src_fname}-part-{b_order:06d}.parquet')
        pq.write_table(table, merged_save_path)

        return merged_save_path


    @classmethod
    def _merge_tokens_save(
            cls,
            save_dir,
            to_merge_pair,
            new_token,
            tokens_pq,
            fc_merge:t.Callable,
            buffer_size,
            executor,
            ):
        '''
        given tokens parquet file `tokens_pq`,
        merge `to_merge_pair` tokens to `new_token` inside every tokens chunk,
        then save result tokens chunks into a same-file-name parquet file to `save_dir`

        batches(buffer_size确定batch大小)
        异步:
        读(yield_tokens) as 生产者 ---> 队列1
        算(_thrd_process_tokens_batch) as 队列1 消费者, 队列2 生产者 ---> 队列2
        写(writer.write_batch to save_dir) as 队列2消费者
        '''
        fname = os.path.basename(tokens_pq)
        L, R = map(cls.token_dtype.to_pandas_dtype(), to_merge_pair)
        new_token = cls.token_dtype.to_pandas_dtype()(new_token)

        yield_batch:t.Generator = yield_parquet_batch(tokens_pq, buffer_size)
        # batch --> (tokens, offsets), order
        data_gen:t.Generator = cls.yield_tokens_offsets_order(yield_batch)

        async def main():
            # 在主线程接收 "读" 的result（tokens_offsets, b_order）到 queue1
            queue1 = asyncio.Queue(cls._MAX_QUEUE_SIZE)

            # 一个 read 任务即可
            read_task = asyncio.create_task(async_queue_get(data_gen, queue1))

            # 从queue1获取素材, 主线程跨进程pickle发送素材到进程池执行fc_merge_pair_batch
            # 然后在主线程跨进程pickle接收进程池"算"的result（merged_tokens_offsets, b_order），collector到queue2.
            queue2 = asyncio.Queue(cls._MAX_QUEUE_SIZE)
            async def compute_collector(result):
                await queue2.put(result)
            
            compute_tasks = [
                asyncio.create_task(
                    async_queue_process(queue1, executor, fc_merge, compute_collector, L, R, new_token)
                    )
                for _ in range(cls._NUM_COMPUTERS)]
            
            # 计算任务全部执行完之后, 要加一个None作为结束信号到queue2
            async def compute_end_tasks():
                await asyncio.gather(*compute_tasks)
                await compute_collector(None)

            # 在主线程接收 "写" 的result（path）到 written_parts. 只有主线程改变 written_parts，所以它安全
            written_parts = []
            async def write_collector(result):
                written_parts.append(result)
            
            # 单个 write 任务: 从 queue2 中得到素材, 送入进程池执行 cls._write_merge_batch，
            write_tasks = [
                asyncio.create_task(
                    async_queue_process(queue2, executor, cls._write_merge_batch, write_collector, save_dir, fname)
                    )
                for _ in range(cls._NUM_WRITERS)]

            await asyncio.gather(read_task, asyncio.create_task(compute_end_tasks()), *write_tasks)
            return written_parts
            
        written_parts = asyncio.run(main())

        concate_parquet_files(written_parts, os.path.join(save_dir, fname), clean=True)
