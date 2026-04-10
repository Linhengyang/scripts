import heapq
from collections import defaultdict, Counter
'''
Huggingface/Tokenizer:
step 0: 统计 词袋bag-of-word, 构建 序列unique_words 和 序列counts, 记录它们的 index 为 positions
step 1: 从 词袋bag-of-word 中统计token-pair的频数和倒排positions, 即 字典pair_counts{pair: int_counts} 和 字典where_to_update{pair: set_positions}
step 2: 构建 三元组(token-pair, p_cnts, positions), 分别指 token-pair, 其频数, 其倒排索引. 将 三元组 push 入一个 max_heap, 以频数排序. 清空 where_to_update
循环 num_merges 次 step 3:  --> 始终维持一个 绝对正确的pair频数统计 pair_counts, 从max_heap中取出顶端三元组, 将max-pair执行于 unique_words, 产出 changes 更新 pair_counts 和 where_to_update
    step 3.1: 从 max_heap 中 pop 出顶端node三元组, 检查其 p_cnts 是否与 pair_counts 符合. 如果不符合说明该pair的p_cnts被更新过, 更新三元组并重新push入 max_heap 重排序.
    step 3.2: 取得 有效的 最大p_cnts 三元组(token-pair, p_cnts, positions), 得到待合并tokens和new_token, 针对 positions 中的每一个 pos, 对该pos代表的 unique word 执行 pair-merge, 产出 changes
        即已有 待合并tokens(l_tok, r_tok)-->new_tok, 遍历 positions 的所有 pos ---> 对 当前 pos 的 unique_word, 产出 该 pos 对应的 changes:
            从 index = 0 开始扫描 unique_word 直到其末尾, 每当匹配到 一对(l_tok, r_tok) 时, 输出(l_tok, r_tok)-->nwe_tok与其前/后邻域 组成的 pair增减记录 到 changes
            实时 in-place 改动 unique_word, 扫描 index 自增 1
        ---> changes of the pos: 序列(changed_pair, change_signal)
    step 3.3: 将 changes of all positions 用于更新 pair_counts 和 where_to_update.
        changes of all positions: 对于某个pos, 其中每个 changed_pair, 其 changed_signal 代表了 该pair在该pos位置的unique_word中的增减信号
        pair_counts 更新:
            遍历 changes of all positions, 针对其中每个 changed_pair, changed_signal * counts[pos] 就是 单个pos 为该 changed_pair 贡献的 pair-counts 增量
            加总所有 pos of positions, 可以更新该 changed_pair 到正确频数.
        where_to_update 更新:
            遍历 changes of all positions, 取出 正 changed_signal 的 changed_pair, 正信号意味着该 changed_pair在该unique_word是新增的
            将 pos 增加到 where_to_update 中 该 changed_pair 对应的 positions 中. 如此意味着 where_to_update 收集了每个 new emerged pair 在 序列unique_words 中存在的最大可能位置索引候选集
    step 3.4: 将 where_to_update 中所有 new emerged pair & positions, 构建三元组(token-pair, p_cnts, positions), push入 max_heap. 清空 where_to_update.
        max_heap 自排序, 顶端将会是 旧pair和new emerged pair共同竞争出的当前最大p_cnts pair. ---> max_heap 是全周期内所有pair共同的竞技场, 除了被pop出去的胜者pair.
'''

class BBPETokenizer(baseBBPETokenizer):
    def train_bpe(self, corpora, num_merges = None, verbose = True, *args, **kwargs):
        self._clear()
        self._prepare_train(num_merges)

        if isinstance(corpora, str):
            corpora = [corpora]
        corpus = ENDOFTEXT.join( corpora )
        chunks_str: t.List[str] = re.findall(self.pat_str, corpus) # pre-split to list of string
        BoW = Counter(chunks_str)

        # bag-of-words <-- list of unique words, freqs, indices
        unique_words = [list( word.encode('utf-8') ) for word in BoW.keys()]
        freqs = list(BoW.values())

        # pair_counts <-- dict of pairs-counts: (l_tok, r_tok): int of cnts_of_key_pair 
        # where_to_update <-- dict of pairs-positions: (l_tok, r_tok): set of postions_of_key_pair
        pair_counts = defaultdict(int)
        where_to_update = defaultdict(set)
        for pos, word in enumerate(unique_words):
            for l_tok, r_tok in zip(word[:-1], word[1:]):
                # defaultdict of int: 当 (l_tok, r_tok) 不存在时, 插入该 key 并设置 default value 为 0 / set()
                pair_counts[(l_tok, r_tok)] += freqs[pos]
                where_to_update[(l_tok, r_tok)].add(pos)
        
        # heap: maxHeap to always pop out token-pairs with max counts along with its positions
        max_heap = []
        for tok_pair, positions in where_to_update.items():
            # node: (tokens_pair, counts_pair, positions_pair)
            to_merge_node = (tok_pair, pair_counts[tok_pair], positions)
            heapq.heappush(max_heap, (-to_merge_node[1], to_merge_node))
        
        # empty the where_to_update, waiting new emerged pairs to insert
        where_to_update = defaultdict(set)

        # merge_loop
        merge_cnts = 0
        while merge_cnts < self._num_train_epochs:
            try:
                _, (tok_pair, p_counts, positions) = heapq.heappop(max_heap)
            except IndexError:
                break

            if p_counts != pair_counts[tok_pair]:
                to_merge_node = (tok_pair, pair_counts[tok_pair], positions)
                heapq.heappush(max_heap, (-to_merge_node[1], to_merge_node))
                continue
            
            if p_counts < 1:
                break

            # BPE begin: we got tok_pair(l_tok, r_tok), positions(indices of unique words to execute merge)
            new_tok = 256 + merge_cnts
            l_tok, r_tok = tok_pair
            self._update_tokenizer((l_tok, r_tok), new_tok, p_counts if verbose else None)

            for pos in positions:
                # changes: records of emerged(+1)/vanished(-1) pairs triggered by mergeing l_tok, r_tok among this word at unique_words[pos]
                changes = [] # list of pair-signal: [(l_tok, r_tok), +-1]

                # merge specific word, e.g:
                # abcbc -> aXbc -> aXX
                # abcabc -> aXabc -> aXaX
                # bcdbca -> Xdbca -> XdXa
                word = unique_words[pos] # in-place update word, which in-place update unique_words
                i = 0
                while True:
                    # escape the loop when i reaches the last token(which means i+1 invalid)
                    if i >= len(word)-1:
                        break

                    # i+1 still valid
                    # the-pair found! when pair index i & i+1 match l_tok, r_tok.
                    if word[i] == l_tok and word[i+1] == r_tok:
                        # record the changes except the merging pair l_tok, r_tok: it was poped out from heap
                        # since the merging pair will no longer be used/encountered, no need to update its pair-counts to 0 --> no need to record in changes
                        # if prev_tok exists, pair(prev_tok, l_tok) vanished 1 times, pair(prev_tok, new_tok) emerged 1 times
                        if i > 0:
                            changes.append( ((word[i-1], l_tok),  -1) )
                            # this will be updated to where_to_update: the merging candidates. if there are pre-requists for merging candidates, filter here
                            changes.append( ((word[i-1], new_tok), 1) )
                        # if fllw_tok exists, pair(r_tok, fllw_tok) vanished 1 times, pair(new_tok, fllw_tok) emerged 1 times
                        if i+1 < len(word)-1:
                            changes.append( ((r_tok, word[i+2]),  -1) )
                            # this will be updated to where_to_update: the merging candidates. if there are pre-requists for merging candidates, filter here
                            changes.append( ((new_tok, word[i+2]), 1) )
                        word[i] = new_tok
                        word.pop(i+1)
                    i += 1
                
                # apply the changes of this position to pair_counts & where_to_update
                for pair, change in changes:
                    # update the counts of pair by change * cnts of the word
                    pair_counts[pair] += change*freqs[pos]
                    # add the emerged pairs to where_to_udpate
                    if change > 0:
                        where_to_update[pair].add(pos)
            
            # use where_to_update to update heap
            for tok_pair, positions in where_to_update.items():
                # node: (tokens_pair, counts_pair, positions_pair)
                if pair_counts[tok_pair] > 0:
                    to_merge_node = (tok_pair, pair_counts[tok_pair], positions)
                    heapq.heappush(max_heap, (-to_merge_node[1], to_merge_node))

            merge_cnts += 1
            where_to_update = defaultdict(set)

        self.explicit_n_vocab = 256 + len(self._merge_ranks) + len(self._special_marks)
        self._register_special_tokens()





# 从上述的例子中可以看出, 在对单个 word 执行 merge(l_tok, r_tok, new_tok) 的过程中, 原生 word 必须被修改
# 论述: 当单个 word 中存在多个 pair(l_tok, r_tok) 时, 定位这些 pair 的位置(l_tok的index) 分别为 j_1, j_2, .., j_p 等
# 定义 pair_n 的邻域: prev(if any), (l_tok, r_tok)->new_tok, fllw(if any)
# 当 p = 1 时, 显然 changes 只和 pair_1 的邻域相关
# 当 任意 j_k - j_q > 2 时, 即任意 2个 pair(l_tok, r_tok) 之间间隔了其他tok, 显然任意 pair_j 在merge时产生的 changes 只与 pair_j 的邻域有关
# ---> pair_1, ..., pair_p 之间具备 local 对称性: 即执行 pair-merge 的顺序不影响 changes 的结果
# ---> word 不需要根据 pair-merge 实时改变

# 当 存在 j_k - j_q = 2 时, 即存在 至少 2个连续的 pair(l_tok, r_tok) pair_k, pair_q, .., 时, 其中一个 pair 的merge会改变另一个相邻pair的邻域
# ---> pair_1, ..., pair_p 之间不具备 local 对称性: 即执行 pair-merge 的顺序影响 changes 的结果
# ---> pair_1的merge, 改变了 pair_2 的左邻域:
#      (l_tok_1, r_tok_1), (l_tok_2, r_tok_2) --> new_tok_1, (l_tok_2, r_tok_2)
#      这里(r_tok_1, l_tok_2)vanish被正确记录于changes, (new_tok_1, l_tok_2)emerge被错误记录于changes
#      new_tok_1(l_tok_2, r_tok_2) --> new_tok_1, new_tok_2
#      这里(new_tok_1, l_tok_2)vanish被错误记录于changes, (new_tok_1, new_tok_2)emerge被正确记录于changes

#      (new_tok_1, l_tok_2)分别出现了一次错误emerge和错误vanish, 所以其在 pair_counts 中的统计被 reduce 抵消 导致 正确
#      (new_tok_1, l_tok_2)出现的一次错误emerge记录会拿来更新 where_to_update，导致这个 pair 在未来从heap中pop出时, 要多扫描一些噪声word位置
#      也可以在每个 word 产生的 changes 作根据 pair 的reduce sum, 以得到 该word真正正确的 pair-emerge 记录(emerge>vanish)
# ---> word 需要根据 pair-merge 实时改变：前一个pair-merge的后邻域的emerge 抵消于 后一个pair-merge的前邻域的vanish

# 总结 --> 定位多个 pair(l_tok, r_tok)之后, 根据扫描顺序依次执行 pair-merge, 执行merge后 实时 更新word, 使得下一个pair-merge执行在更新后的word上