# tabular.py

from collections import defaultdict
import numpy as np

class CategDataParser:
    '''
    输入文本文件, 每行:
    label1, label2, ..., labelM, feat1, feat2, ..., featN
    min_thresholds: 统计的最小频次. 严格低于这个值的将被视为unknown值
    '''
    def __init__(self,
                 data_path,
                 num_cols: list,
                 components = ['y', 'x'],
                 label_first = True,
                 set_label_mappers=None,
                 min_threshold=4, seperator='\t'
                 ):
        
        assert len(num_cols) == len(components), 'num_cols length must match components length'
        if label_first:
            min_thresholds = [1, min_threshold]
        else:
            min_thresholds = [min_threshold, 1]
        if not isinstance(data_path, list):
            data_path = [data_path, ]
        num_label, num_feat, row_count, self._ori_data = num_cols[0], num_cols[1], 0, {}
        self.NUM_LABELS, self.NUM_FEATURES, self.COMPONENTS, self.LABEL_FIRST = num_label, num_feat, components, label_first
        feat_counter = defaultdict(lambda: defaultdict(int)) # key=j, value={'v1':cnt1, 'v2':cnt2, ...}, j=1, 2, ...N
        label_counter = defaultdict(lambda: defaultdict(int)) # key=i, value={'v1':cnt1, 'v2':cnt2, ...}, i=1, 2, ...M
        feat_dims = np.zeros(num_feat, dtype=np.int64) # [feat_size_feat1, feat_size_feat2, ..., feat_size_featN]
        label_dims = np.zeros(num_label, dtype=np.int64) # [label_size_label1, label_size_label2, ..., label_size_labelM]
        for fpath in data_path:
            with open(fpath) as f:
                for line in f:
                    values = line.rstrip('\n').split(seperator) # index 0到M-1是label, index M到M+N-1是feat
                    if len(values) != num_feat + num_label: # 如果该行数据有丢失, 丢弃
                        continue
                    instance = {} # {'y': [y1, y2,.., yM], 'x': [x1, x2, ...xN]}
                    for i in range(1, num_label+1): # i从1到M
                        label_counter[i][values[i-1]] += 1 # label_counter[i]中, values[i-1]对应的Value cnts加1
                        instance.setdefault(components[0], []).append(values[i-1]) # instance['y'] list记录values[i-1]
                    for j in range(1, num_feat+1): # j从1到N
                        feat_counter[j][values[num_label+j-1]] += 1 # feat_counter[j]中, values[M+j-1]对应的Value cnts+1
                        instance.setdefault(components[1], []).append(values[num_label+j-1]) # instance['x'] list记录values[num_label+j-1]
                    self._ori_data[row_count] = instance # ori_data[row_index] = instance
                    row_count += 1
        # key: i, i从1到M; value: set(distinct values of label i)
        label_sets = {i: {value for value, cnt in cnts.items() if cnt >= min_thresholds[0]} for i, cnts in label_counter.items()}
        # key: j, i从1到N; value: set(distinct values of feat j)
        feat_sets = {j: {value for value, cnt in cnts.items() if cnt >= min_thresholds[1]} for j, cnts in feat_counter.items()}
        # key: i, i从1到M; value: dict{y1:y1_idx, y2:y2_idx, ...}
        self._label_mapper = {i:{label_val:idx for idx, label_val in enumerate(valset)} for i, valset in label_sets.items()}
        # key: j, j从1到N; value: dict{x1:x1_idx, x2:x2_idx, ...}
        self._feat_mapper = {j:{feat_val:idx for idx, feat_val in enumerate(valset)} for j, valset in feat_sets.items()}
        if label_first: # 如果用components[0]做出来的_label_mapper是标签的mapper, 要验证其将'0'map成0, '1'map成1, etc
            self._label_mapper = self._check_adjust_for_label_mapper(self._label_mapper, set_label_mappers)
        else: # 如果用components[1]做出来的_feat_mapper是标签的mapper, 要验证其将'0'map成0, '1'map成1, etc
            self._feat_mapper = self._check_adjust_for_label_mapper(self._feat_mapper, set_label_mappers)
        #key: i, i从1到M; value: len(set(distinct values of label_i)), map any unknown value of label_i to a large-enough index
        self._label_defaults = {i:len(valset) for i, valset in label_sets.items()}
        #key: j, j从1到N; value: len(set(distinct values of feat_j)), map any unknown value of feat_j to a large-enough index
        self._feat_defaults = {j:len(valset) for j, valset in feat_sets.items()}
        for i, value_map in self._label_mapper.items(): # i从1到M; value_map for i是 dict{y1:y1_idx, y2:y2_idx, ...}
            label_dims[i-1] = len(value_map) + 1 # plus unknown value
        for j, value_map in self._feat_mapper.items(): # j从1到N; value_map for j是 dict{x1:x1_idx, x2:x2_idx, ...}
            feat_dims[j-1] = len(value_map) + 1 # plus unknown value
        self._label_dims = label_dims
        self._feat_dims = feat_dims

    @staticmethod
    def _check_adjust_for_label_mapper(value_mappers, set_label_mappers=None):
        _new_mappers = {}
        for i, valmap in value_mappers.items(): # 逐一检查valmap, dict{y1:y1_idx, y2:y2_idx, ...}
            num_vals = len(valmap)
            need_adjust = False
            for v, v_idx in valmap.items(): # valmap中所有v:v_idx pair
                if int(v) != v_idx: # 不匹配
                    need_adjust = True
                    break
            if need_adjust:
                _new_mappers[i] = {str(idx):idx for idx in range(num_vals)}
        if _new_mappers and not set_label_mappers: # 如果需要更改, 且没有手动输入set_label_mappers, 那么自动更改原来的value_mappers
            for i, new_val_map in _new_mappers.items():
                value_mappers[i] = new_val_map
        elif _new_mappers and set_label_mappers: # 如果需要更改, 且手动输入了set_label_mappers, 那么仅需对set_label_mappers作数量检查
            assert isinstance(set_label_mappers, dict)
            assert len(value_mappers) == len(set_label_mappers), "set_label_mappers not cover all label columns"
            assert all([len(valmap1) == len(valmap2) for valmap1, valmap2 in zip(value_mappers.values(), set_label_mappers.values())]),\
                f"set_label_mappers not cover all labels' values"
            value_mappers = set_label_mappers
        else: # 不需要更改
            pass
        return value_mappers

    @property
    def label_mapper(self):
        if self.LABEL_FIRST:
            return self._label_mapper
        else:
            return self._feat_mapper
    
    @property
    def feat_mapper(self):
        if self.LABEL_FIRST:
            return self._feat_mapper
        else:
            return self._label_mapper
    
    @property
    def label_defaults(self):
        if self.LABEL_FIRST:
            return self._label_defaults
        else:
            return self._feat_defaults
    
    @property
    def feat_defaults(self):
        if self.LABEL_FIRST:
            return self._feat_defaults
        else:
            return self._label_defaults
    
    @property
    def label_dims(self):
        if self.LABEL_FIRST:
            return self._label_dims
        else:
            return self._feat_dims
    
    @property
    def feat_dims(self):
        if self.LABEL_FIRST:
            return self._feat_dims
        else:
            return self._label_dims
    
    @property
    def raw_data(self):
        return self._ori_data
    
    @property
    def offset(self):
        if self.LABEL_FIRST:
            self._offset = np.array((*[0,]*self.NUM_LABELS, 0, *np.cumsum(self._feat_dims)[:-1]))
        else:
            self._offset = np.array((0, *np.cumsum(self._label_dims)[:-1], *[0,]*self.NUM_FEATURES))
        return self._offset.astype(np.int64)

    @property
    def mapped_data(self):
        # ori_data[row_index] = # {'y': [y1, y2,.., yM], 'x': [x1, x2, ...xN]}
        num_rows = len(self._ori_data)
        num_cols = self.NUM_LABELS + self.NUM_FEATURES
        mapped_data = np.zeros((num_rows, num_cols))
        for r_idx in range(num_rows):
            instance = self._ori_data[r_idx] # {'y': [y1, y2,.., yM], 'x': [x1, x2, ...xN]}
            row = []
            for i, v in enumerate(instance[self.COMPONENTS[0]]): # [y1, y2,.., yM]
                row.append( self._label_mapper[i+1].get(v, self._label_defaults[i+1]) )
            for j, v in enumerate(instance[self.COMPONENTS[1]]): # [x1, x2,.., xN]
                row.append( self._feat_mapper[j+1].get(v, self._feat_defaults[j+1]) )
            mapped_data[r_idx] = np.array(row)
        return mapped_data.astype(np.int64)
    
    @property
    def mapped_offset_data(self):
        return self.mapped_data + self.offset