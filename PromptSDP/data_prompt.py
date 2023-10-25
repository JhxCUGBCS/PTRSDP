import sys
sys.path.append("D:\\code\\PromptReference\\PTR_code\\PTR-main\\code_script")
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from arguments import get_args

class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()
        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def cpu(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cpu()


class REPromptDataset(DictDataset):

    def __init__(self, path = None, name = None, rel2id = None, tokenizer = None, temps = None, features = None):

        with open(rel2id, "r") as f:
            self.rel2id = json.loads(f.read())
        if not 'NA' in self.rel2id:
            self.NA_NUM = self.rel2id['no_relation']
        else:  
            self.NA_NUM = self.rel2id['NA']
        """
        获取每个关系类别对应的数字标签是多少，比如关系类别per:charges数字标签是1
        """

        self.num_class = len(self.rel2id)
        self.temps = temps
        self.get_labels(tokenizer)

        if features is None:
            self.args = get_args()
            with open(path+"/" + name, "r") as f:
                features = []
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line) > 0:
                        features.append(eval(line))
                        #通过eval()函数把字符串的line转换成了字典
            features = self.list2tensor(features, tokenizer)

        super().__init__(**features)
    
    def get_labels(self, tokenizer):
        total = {}
        # total{0: 第0个位置对应的所有的标签值,列表形式，后同  ,1: 第1个位置对应的所有的标签值    ,2: 第2个位置对应的所有的标签值
        #       ,3: 第3个位置对应的所有的标签值    ,4: 第4个位置对应的所有的标签值    }
        self.temp_ids = {}
        #一共有42个键值对，键名是关系类别

        for name in self.temps:
            #这个循环对42个关系类别进行循环，
            last = 0
            self.temp_ids[name] = {}
            # temp_ids[name]一共有两个键值对，但是每一个关系都有一个他自己的label_ids列表和maks_ids列表。并且label_ids是一维列表，mask_ids是二维列表
            # 一个是label_ids,用来依次存储每个关系中每个位置对应的标签在vocab中的index，每个关系有5个，有42个关系.
            # 一个是mask_ids,用来一次存储每个关系中每个带mask的模板经过tokenizer转化得到的index,注意这个index是不带CLS和SEP的，因为原始语句中add_special_tokens=False
            # 每个关系有5个，有42个关系
            self.temp_ids[name]['label_ids'] = []
            #temp_ids[name]['label_ids']存储真实标签在vocab中的index，比如对于关系per:charged 存储真实标签person was charged with enevnt这几个单词在vocab中的index
            #对于重复的元素也会进行存储，也就是说最后会按照读取的关系类别的顺序再按照位置的顺序存储每一个真实标签的值。
            #每个关系对应一个
            #最后也就是存储了210个元素  42X5  42：关系数 5：每个关系中有五个位置，每个位置都有一个对应的真实标签
            self.temp_ids[name]['mask_ids'] = []
            #temp_ids[name]['mask_ids']存储带mask的模板在vocab中的index，比如对于关系per:charged 带mask的模板是the mask   mask mask mask   the mask这几个模板在vocab中对应的index
            #对于重复的元素也会进行存储，也就是说最后会按照读取的关系类别的顺序存储
            #最后也就是存储了210个元素  42X5  42：关系数 5：每个关系中有五个位置，每个位置都有一个对应的带mask的模板

            for index, temp in enumerate(self.temps[name]['temp']):
                #这个循环对单个关系中的temp进行循环，每一个关系共有三个temp,分别是the mask    mask mask mask   the mask
                _temp = temp.copy()
                _labels = self.temps[name]['labels'][index]
                _labels_index = []

                for i in range(len(_temp)):
                    #这个循环对每一个temp中的元素进行循环，把mask位置的元素替换成真实的标签值，并且记录这个标签在temp中第几个位置
                    if _temp[i] == tokenizer.mask_token:
                        _temp[i] = _labels[len(_labels_index)]
                        _labels_index.append(i)
                #1  temp:the MASK          下标temp:the person         labels_index:[1]
                #2  temp:MASK MASK MASK    下标temp:was charged with   label_index:[0,1,2]
                #3  temp:temp MASK         下标temp:the enevt          label_index:[1]
                original = tokenizer.encode(" ".join(temp), add_special_tokens=False)
                #对原始的带MASK的temp进行编码，转换成vocab中的index，并且带有CLS和SEP的下标，存储在列表中
                final =  tokenizer.encode(" ".join(_temp), add_special_tokens=False)
                #对copy的带真实标签的MASK进行编码，转换成vocab中的index，并且带有CLS和SEP的下标，存储在列表中

                assert len(original) == len(final)
                self.temp_ids[name]['label_ids'] += [final[pos] for pos in _labels_index]
                #把真实标签对应的index值存到temp_ids[name]['label_ids']中，比如把person在vocab中的index存在这个里面。temp_ids[name]['label_ids']是一个一维数组
                #1  temp_ids[name]['label_ids']=tokenizer.encode(person )
                #2  temp_ids[name]['label_ids']=tokenizer.encode(person )+tokenizer.encode(person was charged with)
                #3  temp_ids[name]['label_ids']=tokenizer.encode(person )+tokenizer.encode(person was charged with)+tokenizer.encode(event)

                #1  labels_index:[0]      last:0
                #2  label_index:[0,1,2]   last:1
                for pos in _labels_index:
                    if not last in total:
                        total[last] = {}
                    # last:0    pos:1 对应的标签位置   final[pos]:标签person经过tokenizer.encode()获得的分词结果
                    #total{0: 第0个位置对应的所有的标签值  ,1: 第1个位置对应的所有的标签值    ,2: 第2个位置对应的所有的标签值    ,3: 第3个位置对应的所有的标签值    ,4: 第4个位置对应的所有的标签值    }
                    total[last][final[pos]] = 1
                    last+=1
                self.temp_ids[name]['mask_ids'].append(original)
                #把关系对应的带MASK的temp模板进行存储，并且同于同一种temp可以重复存储，也就是说即便列表里有了the MASK的index模板，还可以继续添加the MASK的index模板
                #1  original:tokenizer.encode(the MASK)
                #2  original:tokenizer.encode(the MASK)+tokenizer.encode(MASK MASK MASK)
                #3  original:tokenizer.encode(the MASK)+tokenizer.encode(MASK MASK MASK)+tokenizer.encode(the MASK)
        """生成每个位置对应的答案字的集合"""
        self.set = [(list)((sorted)(set(total[i]))) for i in range(len(total))]
        #把字典total转换成列表，由于一共有五个位置，所以转换后有五个子列表
        #并且对于每一个子列表内的元素去除重复的元素，并且按照index的大小进行从小到大的排序
        #上面那句语句依次执行一下规则，只看for前面的部分
        #1  set(total[i]) 转换成集合，去除重复元素，获取第i个位置所有的label值的index。存储形式为集合，{}
        #2  (sorted)(set(total[i]))  对第i个位置上的标签的index进行排序，最后返回一个完成排序的列表，按照从大到小的顺序
        #3  ((sorted)(set(total[i])))
        #4  (list)((sorted)(set(total[i])))

        print ("=================================")
        for i in self.set:
            print (i)
        print ("=================================")

        for name in self.temp_ids:
            """获取label在那个集合中是第几个。由第一个位置的label构成了一个集合，一共五个位置所以有五个集合。  这个是看这个元素在他同位置的集合中是第几个
            比如看per:charge关系中第一个位置的person在第一个位置的标签构成的集合set[0]中按照大小是排在第几个。
            
            这样的目的：相当于每一个位置是一个sub-prompt子任务，然后看看这个子任务的标签是什么，
            但是这个标签并不是这个字在vocab中的index，而是这个字在这个子任务的answer空间中对应的是第几个，把这个第几个作为真实标签
            """
            for j in range(len(self.temp_ids[name]['label_ids'])):
                self.temp_ids[name]['label_ids'][j] = self.set[j].index(
                    self.temp_ids[name]['label_ids'][j])

        self.prompt_id_2_label = torch.zeros(len(self.temp_ids), len(self.set)).long()
        """
        #维度 [关系数X5]  因为由五个位置    42行5列
        [[0,0,0,0,0],
         [0,0,0,0,0],
         ...]
        """
        
        for name in self.temp_ids:
            """
            元素的类型是tensorlong
            整个矩阵的维度是[num_rel,num_pos]
            第ij个元素的含义:第i个关系的第j个位置的标签
            
            但是要注意的是这个标签并不是真实标签字在vocab中的index，而是它在同位置中排序后对应的第几个
            比如prompt_id_2_label[28][0]对应的真是字是person,表示关系per:charges关系中第一个位置的标签person在vocab中的index在set[0]中对应的index。
            如果person在vocab中的index在set[0]中排在第三个，在set中对应的indx是3，那么prompt_id_2_label[28][0]=3,并且类型是tensorlong
            """
            for j in range(len(self.prompt_id_2_label[self.rel2id[name]])):
                self.prompt_id_2_label[self.rel2id[name]][j] = self.temp_ids[name]['label_ids'][j]


        self.prompt_id_2_label = self.prompt_id_2_label.long().cuda()
        
        self.prompt_label_idx = [
            torch.Tensor(i).long() for i in self.set
        ]
        """
        上面的语句是把set中存储的label在vocab中对应的index转换成tensorlong
        """

    def save(self, path = None, name = None):
        path = path + "/" + name  + "/"
        np.save(path+"input_ids", self.tensors['input_ids'].numpy())
        np.save(path+"token_type_ids", self.tensors['token_type_ids'].numpy())
        np.save(path+"attention_mask", self.tensors['attention_mask'].numpy())
        np.save(path+"labels", self.tensors['labels'].numpy())
        np.save(path+"mlm_labels", self.tensors['mlm_labels'].numpy())
        np.save(path+"input_flags", self.tensors['input_flags'].numpy())
        # np.save(path+"prompt_label_idx_0", self.prompt_label_idx[0].numpy())
        # np.save(path+"prompt_label_idx_1", self.prompt_label_idx[1].numpy())
        # np.save(path+"prompt_label_idx_2", self.prompt_label_idx[2].numpy())

    @classmethod
    def load(cls, path = None, name = None, rel2id = None, temps = None, tokenizer = None):
        path = path + "/" + name  + "/"
        features = {}
        features['input_ids'] = torch.Tensor(np.load(path+"input_ids.npy")).long()
        features['token_type_ids'] = torch.Tensor(np.load(path+"token_type_ids.npy")).long()
        features['attention_mask'] = torch.Tensor(np.load(path+"attention_mask.npy")).long()
        features['labels'] = torch.Tensor(np.load(path+"labels.npy")).long()
        features['input_flags'] = torch.Tensor(np.load(path+"input_flags.npy")).long()
        features['mlm_labels'] = torch.Tensor(np.load(path+"mlm_labels.npy")).long()
        res = cls(rel2id = rel2id, features = features, temps = temps, tokenizer = tokenizer)
        #通过cls又返回一个类
        # res.prompt_label_idx = [torch.Tensor(np.load(path+"prompt_label_idx_0.npy")).long(),
        #     torch.Tensor(np.load(path+"prompt_label_idx_1.npy")).long(),
        #     torch.Tensor(np.load(path+"prompt_label_idx_2.npy")).long()
        # ]
        return res

    def list2tensor(self, data, tokenizer):
        res = {}
        """
        res中共有六个键值对，键名如下，值是列表的形式，值的长度=数据的数量，有多少数据那么值这个列表就有多长，并且他是一个二维列表
        input_ids：原始的句子+prompt构成的序列经过token的index，带有CLS和SEP。 以tensorlong形式，二维列表
        token_type_ids：全为0的列表，让模型判断序列是句子对中的第几个句子。tensorlong  二维列表
        attention_mask：全为1的列表，让模型判断序列中对应位置的元素是否需要参加计算。 tensor long   二维列表
        input_flags：全为0的列表。  tensor long 二维列表
        mlm_labels：检查输入的token中哪几个是mask_token。如果是mask_token那么对应的位置为1，其余的为0。二维列表  tensorlong
        labels：每个数据所属的关系名称在rel2id.json文件中对应的数字标签，tensorlong  一维列表
        """
        res['input_ids'] = []
        res['token_type_ids'] = []
        res['attention_mask'] = []
        res['input_flags'] = []
        res['mlm_labels'] = []
        res['labels'] = []

        for index, i in enumerate(tqdm(data)):
            """三个列表 input_ids  token_type_ids  input_flags
                   input_ids CLS+原始句子+the MASK e1_name MASK MASK MASK the MASK e2_name+SEP    元素是这些token在vocab中对应的index
                   token_type_ids   全为0的元素，长度和上面的列表一样  意义是让模型判断是这个token是句子对中的第几个句子
                   input_flags      全为0的元素，长度和上面的列表一样  意义是让模型判断这个token需不需要关注
                   """
            input_ids, token_type_ids, input_flags = self.tokenize(i, tokenizer)
            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            if padding_length > 0:
                """
                对于padding的句子，在序列的尾部添加padding的标志让模型知道
                """
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
                input_flags = input_flags + ([0] * padding_length)
            
            assert len(input_ids) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(token_type_ids) == self.args.max_seq_length
            assert len(input_flags) == self.args.max_seq_length

            label = self.rel2id[i['relation']]
            res['input_ids'].append(np.array(input_ids))
            res['token_type_ids'].append(np.array(token_type_ids))
            res['attention_mask'].append(np.array(attention_mask))
            res['input_flags'].append(np.array(input_flags))
            res['labels']. append(np.array(label))
            mask_pos = np.where(res['input_ids'][-1] == tokenizer.mask_token_id)[0]
            """上面这一行代码精妙的地方
            1、np.where()[0]来确定是行索引
            2、通过[-1]来确保每一次都锁定到当前需要进行判断的input_ids数据中
            """
            mlm_labels = np.ones(self.args.max_seq_length) * (-1)
            mlm_labels[mask_pos] = 1
            res['mlm_labels'].append(mlm_labels)
            #检查输入的token中哪几个位置上的token是mask_token
        for key in res:
            """把每一个键值对中的值从int数字先转换成ndarray然后再转换成TensorLong"""
            res[key] = np.array(res[key])
            res[key] = torch.Tensor(res[key]).long()
        return res

    def tokenize(self, item, tokenizer):
        """
        {"token": ["Zagat", "Survey", ",", "the", "guide", "empire", "that", "started", "as", "a", "hobby", "for", "Tim", "and", "Nina", "Zagat", "in", "1979", "as", "a", "two-page", "typed", "list", "of", "New", "York", "restaurants", "compiled", "from", "reviews", "from", "friends", ",", "has", "been", "put", "up", "for", "sale", ",", "according", "to", "people", "briefed", "on", "the", "decision", "."],
         "h": {"name": "Zagat", "pos": [0, 1]},
         "t": {"name": "1979", "pos": [17, 18]},
         "relation": "org:founded"}
        """
        sentence = item['token']
        pos_head = item['h']
        pos_tail = item['t']
        rel_name = item['relation']

        temp = self.temps[rel_name]

        sentence = " ".join(sentence)
        sentence = tokenizer.encode(sentence, add_special_tokens=False)
        e1 = tokenizer.encode(" ".join(['was', pos_head['name']]), add_special_tokens=False)[1:]
        e2 = tokenizer.encode(" ".join(['was', pos_tail['name']]), add_special_tokens=False)[1:]

        # prompt =  [tokenizer.unk_token_id, tokenizer.unk_token_id] + \
        prompt = self.temp_ids[rel_name]['mask_ids'][0] + e1 + \
                 self.temp_ids[rel_name]['mask_ids'][1] + \
                 self.temp_ids[rel_name]['mask_ids'][2] + e2 
        #  + \
        #  [tokenizer.unk_token_id, tokenizer.unk_token_id]
        """
        prompt形式
        the MASK e1_name MASK MASK MASK the MASK e2_name   这句话经过tokenizer.encode()产生的结果
        """

        flags = []
        last = 0
        for i in prompt:
            # if i == tokenizer.unk_token_id:
            #     last+=1
            #     flags.append(last)
            # else:
            flags.append(0)
            """构建flags
            flags的长度=prompt列表的长度
            flags的元素全为0
            """
        
        tokens = sentence + prompt
        flags = [0 for i in range(len(sentence))] + flags
        #flags的长度和token一样长，但是元素全为0
        # tokens = prompt + sentence
        # flags =  flags + [0 for i in range(len(sentence))]        
        
        tokens = self.truncate(tokens, 
                               max_length = self.args.max_seq_length - tokenizer.num_special_tokens_to_add(False))
        flags = self.truncate(flags, 
                               max_length = self.args.max_seq_length - tokenizer.num_special_tokens_to_add(False))

        """最终形成的token和flag形式
        token: 原始sentence+prompt
        flag:  [0]*token的长度  
        """

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens)
        #tokens是已经转换成vocab中对应的index，并以列表的形式进行存储
        #这里的build_inputs_with_special_tokens是把这个tokens转换成模型可以接受的形式，给他添加CLS和SEP，并且存储在一个列表中
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens)
        #这个语句是为了判断句子句中的句子属于第一个句子还是第二个句子，但是由于这里不存在句子对，因此返回全为0的元素，存储在一个列表中
        #这个语句相当于为模型生成一个可以判断是句子对中的第一个句子还是第二个句子的输入。但是这里不存在句子对因此元素全为0
        input_flags = tokenizer.build_inputs_with_special_tokens(flags)
        #这个语句是为flags创造token模型可以接受的形式，作用同上，最后存储在列表中。但是实际效果相当于tokenizer()中生成的mask_type，
        # 让模型判断需不需要关注这个位置上的token，0表示关注
        input_flags[0] = 0
        input_flags[-1] = 0
        assert len(input_ids) == len(input_flags)
        assert len(input_ids) == len(token_type_ids)
        """最后返回三个列表 input_ids  token_type_ids  input_flags
        input_ids CLS+原始句子+the MASK e1_name MASK MASK MASK the MASK e2_name+SEP    元素是这些token在vocab中对应的index
        token_type_ids   全为0的元素，长度和上面的列表一样  意义是让模型判断是这个token是句子对中的第几个句子
        input_flags      全为0的元素，长度和上面的列表一样  意义是让模型判断这个token需不需要关注
        """
        return input_ids, token_type_ids, input_flags

    def truncate(self, seq, max_length):
        if len(seq) <= max_length:
            return seq
        else:
            print ("=========")
            return seq[len(seq) - max_length:]



"""
PTR是把关系抽取转换成几个子任务，每一个字任务都可以看作是一个文本分类任务

首先通过get_label来获得每一个位置上的answer word的集合，并把它转换成vocab中的index。同时获得每一个子任务的prompt

然后给answer word的集合根据他们在voca中的index从大到小进行排序，排完序的标签就是这个文本分类子任务的标签，
比如，对于第一个位置来说，排完序后排在第几个位置，那么对于第一个位置对应的文本分类子任务来说，它对应的标签就是第几个位置的数值。
最后把每一个子任务对应的label数字存储在prompt_label_2_ids中。而那些标签字对应的index存储在prompt_label_ids

每一个子任务的prompt存在temp_ids中，当然存的是template word在vocab中对应的index;但是在最后的训练中使用的是完整的prompt。并把它转化成了input_ids存储在npy文件中

最终的数据全部存储在一个字典features中（当然这个在父类中是**tensor），features一共有六个键值对，每个键值对中的值都是列表，除了label对应的列表是一维列表外，其余都是二维列表。
每个列表的长度都是加载的数据总数。例如train对应的features中列表的长度就是train.txt文件中数据的总数
"""