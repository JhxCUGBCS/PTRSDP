import sys
sys.path.append("D:\\code\\PromptReference\\PTR_code\\PTR-main\\code_script")
import torch
import torch.nn as nn
from arguments import get_model_classes, get_args

class Model(torch.nn.Module):

    def __init__(self, args, tokenizer = None, prompt_label_idx = None):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        self.prompt_label_idx = prompt_label_idx

        self.model = model_config['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.ReLU(),
            # nn.Dropout(p=args.dropout_prob),
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            )

        self.extra_token_embeddings = nn.Embedding(args.new_tokens, self.model.config.hidden_size)
        """
        self.mlp和self.extrac_token_embeddings其实就是用来求new_token的一个向量，但是其实根本没有用到
        """

    def forward(self, input_ids, attention_mask, token_type_ids, input_flags, mlm_labels, labels):
        raw_embeddings = self.model.embeddings.word_embeddings(input_ids)
        """
        raw_embeddings相当于做了一个embedding  
        首先输入的input_ids转换成vocab的one-hot编码  每一个token都变成vocab大小的embedding
        然后再乘上模型自身的embedding矩阵[vocab_nums,hidden_size]
        
        经过上面的embedding最后形成以下形式
        [len(input_ids),hidden_size]
        
        加上batch
        [batch,len(input_ids),hidden_size]
        """
        new_token_embeddings = self.mlp(self.extra_token_embeddings.weight)
        new_embeddings = new_token_embeddings[input_flags]
        """pytorch中使用一个tensor索引另外一个tensor，终于知道为什么叫input_flags
            
            也就是说在new_embeddings中只取行标为0的整行数据，取得次数等于input_flag的次数
            构成一个新的矩阵[len(input_flags),hidden_size]
            
             加上batch
            [batch,len(input_flags),hidden_size]
        """
        inputs_embeds = torch.where(input_flags.unsqueeze(-1) > 0, new_embeddings, raw_embeddings)
        """
        input_flahs.unsqueeze(-1)是在input_flags最后一层增加一个维度
        input_flags.unsqueeze(-1)：[len(input_flags),1]
        new_embeddings：[len(input_flags),hidden_size]
        raw_embeddings：[len(input_ids),hidden_size]   且第一个维度的值一样
        
        判断有没有new_tokens  
        如果有new_token那么输入序列中new_token对应位置的向量就用new_embedding来替代，如果没有就用raw_embeddings替代
        
        也就是说先把input_flahs拉成了每行只有一个元素但是有len(input_flags)行的tensorlong矩阵
        
        然后在第i行，如果input_flags的值>0，就表示有new_token那么就返回从mlp学到的new_token的词向量
            在第i行，如果input_flags的值<=0，就表示没有new_token那么就返回从Bert学到的input_ids的词向量
        """
        hidden_states, _ = self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        """
        上面是放入到BERT中，获得隐藏层状态
        hidden_states  [len(input_ids),hidden_size]
        
        加上batch
            [batch,len(input_ids),hidden_size]
        """
        hidden_states = hidden_states[mlm_labels >= 0].view(input_ids.size(0), len(self.prompt_label_idx), -1)
        """
        首先hidden_states经过一个tensor[tensor]的索引，只取5个mask的向量，因此hidden_states的形式变化成如下形式
        hidden_states [5,hidden_size]  并且里面的向量行号和prompt中的位置一一对应
        
        hidden_states在经过一个view()函数进行形状变化
        view(a,b,-1)表示参数a、参数b已知，参数c根据原来tensor矩阵hidden_states的维度自动进行计算
        且c=size(hidden_states)/a/b  在这里c=hidden_size
        也就是说hidden_states变成如下形式
        hidden_states：[batch,5,hidden_size]  并且第二维度中的5对应mask的数量 且和mask的位置一一对应        
        """
        logits = [
            torch.mm(
                hidden_states[:,index,:], 
                self.model.embeddings.word_embeddings.weight[i].transpose(1,0)
            )
            for index, i in enumerate(self.prompt_label_idx)
        ]
        return logits
    """取hidden_states中所有batch的存的第i个位置的向量和第i个位置set(i)中所有answer word生成的embedding向量做矩阵相乘
    矩阵相乘后返回第i个位置的向量分别对应set(i)中每一个answer word的logits，这个logits可以直接使用sofmax转换成概率
    
    torch.mm(a,b)矩阵a和矩阵b进行矩阵相乘比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
    hidden_states[:,index,:]：[1,hidden_size]
    self.model.embeddings.word_embeddings.weight[i]:[num_set(i),hidden_size]  num_set(i)表示第i个mask位置有多少个answer word,也就是有多少个元素
    self.model.embeddings.word_embeddings.weight[i].transpose(1,0)：第1维度和第0维度交换  [hidden_size,num_set(i)]
    
    hidden_states[:,index,:]和self.model.embeddings.word_embeddings.weight[i].transpose(1,0)做矩阵乘法，生成logits的一个元素，该元素形式如下：    
    
    [batch,hidden_size]X[hidden_size,num_set(i)]
    最终生成的logits如下
    [5,batch,num_set(i)]
    
    [i,batch,num_set(i)]:表示这批batch中所有数据的第i个位置的生成的logits的矩阵
    维度应该是[batch,num_set(i)]
    
    他这样做才能保证正常运算，因为不同位置的num_set(i)数量不一样，这样操作可以保证每次取的logits是同纬度的。按照位置把维度统一
    """

def get_model(tokenizer, prompt_label_idx):
    args = get_args()
    model = Model(args, tokenizer, prompt_label_idx)
    #if torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model)
    model.cuda()
    return model

def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_tokens(special)
    return tokenizer

