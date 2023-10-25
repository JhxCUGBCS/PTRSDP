import sys
sys.path.append("D:\\code\\PromptReference\\PTR_code\\PTR-main\\code_script")
from arguments import get_args_parser
from templating import get_temps
from modeling import get_model, get_tokenizer
from data_prompt import REPromptDataset
from optimizing import get_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random

def f1_score(output, label, rel_num, na_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    
    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0 :
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)

    return micro_f1, f1_by_relation

def evaluate(model, dataset, dataloader):
    model.eval()
    scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model(**batch)
            res = []
            for i in dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]                
                _res = _res.detach().cuda()
                res.append(_res)
            logits = torch.stack(res, 0).transpose(1,0)
            labels = batch['labels'].detach().cuda().tolist()
            all_labels+=labels
            scores.append(logits.cuda().detach())
        scores = torch.cat(scores, 0)
        scores = scores.detach().cuda().numpy()
        all_labels = np.array(all_labels)
        np.save("scores.npy", scores)
        np.save("all_labels.npy", all_labels)

        pred = np.argmax(scores, axis = -1)
        mi_f1, ma_f1 = f1_score(pred, all_labels, dataset.num_class, dataset.NA_NUM)
        return mi_f1, ma_f1

args = get_args_parser()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #if args.n_gpu > 0:
        #torch.cuda.manual_seed_all(seed)

set_seed(args.seed)
tokenizer = get_tokenizer(special=[])
temps = get_temps(tokenizer)

# If the dataset has been saved, 
# the code ''dataset = REPromptDataset(...)'' is not necessary.
dataset = REPromptDataset(
    path  = args.data_dir, 
    name = 'train.txt', 
    rel2id = args.data_dir + "/" + "rel2id.json", 
    temps = temps,
    tokenizer = tokenizer,)
dataset.save(path = args.output_dir, name = "train")

# If the dataset has been saved, 
# the code ''dataset = REPromptDataset(...)'' is not necessary.
dataset = REPromptDataset(
    path  = args.data_dir, 
    name = 'val.txt', 
    rel2id = args.data_dir + "/" + "rel2id.json", 
    temps = temps,
    tokenizer = tokenizer)
dataset.save(path = args.output_dir, name = "val")

# If the dataset has been saved, 
# the code ''dataset = REPromptDataset(...)'' is not necessary.
dataset = REPromptDataset(
    path  = args.data_dir, 
    name = 'test.txt', 
    rel2id = args.data_dir + "/" + "rel2id.json", 
    temps = temps,
    tokenizer = tokenizer)
dataset.save(path = args.output_dir, name = "test")

train_dataset = REPromptDataset.load(
    path = args.output_dir, 
    name = "train", 
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")

val_dataset = REPromptDataset.load(
    path = args.output_dir, 
    name = "val", 
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")

test_dataset = REPromptDataset.load(
    path = args.output_dir, 
    name = "test", 
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")

train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

train_dataset.cuda()
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

val_dataset.cuda()
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size//2)

test_dataset.cuda()
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size//2)

model = get_model(tokenizer, train_dataset.prompt_label_idx)
optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
criterion = nn.CrossEntropyLoss()

mx_res = 0.0
hist_mi_f1 = []
hist_ma_f1 = []
mx_epoch = None
last_epoch = None
for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
    model.train()
    model.zero_grad()
    tr_loss = 0.0
    global_step = 0 
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        logits = model(**batch)
        """
        [5,batch,num_set(i)]
        [i,batch,num_set(i)]:表示这批batch中所有数据的第i个位置的生成的logits的矩阵
        维度应该是[batch,num_set(i)]
    
        他这样做才能保证正常运算，因为不同位置的num_set(i)数量不一样
        这样操作可以保证每次取的logits是同纬度的。按照位置把维度统一
        """
        labels = train_dataset.prompt_id_2_label[batch['labels']]
        """
        batch['labels']:关系标签对应的数字类别
        prompt_id_2_label[batch['labels']]：这个关系每个位置对应的标签，注意不是tokenizer转化的数字
                                            而是每个位置内部又对应的标签
        labels:[label_0,label_1,label_2,label_3,label_4]
        balel_i:第i个对应的真实标签
        
        由于有batch，所以最后形式如下
        [batch,5]
        """
        
        loss = 0.0
        for index, i in enumerate(logits):
            """
            logits:[5,batch,num_set(i)]
            
            由于logits的第一个维度是5。所以index也就是从0到4，索引的是mask的位置
            i:[batch,num_set(i)] 
            """
            loss += criterion(i, labels[:,index])
            """
            先分别计算这批batch中每一个位置的loss，也就是相当于先计算每一个sub-prompt任务的loss
            然后再把这些loss加起来
            """
        loss /= len(logits)

        res = []
        for i in train_dataset.prompt_id_2_label:
            """
            i:每一个在这个数据集中的关系标签5个mask位置对应的真实的标签书
            注意：在这个数据集中不一定这个数据集包含所有的关系
            """
            _res = 0.0
            for j in range(len(i)):
                #j 从0到4  共5个数
                _res += logits[j][:, i[j]]
                """
                logits[j]:表示索引第j个mask位置
                i[j]:锁定每一个mask位置对应的真实标签的位置
                第j个元素表示batch的数据第j个位置对于真实标签产生的logit相加
                
                对于每一个关系，获取这批batch中每每一个数据对于这个mask位置针对真实标签产生的logits是多少，把这些加起来
                也就是说_res的维度是[batch,1]  单独从_res拿出一个数据则对应的是每个数据中针对五个真实标签位置产生的标量
                
                然后把五个位置的都加起来，最后的形状还是[batch,1]   也就是长度为batch的一维矩阵
                一个_res表示对一整个完整的关系标签的logits是多少
                """
            res.append(_res)
            """
            res列表中有i个元素，i表示一共在这个数据集中有i种关系
            """
        final_logits = torch.stack(res, 0).transpose(1,0)
        """
        res中一共有num_rels个_res,每一个_res的形状都是[batch,1]
        
        torch.stack(res,0)  在0轴上增加一个维度，把res列表里的元素进行拼接，形成的tensor矩阵形状如下：
        [num_rels,batch]  其中第x列第y行表示batch中第y个数据对于第x个关系生成的logit标量值是多少
                               第y行表示batch中第y个数据对于这个数据集中每一个完整的关系标签生成的logits是多少
                               
        然后维度交换，变为
        [batch,num_rels]  每i行表示batch中第i个数据对数据集中所有关系类别生成的logits是多少
        """

        loss += criterion(final_logits, batch['labels'])
        """
        final_logits:[batch,num_rels]  每一个batch中的数据对于数据集中所有的关系生成的logits分别是多少
        batch['labels']:[batch,1]     每一个batch中的数据有一个真实的完整关系标签
        
        把针对每个完整的关系的loss和前面针对每个sub-prompt的loss进行相加
        """

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer_new_token.step()
            scheduler_new_token.step()
            model.zero_grad()
            print (args)
            global_step += 1
            print (tr_loss/global_step, mx_res)

    mi_f1, ma_f1 = evaluate(model, val_dataset, val_dataloader)
    hist_mi_f1.append(mi_f1)
    hist_ma_f1.append(ma_f1)
    if mi_f1 > mx_res:
        mx_res = mi_f1
        mx_epoch = epoch
        torch.save(model.state_dict(), args.output_dir+"/"+'parameter'+str(epoch)+".pkl")
    last_epoch = epoch

torch.save(model.state_dict(), args.output_dir+"/"+'parameter'+str(last_epoch)+".pkl")

# print (hist_mi_f1)
# print (hist_ma_f1)

# model.load_state_dict(torch.load(args.output_dir+"/"+'parameter'+str(mx_epoch)+".pkl"))
# mi_f1, _ = evaluate(model, test_dataset, test_dataloader)

model.load_state_dict(torch.load(args.output_dir+"/"+'parameter'+str(last_epoch)+".pkl"))
mi_f1, _ = evaluate(model, test_dataset, test_dataloader)

print (mi_f1)