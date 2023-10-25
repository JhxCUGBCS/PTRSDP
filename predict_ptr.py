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


def evaluate(model, dataset, dataloader):
    model.eval()
    scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model(**batch)
            res = []
            for i in dataset.prompt_id_2_label:
                print("显示单个的prompt_id_2_label")
                print(i)
                print("============="+'\n')
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]
                _res = _res.detach().cpu()
                res.append(_res)
            logits = torch.stack(res, 0).transpose(1,0)
            labels = batch['labels'].detach().cpu().tolist()
            all_labels+=labels
            scores.append(logits.cpu().detach())
        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)
        np.save("scores.npy", scores)
        np.save("all_labels.npy", all_labels)

        pred = np.argmax(scores, axis = -1)

        print(pred)
        print("pred结果显示完毕"+'\n')
        return pred

args = get_args_parser()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(args.seed)

tokenizer=get_tokenizer(special=[])
temps=get_temps(tokenizer)

dataset = REPromptDataset(
    path  = args.data_dir,
    name = 'test.txt',
    rel2id = args.data_dir + "/" + "rel2id.json",
    temps = temps,
    tokenizer = tokenizer)
dataset.save(path = args.output_dir, name = "test")

test_dataset = REPromptDataset.load(
    path = args.output_dir,
    name = "test",
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")

train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

test_dataset.cuda()
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size//2)

model=get_model(tokenizer,test_dataset.prompt_label_idx)
model.load_state_dict(torch.load("datasets/tacred/k-shot/parameter4.pkl")).cuda()

outs=evaluate(model,test_dataset,test_dataloader)
