"""
jhx
20220413
切分数据集
"""
import  os
import json


path = "results/dataset_txt"
files=os.listdir(path)
#print(files)
#print(type(files))


train_list = []
test_list = []
val_list = []

for i in files:
    file_path=path+"/"+i
    with open(file_path,mode='r',encoding='utf-8')as file_txt:
        data_list =[]
        for i in file_txt.readlines():
            data_list.append(json.loads(i))
        #print(type(data_list))
        len_file=len(data_list)
        num1=int(len_file//10)*int(8)
        num2=int(len_file//10)*int(9)
        # num3=int((len_file//4)*3)

        #print(len_file)
        #print(num1)
        #print(num2)
        #print(num3)
        #print('\n')

        for i in data_list[0:num1]:
            train_list.append(i)

        for i in data_list[num1:num2]:
            test_list.append(i)

        for i in data_list[num2:]:
            val_list.append(i)

train_path="results/train.txt"
with open(train_path,mode='a',encoding='utf-8')as train_file:
    for i in train_list:
        json.dump(i,train_file,ensure_ascii=False)
        train_file.write('\n')
    train_file.close()


test_path="results/test.txt"
with open(test_path,mode='a',encoding='utf-8')as test_file:
    for i in test_list:
        json.dump(i,test_file,ensure_ascii=False)
        test_file.write('\n')
    test_file.close()


val_path="results/val.txt"
with open(val_path,mode='a',encoding='utf-8')as val_file:
    for i in val_list:
        json.dump(i,val_file,ensure_ascii=False)
        val_file.write('\n')
    val_file.close()