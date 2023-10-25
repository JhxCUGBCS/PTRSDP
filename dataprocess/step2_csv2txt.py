"""
time:20220413
jhx
用作把实验室之前标注的数据处理成template
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import json


# 读取csv文件中的数据
def data_csv_read(path):
    csv_path = path

    data_csv = pd.read_csv(csv_path,encoding='utf-8',header=None)  #原数据无表头
    data_array = np.array(data_csv)
    data_list = data_array.tolist()
    #print(type(data_csv))
    #print(data_csv)
    #print("======================")
    #print(type(data_list))
    #print(data_list)
    #print('\n')
    #print(data_list[0])
    return data_list




#把列表中的数据处理成template的形式
def data_temp(data:list,label:str) ->list:
    data_1 = data

    res_temp_data=[]
    for i in data_1:
        try:
            info={}
            #print(i)
            #print('\n')
            info['token']=list(i[3])

            info['h']={}
            info['h']['name']=i[1]
            info['h']['pos']=search_index(i[3],i[1])

            info['t']={}
            info['t']['name']=i[0]
            info['t']['pos']=search_index(i[3],i[0])

            info['relation']=label

            res_temp_data.append(info)
        except:
            continue

    return res_temp_data




#查找字符串a在字符串b的位置
def search_index(str1:str,str2:str)->list:

    res=[]
    long_s=str1
    short_s=str2
    len1=len(long_s)
    len2=len(short_s)

    for i in range(len1-len2+1):
        if long_s[i]==short_s[0] and long_s[i:i+len2]==short_s:
            res.append(i)

    result=[]
    result.append(res[0])
    result.append(int(res[0])+len2-1)

    return result








if __name__=="__main__":
    rel_dict={"类：实_例_是": 1, "舰：操_作_是": 2, "舰：发_生_国": 3, "舰：参_战_于": 4, "舰：装_备_有": 5}

    for num in range(1,6):
        csv_path = "orig_data/军事数据集/dataset_label"+str(num)+".csv"
        txt_path = "results"+'/dataset_txt/'+str(num)+".txt"

        for key in rel_dict:
            if int(rel_dict[key])==num:
                label=key

        data_list=data_csv_read(csv_path)

        with open(file=txt_path,mode='a',encoding='utf-8')as txt_file:
            for i in data_temp(data_list,label):
                json.dump(i,txt_file,ensure_ascii=False)
                txt_file.write('\n')
            txt_file.close()
