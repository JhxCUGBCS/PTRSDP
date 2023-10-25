import json



"""
"类：实_例_是":1
舰船 属于 类别

"舰：操_作_是": 2
舰船 归属 军队

"舰：发_生_国":3
舰船 被购买 国家

"舰：参_战_于": 4
舰船 参战 战争

"舰：装_备_有": 5
舰船 装备 武器
"""
rel2ids_dict={"类：实_例_是":1,"舰：操_作_是":2,"舰：发_生_国":3,"舰：参_战_于":4,"舰：装_备_有":5}

rel2ids_path='results/junshi/rel2id.json'
with open(file=rel2ids_path,encoding='utf-8',mode='w')as f:
    json.dump(rel2ids_dict,f,ensure_ascii=False)
    f.close()