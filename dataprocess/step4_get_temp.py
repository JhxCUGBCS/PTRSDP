"""
人工定义的template



关系标签对应
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

import json


"""
#基于自定义规则生成的temp
temp_path='results/template/temp_rule.txt'
temp1=['0','类：实_例_是','舰','船','属','于','类','别']
temp2=['0','舰：操_作_是','舰','船','服','役','军','队']
temp3=['0','舰：发_生_国','国','家','签','订','舰','船']
temp4=['0','舰：参_战_于','舰','船','参','于','战','争']
temp5=['0','舰：装_备_有','舰','船','配','备','武','器']
temps=[temp1,temp2,temp3,temp4,temp5]

with open(file=temp_path,encoding='utf-8',mode='w')as temp_file:
    for i in temps:
        temp_str='\t'.join(i)
        temp_file.write(temp_str)
        temp_file.write('\n')
    temp_file.close()
"""


"""
#基于SDP和GSDMM生成的temp
temp_path='results/template/temp_sdp.txt'
temp1=['0','类：实_例_是','舰','船','属','于','类','别']
temp2=['0','舰：操_作_是','舰','船','编','入','军','队']
temp3=['0','舰：发_生_国','国','家','购','买','舰','船']
temp4=['0','舰：参_战_于','舰','船','参','加','战','争']
temp5=['0','舰：装_备_有','舰','船','装','有','武','器']
temps=[temp1,temp2,temp3,temp4,temp5]

with open(file=temp_path,encoding='utf-8',mode='w')as temp_file:
    for i in temps:
        temp_str='\t'.join(i)
        temp_file.write(temp_str)
        temp_file.write('\n')
    temp_file.close()
"""


#基于SDP and 不进行聚类生成的temp
temp_path='results/template/temp_sdpwithoutgsdmm.txt'
temp1=['0','类：实_例_是','舰','船','是','类','别']
temp2=['0','舰：操_作_是','舰','船','是','军','队']
temp3=['0','舰：发_生_国','国','家','强','舰','船']
temp4=['0','舰：参_战_于','舰','船','是','战','争']
temp5=['0','舰：装_备_有','舰','船','有','武','器']
temps=[temp1,temp2,temp3,temp4,temp5]

with open(file=temp_path,encoding='utf-8',mode='w')as temp_file:
    for i in temps:
        temp_str='\t'.join(i)
        temp_file.write(temp_str)
        temp_file.write('\n')
    temp_file.close()