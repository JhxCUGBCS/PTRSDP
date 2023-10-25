import re
import os



def deleshort(data,leg):
    """
    :param data:
        data:list
        leg:int
    :return:
        list
    删除列表中长度小于指定数值的元素
    """

    res=[i for i in data if len(i)>=leg]

    return res



def delelem(data):
    """
    删除多余的标点符号   出现的无用空格   不出现中文字符的数据行
    :param data: list
    :return: list
    """
    res_del = []

    for i in data:
        # 使用空字符替换掉间隔符
        a = re.sub(r'\s', '', i)

        # 使用精准匹配，匹配连续出现的符号;并用空字符替换他
        b = re.sub(r'\W{2,}', '', a)

        # 使用空字符替换空格
        c = re.sub(r' ', '', b)

        # 删除没有中文的数据行
        if len(re.findall(r"[\u4e00-\u9fa5]", c)) >= 10:
            res_del.append(c)

    return res_del



def sympop(data,sym):
    """
    删除指定的特殊符号
    :param data: list
    :param sym: list or path_of_sym_in_txt
    :return: list
    """
    #获得特殊符号的列表
    if type(sym)==list:
        symlist=sym
    else:
        with open(file=sym,mode='r',encoding='utf-8')as symfile:
            symlist=[]
            for i in symfile.readlines():
                symlist.append(re.sub(r'\s','',i))
                #如果报错把上面这行改回之前的
                #symlist.append(i.replace("\n", ''))



    """
    这里不要使用正则表达式，因为要对列表里的每一个元素都有一个特定的匹配规则
    但是有些字符是特殊的，还需要对特殊的字符进行单独的处理，比较麻烦，所以使用replace 
    """
    res=[]
    for i in data:
        for j in range(len(symlist)):     #要使用索引固定住列表，不然每一次列表都是在变动的，可能一个元素都处理不了
            i=i.replace(symlist[j],'')    #要使用i=...因为replace处理后原字符串不变，不赋值最后等于没处理
        res.append(i)

    return res



def savedata(data,path):
    """
    把处理好的列表进行切分，逐行保存
    :param data: list
    :param path: path_str
    :return: none
    """
    with open(file=path,mode='w',encoding='utf-8')as file:
        for i in data:
            split_list=i.split('。')
            for j in split_list:
                if len(re.findall(r"[\u4e00-\u9fa5]", j)) >= 6:
                    file.write(j+'。'+'\n')



def mulunusechar(path):
    """
    文件流批量生成每本刊物对应的符号
    :param path: str 未清洗文件的文件夹
    :return: none 生成有特殊符号组成的txt文件
    """
    #查找文件下文件名称
    file_names = os.listdir(path)
    for i in file_names:
        file_name = path + '\\' + i
        with open(file=file_name, mode='r', encoding='utf-8')as source_file:
            data = source_file.readlines()

        sympath = path.replace("source", "symbol") + "\\" + i

        unuse_lis = []

        rule_1 = r'\W'  # 匹配非英文和非数字和非中文
        compiled_rule_1 = re.compile(rule_1)

        for line in data:
            no_en_and_da = compiled_rule_1.findall(line)
            no_en_and_da_str = ''.join(no_en_and_da)
            reslis = re.findall(r'^\S', ''.join(re.findall(r'[^\，]', ''.join(re.findall(r'[^\。]', no_en_and_da_str)))))

            unuse_lis.append(reslis)

        res = []
        for i in unuse_lis:
            for j in i:
                res.append(j)
        res = set(res)

        with open(file=sympath, mode='w', encoding='utf-8')as symfile:
            for i in res:
                symfile.write(i)
                symfile.write('\n')



def fileprocess(path):
    """
    使用文件流进行批量处理
    :param path: str 未清洗txt数据的文件夹
    :return: none
    """

    file_names=os.listdir(path)
    for i in file_names:
        file_name = path +'\\'+i
        with open(file=file_name,mode='r',encoding='utf-8')as source_file:
            data = source_file.readlines()


        """生成包含符号的txt文件，生成后要自己去筛选，看哪些是需要删除的特殊符号"""
        sympath = path.replace("source", "symbol") + "\\" + i

        """在选出特殊符号后，对文本进行处理"""
        # 删除较短的文本
        noshort = deleshort(data, 30)
        # 删除多余空格、纯英文、连续标点
        deledlis = delelem(noshort)
        # 删除特殊符号
        delsymlist = sympop(deledlis, sympath)

        """把处理后的数据按照。进行切分，并逐行保存在txt文件中"""
        cleaned_txt = path.replace("source", "cleaned") + "\\" +"cleaned_"+ i
        savedata(delsymlist,cleaned_txt)




if __name__=="__main__":
    path="D:\\DataProcess_code\\dataclean\\source_txt"

    #生成特殊字符的txt文件，在symbol_txt文件夹中
    #mulunusechar(path)
    #生成清洗后的txt文件，在cleaned_txt中
    fileprocess(path)