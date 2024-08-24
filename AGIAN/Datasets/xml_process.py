import json
import re
import xml.etree.cElementTree as ET
import pandas as pd
from sklearn.utils import shuffle


def clean_text(text):
    # eng_stopwords = set(stopwords.words('english'))
    # 过滤标点符号
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 将词汇转为小写，并过滤掉停用词
    text = text.lower().split()
    # text = [word for word in text if word not in eng_stopwords]
    return ' '.join(text)


# 提取ACSA数据集
def get_ACSA_data(raw_xml, new_csv):
    # 读文件，获取根节点
    # 子节点是嵌套的，可以通过索引访问特定的子节点 如：root[0][0].text
    tree = ET.parse(raw_xml)
    root = tree.getroot()
    # print(root.findall('sentence'))
    data = []
    # 获取aspectCategory
    # Element.findall(): 只找到带有标签的元素，该标签是当前元素的直接子元素。
    # Element.find() :找到第一个带有特定标签的子元素。
    # Element.text:访问标签的内容
    # Element.get()：访问标签的属性值
    i = 0
    for sentence in root.findall('sentence'):
        # ids = sentence.get('id')
        i = i + 1
        ids = i
        text = sentence.find('text').text
        c_text = clean_text(text)
        aspectCategories = sentence.find('aspectCategories')
        for aspectCategory in aspectCategories.findall('aspectCategory'):
            category = aspectCategory.get('category')
            polarity = aspectCategory.get('polarity')
            data.append((ids, text, c_text, category, polarity))

    df = pd.DataFrame(data, columns=['id', 'text', 'clean_text', 'category', 'polarity'])
    # isin() 仅保留的内容
    df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
    df['polarity_id'] = df['polarity'].map(
        {'neutral': 0, 'positive': 1, 'negative': 2})
    df.to_csv(new_csv, encoding='utf-8', index=None)


def get_ACSA_data_15r(raw_xml, new_csv):
    # 读文件，获取根节点
    # 子节点是嵌套的，可以通过索引访问特定的子节点 如：root[0][0].text
    tree = ET.parse(raw_xml)
    root = tree.getroot()
    # print(root)
    data = []
    # 获取aspectCategory
    # Element.findall(): 只找到带有标签的元素，该标签是当前元素的直接子元素。
    # Element.find() :找到第一个带有特定标签的子元素。
    # Element.text:访问标签的内容
    # Element.get()：访问标签的属性值
    for review in root.findall('Review'):
        sentences = review.find('sentences')
        for sentence in sentences.findall('sentence'):
            ids = sentence.get('id')
            text = sentence.find('text').text
            # print(text)
            c_text = clean_text(text)
            opinions = sentence.find('Opinions')
            try:
                all_opinions = opinions.findall('Opinion')
                if len(all_opinions) == 1:
                    category = all_opinions[0].get('category')
                    polarity = all_opinions[0].get('polarity')
                    data.append((ids, text, c_text, category, polarity))
                else:
                    cp_s = []
                    for opinion in opinions.findall('Opinion'):
                        c = opinion.get('category')
                        p = opinion.get('polarity')
                        cp = [c, p]
                        cp_s.append(cp)
                    df = pd.DataFrame(data=cp_s, columns=['col1', 'col2'])
                    # 去重+去冲突
                    df = df.drop_duplicates().drop_duplicates(subset=['col1'], keep=False)
                    tuples = [tuple(x) for x in df.values]
                    for t in tuples:
                        data.append((ids, text, c_text) + t)
            except Exception as e:
                pass

    df = pd.DataFrame(data, columns=['id', 'text', 'clean_text', 'category', 'polarity'])
    # isin() 仅保留的内容
    df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
    df['polarity_id'] = df['polarity'].map(
        {'neutral': 0, 'positive': 1, 'negative': 2})
    df.to_csv(new_csv, encoding='utf-8', index=None)


# get_ACSA_data('./raw_data/val.xml', './processed_data/MAMS/MAMS_val.csv')
# get_ACSA_data_15r('./raw_data/16_rest_train.xml', './processed_data/2016/r_train.csv')


def data_distribution(path):
    df = pd.read_csv(path)
    print("数据总量: %d ." % len(df))
    # 数据分布
    d = {
        'category': df['category'].value_counts().index,
        'count': df['category'].value_counts()
    }

    # DataFrame构造数据框
    # reset_index用来重置索引，因为有时候对dataframe做处理后索引可能是乱的。
    # drop=True就是把原来的索引index列去掉，重置index。
    # drop=False就是保留原来的索引，添加重置的index。
    df_doctype = pd.DataFrame(data=d).reset_index(drop=True)
    df = pd.read_csv('./processed_data/2016/l_test.csv')
    print("数据总量: %d ." % len(df))
    # 数据分布
    dv = {
        'category': df['category'].value_counts().index,
        'count': df['category'].value_counts()
    }
    dv_doctype = pd.DataFrame(data=dv).reset_index(drop=True)
    a = list(df_doctype['category'])
    b = list(dv_doctype['category'])
    for i in b:
        if i not in a:
            a.append(i)
    print(len(a))
    index = [i for i in range(len(a))]
    id_triples = dict(zip(a, index))
    new_json_data = json.dumps(id_triples, indent=4, ensure_ascii=False)
    f = open('./processed_data/2016/l_category.json', "w", encoding='utf-8')
    f.write(new_json_data)
    f.close()



# data_distribution('./processed_data/2016/l_train.csv')


# 提取ACSA数据集
def get_ACSA_data_with_AT(raw_xml, new_csv):
    # 读文件，获取根节点
    # 子节点是嵌套的，可以通过索引访问特定的子节点 如：root[0][0].text
    tree = ET.parse(raw_xml)
    root = tree.getroot()
    # print(root.findall('sentence'))
    data = []
    data_ac = []
    # 获取aspectCategory
    # Element.findall(): 只找到带有标签的元素，该标签是当前元素的直接子元素。
    # Element.find() :找到第一个带有特定标签的子元素。
    # Element.text:访问标签的内容
    # Element.get()：访问标签的属性值
    i = 0
    for sentence in root.findall('sentence'):
        ids = sentence.get('id')
        # i = i + 1
        # ids = i
        text = sentence.find('text').text
        c_text = clean_text(text)
        ac_data = []
        at_data = []
        aspectCategories = sentence.find('aspectCategories')
        for aspectCategory in aspectCategories.findall('aspectCategory'):
            category = aspectCategory.get('category')
            polarity = aspectCategory.get('polarity')
            ac_data.append((ids, text, c_text, category, polarity))
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            aspectTerms_list = []
        else:
            aspectTerms_list = aspectTerms.findall('aspectTerm')
        for aspectTerm in aspectTerms_list:
            term = aspectTerm.get('term')
            polarity = aspectTerm.get('polarity')
            at_data.append((ids, text, c_text, term, polarity))
        data_ac.extend(ac_data)
        for ac in ac_data:
            p = len(data)
            for t in range(len(at_data)):
                if ac[4] == at_data[t][4]:
                    new_ac = (ac[0], ac[1], ac[2], ac[3], ac[4], at_data[t][3])
                    data.append(new_ac)
                    at_data.remove(at_data[t])
                    break
            q = len(data)
            if p == q:
                data.append((ac[0], ac[1], ac[2], ac[3], ac[4], 1))

    df = pd.DataFrame(data, columns=['id', 'text', 'clean_text', 'category', 'polarity', 'term'])
    # isin() 仅保留的内容
    df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
    df['polarity_id'] = df['polarity'].map(
        {'neutral': 0, 'positive': 1, 'negative': 2})
    df.to_csv(new_csv, encoding='utf-8', index=None)


# get_ACSA_data_with_AT('./raw_data/14_test.xml', './processed_data/LC/2014/r_test.csv')
