import json
import math
import re
import pandas as pd
import spacy
from spacy.tokens import Doc
from gensim import corpora
from gensim import models
from transformers import BertTokenizer

max_length = 128


# 根据句子语法关系创建依赖关系树
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def create_dependency_tree(text):
    #  http://localhost:5000
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    tokens = nlp(text)
    edges_list = []  # 依存关系边
    edges_relation_list = []  # 依存关系
    for token in tokens:
        edges_list.append([token.i, token.head.i])
        edges_relation_list.append(token.dep_)
    # print(edges_relation_list)
    # print(edges_list)
    return edges_list, edges_relation_list


def create_bert_dependency_tree(text):
    pre_model_path = '../file_PTMs/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pre_model_path)
    tokens = tokenizer.tokenize(text)
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    # if len(tokens) > 63:
    #     print(len(tokens))
    #     tokens = tokens[:62]

    tokens = nlp(' '.join(tokens))
    edges_list = []  # 依存关系边
    edges_relation_list = []  # 依存关系
    for token in tokens:
        edges_list.append([token.i, token.head.i])
        edges_relation_list.append(token.dep_)

    return edges_list, edges_relation_list


# 转换为以bert为基础的依赖树
def convert_to_bert_dependency_tree(text):
    text = 'the filet mignon is awesome along with everything else on the menu'
    pre_model_path = '../file_PTMs/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pre_model_path)
    tokens = tokenizer.tokenize(text)
    print(text)
    print(tokens)
    first_str = [t[0] for t in tokens]
    for i in range(len(first_str)):
        if first_str[i] == '#':
            later_str = first_str[i:]
            for li in range(len(later_str)):
                if later_str[li] != '#':
                    break
            print(i-1)
            i = i + li
            print(i-1)


# convert_to_bert_dependency_tree('ddd')


# 创建图的json格式 保存id，边和关系
def create_graph_json(dataset_path, json_path):
    # 读取+去重
    dataset = pd.read_csv(dataset_path, keep_default_na=False)
    id_text = dataset[["id", "clean_text"]]
    df = id_text.drop_duplicates(subset=["id"])
    ids = list(df["id"])
    graphs_list = []
    for index, row in df.iterrows():
        text = row["clean_text"]
        # a, b = create_dependency_tree(text)
        a, b = create_bert_dependency_tree(text)
        graph_dict = {
            "edges_list": a,
            "relation_list": b
        }
        graphs_list.append(graph_dict)

    graphs_dict = dict(zip(ids, graphs_list))

    graph_json = json.dumps(graphs_dict, indent=4)
    f = open(json_path, "w")
    f.write(graph_json)
    f.close()


# create_graph_json("./processed_data/2016/r_train.csv", "./bert_vocab_graph/2016/r_train_graph.json")


# 使用训练集创建tfidf模型
def generate_tfidf(train_path, dic_path, tifidf_path):
    f_train = open(train_path, "r")
    train_graph_dict = json.load(f_train)
    graphs_list = []
    for graph_dict in train_graph_dict.values():
        graphs_list.append(graph_dict["relation_list"])
    # 赋给语料库中每个词(不重复的词)一个整数id
    dic = corpora.Dictionary(graphs_list)
    dic.save(dic_path)
    # print(dic, type(dic))
    new_corpus = [dic.doc2bow(words) for words in graphs_list]
    # print(dic.token2id)

    tfidf = models.TfidfModel(new_corpus)
    tfidf.save(tifidf_path)


# generate_tfidf("./bert_vocab_graph/2016/r_train_graph.json",
#                "./bert_vocab_graph/2016/r_tfidf.dict",
#                "./bert_vocab_graph/2016/r_tfidf.model")


# 关系映射为id,并且对应tfidf值
def relation_to_id(json_path, dic_path, tifidf_path, save_path):
    f = open(json_path, "r")
    graphs_dict = json.load(f)
    graphs_list = []
    for graph_dict in graphs_dict.values():
        graphs_list.append(graph_dict["relation_list"])
    dic = corpora.Dictionary.load(dic_path)
    id_dic = dic.token2id
    # print(id_dic)
    id_lists = []
    for g in graphs_list:
        id_list = []
        for r in g:
            if r in id_dic.keys():
                v = id_dic[r]
            else:
                v = -1
            id_list.append(v)
        id_lists.append(id_list)

    # 载入模型
    tfidf = models.TfidfModel.load(tifidf_path)
    v_lists = []
    for graph in range(len(graphs_list)):
        count = dic.doc2bow(graphs_list[graph])
        tfidf_values = dict(tfidf[count])
        v_list = []
        for id in id_lists[graph]:
            if id == 0 or id == -1:
                v = float(1)
            else:
                v = tfidf_values[id]
            v = math.exp(v)
            v_list.append(v)
        v_lists.append(v_list)

    g_list = list(graphs_dict.values())
    for i in range(len(g_list)):
        g_list[i]["relation_tfidf"] = v_lists[i]
    graph_json = json.dumps(graphs_dict, indent=4)
    f = open(save_path, "w")
    f.write(graph_json)
    f.close()


relation_to_id("./bert_vocab_graph/2016/r_test_graph.json",
               './bert_vocab_graph/2016/r_tfidf.dict',
               './bert_vocab_graph/2016/r_tfidf.model',
               "./bert_vocab_graph/2016/r_test_graph_tfidf.json")

