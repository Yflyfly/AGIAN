import json
import os
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import LSTM

from GCN import GCNConv
from GAT import MultiHeadGATLayer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # （代表仅使用第0，1号GPU）


max_length = 128
aspect_max_length = 8
type_nums = 3
BatchSize = 32
learning_rate = 1e-3
epochs = 20
heads = 3
layers = 2

train_path = './Datasets/processed_data/2014/r_train.csv'
test_path = './Datasets/processed_data/2014/r_test.csv'
filepath = './model/AIGAN_' + str(heads) + 'H' + str(layers) + 'L_{epoch:02d}.pb'  # 模型保存路径
train_graph_path = './Datasets/processed_data/2014/graph/r_train_graph_tfidf.json'
test_graph_path = './Datasets/processed_data/2014/graph/r_test_graph_tfidf.json'
WordVectorPath = './file_WordVector/glove.840B.300d.word2vec.txt'

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(WordVectorPath, binary=False, limit=200000)
vocab_list = [word for word, Vocab in w2v_model.vocab.items()]

words_index = {' ': 0}  # 存储单词到序号的映射
embeddings_matrix = np.zeros((len(vocab_list)+1, w2v_model.vector_size))  # 序号到词向量的映射，实际上就是个矩阵而已
# 构建映射关系，注意腾出序号0存放空的向量，用于表示词库以外的陌生词
for i in range(len(vocab_list)):
    word = vocab_list[i]
    words_index[word] = i+1
    embeddings_matrix[i+1] = w2v_model[word]

# 邻接矩阵处理
edges_value = [1.0] * max_length
eyes_matrix = tf.eye(max_length+1, dtype=tf.float32)
aware_node = [1.0] * BatchSize
aware_node = tf.constant(aware_node, tf.float32, (BatchSize, 1, 1))


# 将语句转化为序号列表
def tokenizer(texts, word_index, max_len):
    tokenized = []
    for sentence in texts:
        new_txt = []
        for word in sentence.split():  # 切分语句
            try:
                new_txt.append(word_index[word])
            except:
                # 陌生词判定为0
                new_txt.append(0)
        tokenized.append(new_txt)
    # 将语句对应的序号列表整成同一长度，这样一来，短的语句会被补零，长的会被截断，
    tokenized = pad_sequences(tokenized, maxlen=max_len, padding='post', truncating='post')  # 这里使用了后置补零截断的方式
    return tokenized


# 映射datasets
def map_emb_to_dict(text_ids, target_ids, graph_edges, relation_values, labels):
    return {
        "text_ids": text_ids,
        "target_ids": target_ids,
        "graph_edges": graph_edges,
        "relation_values": relation_values
    }, labels


# 分割字符串
def split_str(aspect):
    word_list = aspect.split('/')
    return ' '.join(word_list)


# 构建输入数据集的例子
def encode_dataset(dataset, graph_path):
    cleaned_text = dataset["clean_text"]
    x = tokenizer(cleaned_text, words_index, max_length)
    y = dataset["polarity_id"]
    f = open(graph_path, "r")
    graphs_dict = json.load(f)
    graphs_list = []
    relation_tfidf_list = []
    for index, row in dataset.iterrows():
        text_id = row["id"]
        graph_dict = graphs_dict[str(text_id)]
        edge_list = graph_dict["edges_list"]
        tfidf_list = graph_dict["relation_tfidf"]
        if len(edge_list) < max_length:
            edge_list.extend([[k, k] for k in range(len(edge_list), max_length)])
            tfidf_list.extend([1.0] * (max_length - len(tfidf_list)))
        else:
            edge_list = edge_list[:max_length]
            tfidf_list = tfidf_list[:max_length]
        graphs_list.append(edge_list)
        relation_tfidf_list.append(tfidf_list)

    aspect_words = dataset["category"].apply(split_str)
    target_ids = tokenizer(aspect_words, words_index, aspect_max_length)

    return tf.data.Dataset.from_tensor_slices((x, target_ids, graphs_list, relation_tfidf_list, y)).map(map_emb_to_dict)


# 读取数据
raw_train = pd.read_csv(train_path, keep_default_na=False)
# raw_test = pd.read_csv(test_path, keep_default_na=False)
raw_train = shuffle(raw_train)
raw_test = raw_train.sample(frac=0.1, random_state=9, axis=0)
raw_train = raw_train.loc[list(set(raw_train.index)-set(raw_test.index))]
# 转为tf.dataset格式
train = encode_dataset(raw_train, train_graph_path).batch(BatchSize, drop_remainder=True)
test = encode_dataset(raw_test, train_graph_path).batch(BatchSize, drop_remainder=True)


class AGIAN(object):
    def __init__(self, label_num):
        self.label_num = label_num

    def get_model(self):
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        target_ids = Input(shape=(None,), dtype=tf.int32, name="target_ids")
        graph = Input(shape=(max_length, 2), dtype=tf.int32, name="graph")
        relation_value = Input(shape=(max_length), dtype=tf.float32, name="tfidf")

        emb_sentence = Embedding(
            input_dim=len(embeddings_matrix),
            output_dim=300,
            weights=[embeddings_matrix],
            input_length=max_length,
            trainable=False
        )(input_ids)

        emb_target = Embedding(
            input_dim=len(embeddings_matrix),
            output_dim=300,
            weights=[embeddings_matrix],
            input_length=aspect_max_length,
            trainable=False
        )(target_ids)

        # 平均方面词
        target_zeros = tf.ones_like(emb_target)
        target_mask = tf.where(emb_target == 0, emb_target, target_zeros)
        target_avg = tf.reduce_sum(emb_target, axis=1, keepdims=True) / tf.reduce_sum(target_mask, axis=1, keepdims=True)

        # 计算相似度，初始化方面感知词的关联度
        sim_scores = tf.matmul(target_avg, emb_sentence, transpose_b=True)
        sim_zeros = -9e15 * tf.ones_like(sim_scores)
        sim_scores = tf.where(tf.expand_dims(input_ids, axis=1) != 0, sim_scores, sim_zeros)
        sim = tf.nn.softmax(sim_scores, axis=2)
        sim = tf.concat([sim, aware_node], axis=2)

        # 构造邻接矩阵并归一化
        adjacency_matrix = []
        adjacency_matrix_weight = []
        for j in range(BatchSize):
            edges = tf.slice(graph, [j, 0, 0], [1, max_length, 2])
            edges = tf.reduce_mean(edges, axis=0)
            # 函数用于将输入解释为矩阵
            sparse_adj = tf.SparseTensor(tf.cast(edges, tf.int64), edges_value, [max_length + 1, max_length + 1])
            sparse_adj = tf.sparse.to_dense(tf.sparse.reorder(sparse_adj))
            # 有向图
            ones = eyes_matrix - (sparse_adj * eyes_matrix)
            sparse_adj = sparse_adj + ones
            sum_ones = tf.reduce_sum(ones, axis=0, keepdims=True)
            adj = tf.concat([tf.slice(sparse_adj, [0, 0], [max_length, max_length+1]), sum_ones], axis=0)
            adjacency_matrix.append(adj)

            # 构建tfidf和相似度邻接矩阵
            edges_tfidf = tf.slice(relation_value, [j, 0], [1, max_length])
            edges_tfidf = tf.reduce_mean(edges_tfidf, axis=0)
            tfidf_adj = tf.SparseTensor(tf.cast(edges, tf.int64), edges_tfidf, [max_length + 1, max_length + 1])
            tfidf_adj = tf.sparse.to_dense(tf.sparse.reorder(tfidf_adj))
            tfidf_adj = tfidf_adj + ones
            # tfidf_adj = tf.concat([tf.slice(tfidf_adj, [0, 0], [64, 65]), sum_ones], axis=0)
            adjacency_matrix_weight.append(tfidf_adj)

        # 全一邻接矩阵
        emb_adjacency = tf.stack(adjacency_matrix)
        emb_adjacency_t = tf.transpose(emb_adjacency, perm=[0, 2, 1])

        # 带权邻接矩阵
        emb_adjacency_weight = tf.stack(adjacency_matrix_weight)
        emb_adjacency_weight = tf.concat(
            [tf.slice(emb_adjacency_weight, [0, 0, 0], [BatchSize, max_length, max_length+1]), sim], axis=1)
        emb_adjacency_weight_t = tf.transpose(emb_adjacency_weight, perm=[0, 2, 1])

        # 句尾拼接category
        emb_sentence = LSTM(units=300, dropout=0.2, return_sequences=True, name='LSTM')(emb_sentence)
        emb_sentence_target = tf.concat([emb_sentence, target_avg], 1)

        # GCN
        GCN = GCNConv(150, activation='relu', name="GCN_category")([emb_adjacency_weight, emb_sentence_target])
        GCN_t = GCNConv(150, activation='relu', name="GCN_category_t")([emb_adjacency_weight_t, emb_sentence_target])
        if layers > 1:
            for l in range(layers-1):
                GCN = GCNConv(150, activation='relu')([emb_adjacency_weight, GCN])
                GCN_t = GCNConv(150, activation='relu')([emb_adjacency_weight_t, GCN_t])
        BiGCN = tf.concat([GCN, GCN_t], 2)

        GCN_sentence = tf.slice(BiGCN, [0, 0, 0], [BatchSize, max_length, 300], name="s1")
        GCN_category = tf.slice(BiGCN, [0, max_length, 0], [BatchSize, 1, 300], name="s2")

        # GAT聚合category特征
        GAT = MultiHeadGATLayer(300, 150, attn_heads=heads, activation=tf.keras.activations.relu)(
            [emb_sentence_target, emb_adjacency]
        )
        GAT_t = MultiHeadGATLayer(300, 150, attn_heads=heads, activation=tf.keras.activations.relu)(
            [emb_sentence_target, emb_adjacency_t]
        )

        BiGAT = tf.concat([GAT, GAT_t], 2)

        GAT_sentence = tf.slice(BiGAT, [0, 0, 0], [BatchSize, max_length, 300 * heads], name="s3")
        GAT_category = tf.slice(BiGAT, [0, max_length, 0], [BatchSize, 1, 300 * heads], name="s4")
        # lstm = LSTM(units=200, dropout=0.2, recurrent_dropout=0.1, return_sequences=True, name='LSTM')(GAT_sentence)

        # 交互attention GCN to GAT
        V = tf.keras.layers.Dense(1)
        W1 = tf.keras.layers.Dense(300)(GAT_sentence)
        GCN_category_tile = tf.tile(GCN_category, (1, max_length, 1))
        W2 = tf.keras.layers.Dense(300)(GCN_category_tile)
        GAT_score = V(tf.nn.tanh(tf.concat([W1, W2], 2)))
        GAT_score = tf.reduce_mean(GAT_score, axis=2)
        zero_vec = -9e15 * tf.ones_like(GAT_score)
        GAT_score = tf.where(input_ids != 0, GAT_score, zero_vec)
        GAT_attention_weights = tf.nn.softmax(GAT_score, axis=1)
        GAT_vector = tf.matmul(tf.expand_dims(GAT_attention_weights, axis=1), GAT_sentence)
        # GAT_vector = tf.reduce_mean(GAT_vector, axis=1)

        # 交互attention GAT to GCN
        V1 = tf.keras.layers.Dense(1)
        W11 = tf.keras.layers.Dense(300)(GCN_sentence)
        GAT_category_tile = tf.tile(GAT_category, (1, max_length, 1))
        W22 = tf.keras.layers.Dense(300)(GAT_category_tile)
        GCN_score = V1(tf.nn.tanh(tf.concat([W11, W22], 2)))
        GCN_score = tf.reduce_mean(GCN_score, axis=2)
        zero_vec1 = -9e15 * tf.ones_like(GCN_score)
        GCN_score = tf.where(input_ids != 0, GCN_score, zero_vec1)
        GCN_attention_weights = tf.nn.softmax(GCN_score, axis=1)
        GCN_vector = tf.matmul(tf.expand_dims(GCN_attention_weights, axis=1), GCN_sentence)

        context_vector = tf.concat([GAT_vector, GCN_vector], 2)
        context_vector = tf.reduce_mean(context_vector, axis=1)

        cla_outputs = Dense(self.label_num, activation='softmax')(context_vector)
        model = Model(
            inputs={
                "text_ids": input_ids,
                "target_ids": target_ids,
                "graph_edges": graph,
                "relation_values": relation_value,
            },
            outputs=cla_outputs)
        # model.summary()
        return model


model = AGIAN(type_nums).get_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=tf.keras.metrics.SparseCategoricalAccuracy()
)

# 添加 EarlyStopping 回调
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,  # 当连续5个epoch val_accuracy 没有提升时停止训练
    mode='max',
    verbose=1
)

checkpoint = ModelCheckpoint(filepath,
                             # monitor='val_accuracy',
                             monitor='val_sparse_categorical_accuracy',
                             verbose=1,
                             save_weights_only=True,
                             save_best_only=True,
                             mode='max')

history = model.fit(train,
                    epochs=epochs,
                    validation_data=test,
                    callbacks=[checkpoint, early_stopping]
                    )
