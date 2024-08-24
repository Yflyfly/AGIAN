import json
import os
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from transformers import BertTokenizer, TFBertModel
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

from AIGAN.GAT import MultiHeadGATLayer
from BERT_utils import convert_example_to_feature, read_dataset, map_emb_to_dict
from GCN import GCNConv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # （代表仅使用第0，1号GPU）

pre_model_path = './file_PTMs/bert-base-uncased'
max_length = 128
aspect_max_length = 8
type_nums = 3
batch_size = 16
learning_rate = 2e-5
epochs = 10
heads = 3
layers = 2
train_path = './Datasets/processed_data/2014/r_train.csv'
test_path = './Datasets/processed_data/2014/r_test.csv'
filepath = './model/AIGAN-BERT_' + str(heads) + 'H' + str(layers) + 'L_{epoch:02d}.pb'  # 模型保存路径
train_graph_path = './Datasets/bert_vocab_graph/2014/r_train_graph_tfidf.json'
test_graph_path = './Datasets/bert_vocab_graph/2014/r_test_graph_tfidf.json'

# 加载分词器和bert
tokenizer = BertTokenizer.from_pretrained(pre_model_path)
bert = TFBertModel.from_pretrained(pre_model_path)

# 读取数据集
train_data = read_dataset(train_path)
test_data = read_dataset(test_path)

# 邻接矩阵处理
edges_value = [1.0] * max_length
ones_matrix = tf.eye(max_length+1, dtype=tf.float32)
aware_node = [1.0] * batch_size
aware_node = tf.constant(aware_node, tf.float32, (batch_size, 1, 1))


def split_str(aspect):
    word_list = aspect.split('/')
    return ' '.join(word_list)


# 构建输入数据集的例子
def encode_dataset(dataset, graph):
    input_ids_list = []
    attention_mask_list = []
    aspect_input_ids_list = []
    aspect_attention_mask_list = []
    label_list = []
    f = open(graph, "r")
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

        review = row["clean_text"]
        if review == '':
            continue
        category = split_str(row["category"])
        label = row["polarity_id"]

        graphs_list.append(edge_list)
        relation_tfidf_list.append(tfidf_list)
        # 将数据集中每一行数据映射token
        review_input = convert_example_to_feature(review, tokenizer, max_length)
        aspect_input = convert_example_to_feature(category, tokenizer, aspect_max_length)
        input_ids_list.append(review_input['input_ids'])
        attention_mask_list.append(review_input['attention_mask'])
        aspect_input_ids_list.append(aspect_input['input_ids'])
        aspect_attention_mask_list.append(aspect_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, aspect_input_ids_list, aspect_attention_mask_list, graphs_list, relation_tfidf_list, label_list)
    ).map(map_emb_to_dict)


ds_train_encoded = encode_dataset(train_data, train_graph_path).padded_batch(batch_size, drop_remainder=True)
ds_test_encoded = encode_dataset(test_data, test_graph_path).padded_batch(batch_size, drop_remainder=True)


class AGIAN_BERT(object):
    def __init__(self, label_num):
        self.label_num = label_num

    def get_model(self):
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_masks = Input(shape=(None,), dtype=tf.int32, name="attention_masks")
        aspect_input_ids = Input(shape=(None,), dtype=tf.int32, name="aspect_input_ids")
        aspect_attention_mask = Input(shape=(None,), dtype=tf.int32, name="aspect_attention_masks")
        graphs = Input(shape=(max_length, 2), dtype=tf.int32, name="graphs")
        relation_value = Input(shape=(max_length), dtype=tf.float32, name="tfidf")

        # bert_embedding
        outputs = bert(input_ids, attention_mask=attention_masks)
        token_outputs = outputs[0]
        target_output = bert(aspect_input_ids, attention_mask=aspect_attention_mask)[0]
        # target_avg = tf.expand_dims(target_avg, axis=1)
        aspect_input_ids = tf.expand_dims(aspect_input_ids, axis=2)
        target_ones = tf.ones_like(aspect_input_ids)
        target_mask = tf.where(aspect_input_ids == 0, aspect_input_ids, target_ones)
        target_avg = tf.reduce_sum(target_output, axis=1, keepdims=True) / tf.reduce_sum(
            tf.cast(target_mask, dtype=tf.float32), axis=1, keepdims=True)

        # mask_ids = tf.slice(input_ids, [0, aspect_max_length], [batch_size, max_length])
        sim_scores = tf.matmul(target_avg, token_outputs, transpose_b=True)
        sim_zeros = -9e15 * tf.ones_like(sim_scores)
        sim_scores = tf.where(tf.expand_dims(input_ids, axis=1) != 0, sim_scores, sim_zeros)
        sim = tf.nn.softmax(sim_scores, axis=2)
        sim = tf.concat([sim, aware_node], axis=2)

        # 构造邻接矩阵并归一化
        adjacency_matrix = []
        adjacency_matrix_weight = []
        for j in range(batch_size):
            edges = tf.slice(graphs, [j, 0, 0], [1, max_length, 2])
            edges = tf.reduce_mean(edges, axis=0)
            # 函数用于将输入解释为矩阵
            sparse_adj = tf.SparseTensor(tf.cast(edges, tf.int64), edges_value,
                                         [max_length + 1, max_length + 1])
            sparse_adj = tf.sparse.to_dense(tf.sparse.reorder(sparse_adj))
            # 有向图
            ones = ones_matrix - (sparse_adj * ones_matrix)
            sparse_adj = sparse_adj + ones
            sum_ones = tf.reduce_sum(ones, axis=0, keepdims=True)
            adj = tf.concat([tf.slice(sparse_adj, [0, 0], [max_length, max_length + 1]), sum_ones], axis=0)
            adjacency_matrix.append(adj)

            # 构建tfidf和相似度邻接矩阵
            edges_tfidf = tf.slice(relation_value, [j, 0], [1, max_length])
            edges_tfidf = tf.reduce_mean(edges_tfidf, axis=0)
            tfidf_adj = tf.SparseTensor(tf.cast(edges, tf.int64), edges_tfidf,
                                        [max_length + 1, max_length + 1])
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
            [tf.slice(emb_adjacency_weight, [0, 0, 0], [batch_size, max_length, max_length + 1]), sim], axis=1)
        emb_adjacency_weight_t = tf.transpose(emb_adjacency_weight, perm=[0, 2, 1])

        # 句尾拼接category
        emb_sentence_target = tf.concat([token_outputs, target_avg], 1)

        # GCN
        GCN = GCNConv(384, activation='relu', name="GCN_category")([emb_adjacency_weight, emb_sentence_target])
        GCN_t = GCNConv(384, activation='relu', name="GCN_category_t")([emb_adjacency_weight_t, emb_sentence_target])
        if layers > 1:
            for l in range(layers - 1):
                GCN = GCNConv(384, activation='relu')([emb_adjacency_weight, GCN])
                GCN_t = GCNConv(384, activation='relu')([emb_adjacency_weight_t, GCN_t])
        BiGCN = tf.concat([GCN, GCN_t], 2)

        GCN_sentence = tf.slice(BiGCN, [0, 0, 0], [batch_size, max_length, 768], name="s1")
        GCN_category = tf.slice(BiGCN, [0, max_length, 0], [batch_size, 1, 768], name="s2")

        # GAT聚合category特征
        GAT = MultiHeadGATLayer(768, 384, attn_heads=heads, activation=tf.keras.activations.relu)(
            [emb_sentence_target, emb_adjacency]
        )
        GAT_t = MultiHeadGATLayer(768, 384, attn_heads=heads, activation=tf.keras.activations.relu)(
            [emb_sentence_target, emb_adjacency_t]
        )

        BiGAT = tf.concat([GAT, GAT_t], 2)

        GAT_sentence = tf.slice(BiGAT, [0, 0, 0], [batch_size, max_length, 768 * heads], name="s3")
        GAT_category = tf.slice(BiGAT, [0, max_length, 0], [batch_size, 1, 768 * heads], name="s4")
        # lstm = LSTM(units=200, dropout=0.2, recurrent_dropout=0.1, return_sequences=True, name='LSTM')(GAT_sentence)

        # 交互attention GCN to GAT
        V = tf.keras.layers.Dense(1)
        W1 = tf.keras.layers.Dense(768)(GAT_sentence)
        GCN_category_tile = tf.tile(GCN_category, (1, max_length, 1))
        W2 = tf.keras.layers.Dense(768)(GCN_category_tile)
        GAT_score = V(tf.nn.tanh(tf.concat([W1, W2], 2)))
        GAT_score = tf.reduce_mean(GAT_score, axis=2)
        zero_vec = -9e15 * tf.ones_like(GAT_score)
        GAT_score = tf.where(input_ids != 0, GAT_score, zero_vec)
        GAT_attention_weights = tf.nn.softmax(GAT_score, axis=1)
        GAT_vector = tf.matmul(tf.expand_dims(GAT_attention_weights, axis=1), GAT_sentence)
        # GAT_vector = tf.reduce_mean(GAT_vector, axis=1)

        # 交互attention GAT to GCN
        V1 = tf.keras.layers.Dense(1)
        W11 = tf.keras.layers.Dense(768)(GCN_sentence)
        GAT_category_tile = tf.tile(GAT_category, (1, max_length, 1))
        W22 = tf.keras.layers.Dense(768)(GAT_category_tile)
        GCN_score = V1(tf.nn.tanh(tf.concat([W11, W22], 2)))
        GCN_score = tf.reduce_mean(GCN_score, axis=2)
        zero_vec1 = -9e15 * tf.ones_like(GCN_score)
        GCN_score = tf.where(input_ids != 0, GCN_score, zero_vec1)
        GCN_attention_weights = tf.nn.softmax(GCN_score, axis=1)
        GCN_vector = tf.matmul(tf.expand_dims(GCN_attention_weights, axis=1), GCN_sentence)

        GAT_vector = Dense(768)(GAT_vector)
        context_vector = tf.concat([GAT_vector, GCN_vector], 2)
        context_vector = tf.reduce_mean(context_vector, axis=1)


        cla_outputs = Dense(self.label_num, activation='softmax')(context_vector)
        model = Model(
            inputs={
                "input_ids": input_ids,
                "attention_masks": attention_masks,
                "aspect_input_ids": aspect_input_ids,
                "aspect_attention_masks": aspect_attention_mask,
                "graphs": graphs,
                "relation_value": relation_value
            },
            outputs=[cla_outputs])

        return model


model = AGIAN_BERT(type_nums).get_model()
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
                             monitor='val_sparse_categorical_accuracy',
                             verbose=1,
                             save_weights_only=True,
                             save_best_only=True,
                             mode='max')

bert_history = model.fit(ds_train_encoded,
                         epochs=epochs,
                         validation_data=ds_test_encoded,
                         callbacks=[checkpoint, early_stopping]
                         )


