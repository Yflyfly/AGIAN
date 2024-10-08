import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # （代表仅使用第0，1号GPU）


class MultiHeadGATLayer(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.attn_kernel_initializer = attn_kernel_initializer

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.attn_kernel_regularizer = attn_kernel_regularizer
        self.activity_regularizer = activity_regularizer

        self.kernels = []
        self.biases = []
        self.atten_kernels = []

        super(MultiHeadGATLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        for head in range(self.attn_heads):
            kernel = self.add_weight(shape=(self.in_dim, self.out_dim),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            if self.use_bias:
                bias = self.add_weight(shape=(self.out_dim,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            atten_kernel = self.add_weight(shape=(2 * self.out_dim, 1),
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           name='kernel_{}'.format(head))
            self.atten_kernels.append(atten_kernel)

        self.built = True

    def call(self, inputs, training=False):
        X = inputs[0]
        A = inputs[1]
        # print(X.shape, A.shape)

        N = A.shape[-1]
        # print(N)

        outputs = []
        for head in range(self.attn_heads):

            kernel = self.kernels[head]

            features = tf.matmul(X, kernel)
            # print(0, features.shape)

            concat_features = tf.concat([
                tf.reshape(tf.tile(features, [1, 1, N]), [-1, N * N, self.out_dim]),
                tf.tile(features, [1, N, 1])
            ], axis=1)
            # print(tf.reshape(tf.tile(features, [1, N]), [N * N, -1]))
            # print(1, concat_features.shape)

            concat_features = tf.reshape(concat_features, [-1, N, N, 2 * self.out_dim])
            # print(2, concat_features.shape)

            atten_kernel = self.atten_kernels[head]
            # print(3, atten_kernel.shape)

            dense = tf.matmul(concat_features, atten_kernel)
            # print(4, dense.shape)

            dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)
            # print(5, dense.shape)

            # dense = tf.reshape(dense1, [N, -1])
            # print(6, dense)
            dense = tf.reduce_mean(dense, axis=-1)
            # print(6, dense.shape)

            zero_vec = -9e15 * tf.ones_like(dense)
            # print(7, zero_vec.shape)
            attention = tf.where(A > 0, dense, zero_vec)

            dense = tf.keras.activations.softmax(attention, axis=-1)

            dropout_attn = tf.keras.layers.Dropout(self.dropout_rate)(dense, training=training)
            dropout_feat = tf.keras.layers.Dropout(self.dropout_rate)(features, training=training)

            node_features = tf.matmul(dropout_attn, dropout_feat)

            if self.use_bias:
                node_features = tf.add(node_features, self.biases[head])

            outputs.append(node_features)

        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=-1)
        else:
            output = tf.reduce_mean(tf.stack(outputs), axis=-1)

        if self.activation is not None:
            output = self.activation(output)

        return output


class GraphAttentionModel(tf.keras.Model):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GraphAttentionModel, self).__init__()

        self.attention_layer1 = MultiHeadGATLayer(in_dim, hidden_dim, attn_heads=num_heads,
                                                  activation=tf.keras.activations.elu)

        self.attention_layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, attn_heads=1)

    def call(self, x, training=False):
        adj = x[1]

        x = self.attention_layer1(x, training)

        output = self.attention_layer2([x, adj], training)

        return output
