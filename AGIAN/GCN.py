import tensorflow as tf
from tensorflow.python.keras import activations, regularizers, constraints, initializers

spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul


class GCNConv(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(GCNConv, self).__init__()
        # 初始化不需要训练的参数
        self.units = units
        # activation=None 使用线性激活函数（等价不使用激活函数）
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        # 初始化方法定义了对Keras层设置初始化权重（bias）的方法 glorot_uniform
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # 加载正则化的方法
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # 约束：对权重值施加约束的函数。
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        # gsize = input_shape[0][0]  # graph size
        fdim = input_shape[1][-1]  # feature dim
        # fdim = 300

        # hasattr 检查该对象self是否有某个属性'weight'
        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name="weight",
                                          shape=(fdim, self.units),
                                          initializer=self.kernel_initializer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias = self.add_weight(name="bias",
                                            shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
        # super(GCNConv, self).build(input_shape)

    def call(self, inputs):
        """ GCN has two inputs : [An, X]
            对称归一化版本的GCN的核心公式计算过程
        """
        self.An = inputs[0]
        self.X = inputs[1]

        # isinstance 函数来判断一个对象是否是一个已知的类型
        if isinstance(self.X, tf.SparseTensor):
            h = spdot(self.X, self.weight)
        else:
            # 二维数组矩阵之间的dot函数运算得到的乘积是矩阵乘积
            # h = dot(self.X, self.weight)
            h = tf.matmul(self.X, self.weight)
        # output = dot(self.An, h)
        # _dseq = tf.math.reduce_sum(self.An, axis=2, keepdims=True)
        # p = tf.pow(_dseq, -0.5)
        # _D_half = tf.linalg.diag(p)  # 开平方构成对角矩阵
        # adj_normalized = _D_half @ tf.cast(self.An, dtype=tf.float32) @ _D_half  # 矩阵乘法
        # output = tf.matmul(self.An, h) / _dseq
        output = tf.matmul(self.An, h)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)

        return output

