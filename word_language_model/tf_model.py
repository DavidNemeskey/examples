import tensorflow as tf
import tensorflow.contrib.rnn as rnn

class LSTMModel(object):
    """Generic LSTM language model based on the PTB model in tf/models."""
    def __init__(self, is_training, ntoken, nhid, nlayers, batch_size,
                 bptt, clip, dropout=0.5, tie_weights=False):
        self.is_training = is_training

        dims = [batch_size, bptt]
        self._input_data = tf.placeholder(
            tf.int32, dims, name='input_placeholder')
        self._targets = tf.placeholder(
            tf.int32, dims, name='target_placeholder')

        cell = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(nhid, forget_bias=1.0, state_is_tuple=True)
             for _ in range(nlayers)]
        )
        self._initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                'embedding', [ntoken, nhid], trainable=True, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        outputs, state = tf.nn.dynamic_rnn(
            inputs=inputs, cell=cell, dtype=tf.float32,
            initial_state=self._initial_state)
        self._final_state = state

        softmax_w = tf.get_variable(
            "softmax_w", [nhid, ntoken], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [ntoken], dtype=tf.float32)
        logits = tf.einsum('ijk,kl->ijl', outputs, softmax_w) + softmax_b

        cost = tf.contrib.seq2seq.sequence_loss(
            logits,
            self._targets,
            tf.ones([batch_size, bptt], dtype=tf.float32),
            average_across_timesteps=False,  # BPTT
            average_across_batch=True)
        self._cost = tf.reduce_sum(cost)
        self._prediction = tf.reshape(
            tf.nn.softmax(logits), [-1, bptt, ntoken])

        if is_training:
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            if clip:
                grads, _ = tf.clip_by_global_norm(grads, clip)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars))

            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)
        else:
            self._train_op = tf.no_op()

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def predictions(self):
        return self._prediction

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
