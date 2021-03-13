import tensorflow as tf

from modeler.tfmodel import TFModel


class MultiRNNModel(TFModel):
    def __init__(self, is_training=None, config=None, input_=None):
        self._input = input_
        self.is_training = is_training
        self.config = config
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        self.size = config.hidden_size
        self.vocab_size = config.vocab_size

    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=0.0, state_is_tuple=True)

    def build(self):
        attn_cell = self.lstm_cell
        if self.is_training and self.config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    self.lstm_cell(), output_keep_prob=self.config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [self.vocab_size, self.size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input.input_data)

        if self.is_training and self.config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, self.size])
        softmax_w = tf.get_variable(
            "softmax_w", [self.size, self.vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._input.targets, [-1])],
            [tf.ones([self.batch_size * self.num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / self.batch_size
        self._final_state = state

        if not self.is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

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
