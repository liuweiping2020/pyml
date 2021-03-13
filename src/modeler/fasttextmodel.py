import tensorflow as tf

from modeler.tfmodel import TFModel


class FastTextModel(TFModel):
    def __init__(self, label_size, learning_rate, batch_size, decay_steps, decay_rate, num_sampled, sentence_len,
                 vocab_size, embed_size, is_training):
        """init all hyperparameter here"""
        # set hyperparamter
        self.label_size = label_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        super(FastTextModel, self).__init__()


    def add_placeholder(self):
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")  # X
        self.labels = tf.placeholder(tf.int32, [None], name="Labels")  # y

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.instantiate_weights()

    def build(self):
        self.logits = self.inference()  # [None, self.label_size]

    def cal_loss(self):
        self.loss_val = self.loss()

    def recon_optimize(self):
        self.train_op = self.train()

    def cal_predict(self):
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]

    def cal_accuracy(self):
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
        # tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.get emebedding of words in the sentence
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence)
        # [None,self.sentence_len,self.embed_size]

        # 2.average vectors, to get representation of the sentence
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)
        # [None,self.embed_size]

        # 3.linear classifier layer
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b
        # [None, self.label_size]==tf.matmul([None,self.embed_size],[self.embed_size,self.label_size])
        return logits

    def loss(self, l2_lambda=0.01):  # 0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        if self.is_training:  # training
            labels = tf.reshape(self.labels, [-1])  # [batch_size,1]------>[batch_size,]
            labels = tf.expand_dims(labels, 1)  # [batch_size,]----->[batch_size,1]
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W),
                               biases=self.b,
                               labels=labels,
                               inputs=self.sentence_embeddings,
                               num_sampled=self.num_sampled,  # scalar. 100
                               num_classes=self.label_size, partition_strategy="div"))  # scalar. 1999
        else:
            labels_one_hot = tf.one_hot(self.labels, self.label_size)  # [batch_size]---->[batch_size,label_size]
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
            # labels:[batch_size,label_size];logits:[batch, label_size]
            print("loss0:", loss)  # shape=(?, 1999)
            loss = tf.reduce_sum(loss, axis=1)
            print("loss1:", loss)  # shape=(?,)
        l2_losses = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        return loss + l2_losses

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam")
        return train_op




