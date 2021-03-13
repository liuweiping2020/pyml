import random

import numpy as np

from modeler.bilstmrelmodel import BiLSTMRelModel
from trainer.tftrainer import TFTrainer


class BiLSTMTrainer(TFTrainer):
    def __init__(self):
        self.num_classes = 2
        self.learning_rate = 0.001
        self.batch_size = 1
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 3
        self.vocab_size = 10000
        self.embed_size = 100
        self.is_training = True
        self.dropout_keep_prob = 1  # 0.5
        super(BiLSTMTrainer, self).__init__()
        pass

    def get_model(self):
        self.textRNN = BiLSTMRelModel(self.num_classes, self.learning_rate, self.batch_size,
                                      self.decay_steps, self.decay_rate, self.sequence_length,
                                      self.vocab_size, self.embed_size, self.is_training)
        pass

    def get_fetch_list(self):
        self.fetch_list = [self.textRNN.loss_val, self.textRNN.accuracy,
                           self.textRNN.predictions, self.textRNN.train_op]

    def get_data(self):
        x1 = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        x2 = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.input_x = np.array([[x1, 0, x2]])
        label = 0
        if np.abs(x1 - x2) < 3: label = 1
        self.input_y = np.array([label])  # np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]

    def get_feed_dict(self):

        self.feed_dict = {self.textRNN.input_x: self.input_x, self.textRNN.input_y: self.input_y,
                          self.textRNN.dropout_keep_prob: self.dropout_keep_prob}

    def train(self):
        for i in range(100):
            loss, acc, predict, _ = self.session.run(self.fetch_list, feed_dict=self.feed_dict)
            if i % 10 == 0:
                print(i, "loss:", loss, ";acc:", acc, ";label:", self.input_y, ";prediction:", predict)

    pass
if __name__ == '__main__':
    BiLSTMTrainer=BiLSTMTrainer()
    BiLSTMTrainer.run()

    pass