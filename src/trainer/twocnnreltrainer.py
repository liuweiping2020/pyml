import random

import numpy as np
from tflearn.data_utils import pad_sequences

from modeler.twocnnrelmodel import TwoCNNRelModel
from trainer.tftrainer import TFTrainer


class TwoCNNRelTrainer(TFTrainer):
    def __init__(self):
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 128
        self.num_classes = 2
        self.learning_rate = 0.001
        self.batch_size = 1
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 15
        self.vocab_size = 100
        self.embed_size = 100
        self.is_training = True
        self.dropout_keep_prob = 1
        super(TwoCNNRelTrainer, self).__init__()
        pass

    def get_model(self):
        self.twoCNNTR = TwoCNNRelModel(self.filter_sizes, self.num_filters, self.num_classes,
                                       self.learning_rate, self.batch_size, self.decay_steps,
                                       self.decay_rate,
                                       self.sequence_length, self.vocab_size, self.embed_size, self.is_training)

        pass

    def get_data(self):
        x1 = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        x2 = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        label = 0
        if np.abs(x1 - x2) < 3: label = 1
        self.input_y = np.array([label])  # np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
        self.x1 = pad_sequences([[x1, x1, x1, x1, x1]], maxlen=self.sequence_length, value=0)
        self.x2 = pad_sequences([[x2, x2, x2, x2, x2]], maxlen=self.sequence_length, value=0)

        pass

    def get_fetch_list(self):
        self.fetch_list = [self.twoCNNTR.loss_val, self.twoCNNTR.accuracy,
                           self.twoCNNTR.predictions, self.twoCNNTR.train_op]

        pass

    def get_feed_dict(self):

        self.feed_dict = {self.twoCNNTR.input_x: self.x1, self.twoCNNTR.input_x2: self.x2,
                          self.twoCNNTR.input_y: self.input_y,
                          self.twoCNNTR.dropout_keep_prob: self.dropout_keep_prob}

        pass

    def train(self):
        for i in range(100):
            loss_sum = 0.0
            acc_sum = 0.0
            loss, acc, predict, _ = self.session.run(self.fetch_list, feed_dict=self.feed_dict)
            loss_sum = loss_sum + loss
            acc_sum = acc_sum + acc
            if i % 10 == 0:
                print(i, "x1:", self.x1, ";x2:", self.x2, "loss:", loss, "acc:", acc, "avg loss:", loss_sum / (i + 1),
                      ";avg acc:", acc_sum / (i + 1), "label:", self.input_y, "prediction:", predict)
if __name__ == '__main__':
    TwoCNNRelTrainer=TwoCNNRelTrainer()
    TwoCNNRelTrainer.run()
    pass