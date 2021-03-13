import numpy as np

from modeler.textcnn import TextCNNModel
from trainer.tftrainer import TFTrainer


class TextCNNTrainer(TFTrainer):
    def __init__(self):
        self.num_classes = 5
        self.learning_rate = 0.001
        self.batch_size = 8
        self.decay_steps = 1000
        self.decay_rate = 0.95
        self.sequence_length = 5
        self.vocab_size = 10000
        self.embed_size = 100
        self.is_training = True
        self.dropout_keep_prob = 1.0  # 0.5
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 128
        self.multi_label_flag = True
        self.iter=0
        pass

    def get_model(self):
        self.textRNN = TextCNNModel(self.filter_sizes, self.num_filters, self.num_classes,
                                    self.learning_rate, self.batch_size, self.decay_steps,
                                    self.decay_rate, self.sequence_length, self.vocab_size, self.embed_size,
                                    self.is_training, multi_label_flag=self.multi_label_flag)

    def get_data(self):
        input_x = np.random.randn(self.batch_size, self.sequence_length)  # [None, self.sequence_length]
        input_x[input_x >= 0] = 1
        input_x[input_x < 0] = 0
        self.input_x = input_x
        self.input_y_multilabel = self.get_label_y(input_x)

    def get_fetch_list(self):
        self.fetch_list = [self.textRNN.loss_val, self.textRNN.possibility,
                           self.textRNN.W_projection, self.textRNN.train_op]

    def get_feed_dict(self):

        self.feed_dict = {self.textRNN.input_x: self.input_x, self.textRNN.input_y_multilabel: self.input_y_multilabel,
                          self.textRNN.dropout_keep_prob: self.dropout_keep_prob, self.textRNN.iter: self.iter,
                          self.textRNN.tst: False}

    def train(self):
        for i in range(500):
            self.iter = i
            loss, possibility, W_projection_value, _ = self.session.run(self.fetch_list, feed_dict=self.feed_dict)
            print(i, "loss:", loss, "-------------------------------------------------------")
            print("label:", self.input_y_multilabel)
            print("possibility:", possibility)

    def get_label_y(self, input_x):
        length = input_x.shape[0]
        input_y = np.zeros((input_x.shape))
        for i in range(length):
            element = input_x[i, :]  # [5,]
            result = self.compute_single_label(element)
            input_y[i, :] = result
        return input_y

    def compute_single_label(self, listt):
        result = []
        length = len(listt)
        for i, e in enumerate(listt):
            previous = listt[i - 1] if i > 0 else 0
            current = listt[i]
            next = listt[i + 1] if i < length - 1 else 0
            summ = previous + current + next
            if summ >= 2:
                summ = 1
            else:
                summ = 0
            result.append(summ)
        return result


if __name__ == '__main__':
    TextCNNTrainer = TextCNNTrainer()
    TextCNNTrainer.run()

    pass
