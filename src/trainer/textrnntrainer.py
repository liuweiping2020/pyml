import numpy as np

from modeler.textrnn import TextRNNModel
from trainer.tftrainer import TFTrainer


class TextRNNTrainer(TFTrainer):
    def __init__(self):
        self.num_classes = 10
        self.learning_rate = 0.01
        self.batch_size = 8
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 5
        self.vocab_size = 10000
        self.embed_size = 100
        self.is_training = True
        self.dropout_keep_prob = 1  # 0.5
        super(TextRNNTrainer, self).__init__()
        pass

    def get_model(self):
        self.textRNN = TextRNNModel(self.num_classes, self.learning_rate, self.batch_size,
                                    self.decay_steps, self.decay_rate, self.sequence_length,
                                    self.vocab_size, self.embed_size, self.is_training)

    def get_feed_dict(self):
        self.feed_dict = {self.textRNN.input_x: self.input_x, self.textRNN.input_y: self.input_y,
                          self.textRNN.dropout_keep_prob: self.dropout_keep_prob}
        pass

    def get_data(self):
        self.input_x = np.zeros((self.batch_size, self.sequence_length))  # [None, self.sequence_length]
        self.input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1])

    def get_fetch_list(self):
        self.fetch_list = [self.textRNN.loss_val, self.textRNN.accuracy,
                           self.textRNN.predictions, self.textRNN.train_op]

    def train(self):
        for i in range(100):
            loss, acc, predict, _ = self.session.run(self.fetch_list, feed_dict=self.feed_dict)
            print("loss:", loss, "acc:", acc, "label:", self.input_y, "prediction:", predict)

if __name__ == '__main__':
    TextRNNTrainer=TextRNNTrainer()
    TextRNNTrainer.run()
    pass
