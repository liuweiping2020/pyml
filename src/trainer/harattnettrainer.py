import numpy as np

from trainer.harattnetmodel import HarAttNetMideler
from trainer.tftrainer import TFTrainer


class HarAttNetTrainer(TFTrainer):
    def __init__(self):
        self.num_classes = 3
        self.learning_rate = 0.01
        self.batch_size = 8
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 30
        self.num_sentences = 6  # number of sentences
        self.vocab_size = 10000
        self.embed_size = 100  # 100
        self.hidden_size = 100
        self.is_training = True
        self.dropout_keep_prob = 1  # 0.5 #num_sentences
        super(HarAttNetTrainer,self).__init__()
        pass

    def get_data(self):
        self.input_x = np.zeros((self.batch_size, self.sequence_length))  # num_sentences
        self.input_x[self.input_x > 0.5] = 1
        self.input_x[self.input_x <= 0.5] = 0
        self.input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1])
        # np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
        pass

    def get_model(self):
        self.model = HarAttNetMideler(self.num_classes, self.learning_rate, self.batch_size,
                                      self.decay_steps, self.decay_rate,
                                      self.sequence_length,
                                      self.num_sentences, self.vocab_size, self.embed_size,
                                      self.hidden_size, self.is_training, multi_label_flag=False)
        pass

    def get_fetch_list(self):
        self.fetch_list = [self.model.loss_val, self.model.accuracy, self.model.predictions, self.model.W_projection,
                           self.model.train_op]

        pass

    def get_feed_dict(self):
        self.feed_dict = {self.model.input_x: self.input_x,
                          self.model.input_y: self.input_y,
                          self.model.dropout_keep_prob: self.dropout_keep_prob}

        pass

    def train(self):
        for i in range(100):
            loss, acc, predict, W_projection_value, _ = self.session.run(self.fetch_list, feed_dict=self.feed_dict)
            print("loss:", loss, "acc:", acc, "label:", self.input_y, "prediction:", predict)





if __name__ == '__main__':
    harAttNetTrainer=HarAttNetTrainer()
    harAttNetTrainer.run()

    pass