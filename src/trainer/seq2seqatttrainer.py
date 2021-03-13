import copy
import random

import numpy as np

from modeler.seq2seqattmodel import Seq2SeqAttModel
from trainer.tftrainer import TFTrainer


class Seq2SeqAttTrainer(TFTrainer):
    def __init__(self):
        self.num_classes = 9 + 2
        self.learning_rate = 0.0001
        self.batch_size = 1
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 5
        self.vocab_size = 300
        self.embed_size = 100  # 100
        self.hidden_size = 100
        self.is_training = True
        self.dropout_keep_prob = 1  # 0.5 #num_sentences
        self.decoder_sent_length = 6
        self.l2_lambda = 0.0001
        self.ckpt_dir = '../../model/seq2seqatt/'
        super(Seq2SeqAttTrainer,self).__init__()
        pass

    def get_data(self):
        label_list = self.get_unique_labels()
        self.input_x = np.array([label_list], dtype=np.int32)  # [2,3,4,5,6]
        self.label_list_original = copy.deepcopy(label_list)
        label_list.reverse()
        self.decoder_input = np.array([[0] + label_list], dtype=np.int32)  # [[0,2,3,4,5,6]]
        self.input_y_label = np.array([label_list + [1]], dtype=np.int32)  # [[2,3,4,5,6,1]]
        pass

    def get_model(self):
        self.model = Seq2SeqAttModel(self.num_classes, self.learning_rate, self.batch_size,
                                     self.decay_steps, self.decay_rate, self.sequence_length,
                                     self.vocab_size, self.embed_size, self.hidden_size, self.is_training,
                                     decoder_sent_length=self.decoder_sent_length, l2_lambda=self.l2_lambda)
        pass

    def get_fetch_list(self):
        self.fetch_list = [self.model.loss_val, self.model.accuracy, self.model.predictions,
                           self.model.W_projection, self.model.train_op]

        pass

    def get_feed_dict(self):
        self.feed_dict = {self.model.input_x: self.input_x,
                          self.model.decoder_input: self.decoder_input,
                          self.model.input_y_label: self.input_y_label,
                          self.model.dropout_keep_prob: self.dropout_keep_prob}
        pass

    def train(self):
        for i in range(150):
            loss, acc, predict, W_projection_value, _ = self.session.run(self.fetch_list, feed_dict=self.feed_dict)
            print(i, "loss:", loss, "acc:", acc, "label_list_original as input x:", self.label_list_original,
                  ";input_y_label:", self.input_y_label, "prediction:", predict)

            if i % 30 == 0:
                save_path = self.ckpt_dir + "model.ckpt"
                self.saver.save(self.session, save_path, global_step=i * 300)

    def get_unique_labels(self):
        x = [2, 3, 4, 5, 6]
        random.shuffle(x)
        return x

    pass


if __name__ == '__main__':
    seq2SeqAttTrainer = Seq2SeqAttTrainer()
    seq2SeqAttTrainer.run()

    pass
