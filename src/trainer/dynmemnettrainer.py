# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from modeler.dynamicmemorynetmodel import DynamicMemoryNet
from trainer.tftrainer import TFTrainer


class DynMemNetTrainer(TFTrainer):
    def __init__(self,config):
        self.num_classes = config.num_classes
        self.learning_rate =config.learning_rate
        self.batch_size = config.batch_size
        self.decay_steps = config.decay_steps
        self.decay_rate = config.decay_rate
        self.sequence_length = config.sequence_length
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.is_training = config.is_training
        self.story_length = config.story_length
        self.dropout_keep_prob = config.dropout_keep_prob
        super(DynMemNetTrainer,self).__init__()
        pass

    def get_model(self):
        self.model = DynamicMemoryNet(self.num_classes, self.learning_rate, self.batch_size,
                                      self.decay_steps, self.decay_rate, self.sequence_length,
                                      self.story_length, self.vocab_size, self.embed_size, self.hidden_size,
                                      self.is_training, multi_label_flag=False)

    def get_feed_dict(self):
        self.train_feed = {self.model.query: self.query, self.model.story: self.story_feed,
                           self.model.answer_single: self.answer_single,
                           self.model.dropout_keep_prob: self.dropout_keep_prob}
        pass

    def get_data(self):
        self.story_feed = np.random.randn(self.batch_size, self.story_length, self.sequence_length)
        self.story_feed[self.story_feed > 0] = 1
        self.story_feed[self.story_feed <= 0] = 0
        self.query = np.random.randn(self.batch_size, self.sequence_length)  # [batch_size, sequence_length]
        self.query[self.query > 0] = 1
        self.query[self.query <= 0] = 0
        self.answer_single = np.sum(self.query, axis=1) + np.round(0.1 * np.sum(np.sum(self.story_feed, axis=1),
                                                                                axis=1))
        pass

    def train(self):
        ckpt_dir = '../../model/dmn/'
        saver = tf.train.Saver()
        for i in range(150):
            # [batch_size].e.g. np.array([1, 0, 1, 1, 1, 2, 1, 1])
            loss, acc, predict, _ = self.session.run(
                [self.model.loss_val, self.model.accuracy, self.model.predictions, self.model.train_op],
                feed_dict=self.train_feed)
            print(i, "query:", self.query, "=====================>")
            print(i, "loss:", loss, "acc:", acc, "label:", self.answer_single, "prediction:", predict)
            if i % 30 == 0:
                save_path = ckpt_dir + "model.ckpt"
                saver.save(self.session, save_path, global_step=i * 300)

    pass
if __name__ == '__main__':
    dmn=DynMemNetTrainer()
    dmn.run()

    pass