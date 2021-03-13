from modeler.entitynetmodel import EntityNetModel
from trainer.tftrainer import TFTrainer
import tensorflow as tf
import numpy as np


class EntityNetTrainer(TFTrainer):
    def __init__(self):
        self.num_classes = 15
        self.learning_rate = 0.001
        self.batch_size = 8
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 10
        self.vocab_size = 10000
        self.embed_size = 100
        self.hidden_size = 100
        self.is_training = True
        self.story_length = 3
        self.dropout_keep_prob = 1
        self.use_bi_lstm = False
        self.ckpt_dir = '../../model/entitynet/'
        super(EntityNetTrainer,self).__init__()
        pass

    def get_model(self):
        self.model = EntityNetModel(self.num_classes, self.learning_rate, self.batch_size, self.decay_steps,
                                    self.decay_rate, self.sequence_length,
                                    self.story_length, self.vocab_size, self.embed_size,
                                    self.hidden_size, self.is_training,
                                    multi_label_flag=False, block_size=20, use_bi_lstm=self.use_bi_lstm)
        pass

    def get_data(self):
        self.story = np.random.randn(self.batch_size, self.story_length, self.sequence_length)
        self.story[self.story > 0] = 1
        self.story[self.story <= 0] = 0
        self.query = np.random.randn(self.batch_size, self.sequence_length)  # [batch_size, sequence_length]
        self.query[self.query > 0] = 1
        self.query[self.query <= 0] = 0
        self.answer_single = np.sum(self.query, axis=1) + np.round(0.1 * np.sum(np.sum(self.story, axis=1), axis=1))
        pass

    def get_feed_dict(self):
        self.feed_dict = {self.model.query: self.query, self.model.story: self.story,
                          self.model.answer_single: self.answer_single,
                          self.model.dropout_keep_prob: self.dropout_keep_prob}
        pass

    def get_fetch_list(self):
        self.feed_list = [self.model.loss_val, self.model.accuracy, self.model.predictions, self.model.train_op]
        pass

    def train(self):
        saver = tf.train.Saver()
        for i in range(1500):
            loss, acc, predict, _ = self.session.run(self.feed_list, feed_dict=self.feed_dict)
            print(i, "query:", self.query, "=====================>")
            print(i, "loss:", loss, "acc:", acc, "label:", self.answer_single, "prediction:", predict)
            if i % 300 == 0:
                save_path = self.ckpt_dir + "model.ckpt"
                saver.save(self.session, save_path, global_step=i * 300)
    pass


if __name__ == '__main__':
    entityNetTrainer=EntityNetTrainer()
    entityNetTrainer.run()

    pass

