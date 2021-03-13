# -*- coding: utf-8 -*-
from dater.dynmemnet import DynMemNetDataSet

from modeler.dynamicmemorynetmodel import DynamicMemoryNet
from predicter.tfpredicter import TFPredicter


class DynMemNetPredicter(TFPredicter):
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
        self.is_training = False
        self.story_length = 3
        self.dropout_keep_prob = 1
        self.epoches = 10
        self.iters = 100
        super(DynMemNetPredicter, self).__init__()

        pass

    def get_ckpt_dir(self, ckpt_dir='../../model/dmn/'):
        self.ckpt_dir = ckpt_dir

    def get_model(self):
        self.model = DynamicMemoryNet(self.num_classes, self.learning_rate,
                                      self.batch_size, self.decay_steps,
                                      self.decay_rate, self.sequence_length,
                                      self.story_length, self.vocab_size,
                                      self.embed_size, self.hidden_size,
                                      self.is_training, multi_label_flag=False)

        pass

    def predict(self):
        for i in range(self.iters):
            self.run_batch(self.epoches)

    def run_batch(self, epoches):
        for i in range(epoches):
            self.get_feed_dict()
            predict = self.session.run([self.model.predictions], feed_dict=self.predict_feed)
            print(i, "query:", self.query, "=====================>")
            print(i, "label:", self.answer_single, "prediction:", predict)

    pass

    def get_data(self):
        dynMemNetDataSet = DynMemNetDataSet()
        dynMemNetDataSet.run_predict()
        self.story = dynMemNetDataSet.story
        self.query = dynMemNetDataSet.query
        self.answer_single = dynMemNetDataSet.answer_single
        pass

    def get_feed_dict(self):
        self.predict_feed = {self.model.query: self.query, self.model.story: self.story,
                             self.model.dropout_keep_prob: self.dropout_keep_prob}
        pass


if __name__ == '__main__':
    dynMemNetPredicter = DynMemNetPredicter()
    dynMemNetPredicter.run()
    pass
