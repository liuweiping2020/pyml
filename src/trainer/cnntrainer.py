from modeler.cnnmodel import CNNModel
from src.dater.mnist import read_data_sets
from trainer.tftrainer import TFTrainer


class CNNTrainer(TFTrainer):
    def __init__(self):
        self.iter = 100
        self.batch_size = 100
        self.train_keep_prob = 0.8
        pass

    def get_data(self):
        self.data = read_data_sets("data/MNIST_data/", one_hot=True)
        self.batch = self.data.train.next_batch(self.batch_size)

    def get_model(self):
        cnnModel = CNNModel()
        self.accuracy = cnnModel.accuracy
        self.x = cnnModel.x
        self.y = cnnModel.y
        self.keep_prob = cnnModel.keep_prob
        self.train_step = cnnModel.train_step

    def get_feed_data(self):
        self.train_feed = {self.x: self.batch[0], self.y: self.batch[1], self.keep_prob: 1.0}
        self.test_feed = {self.x: self.batch[0], self.y: self.batch[1], self.keep_prob: self.train_keep_prob}
        self.valid_feed = {self.x: self.data.test.images, self.y: self.data.test.labels, self.keep_prob: 1.0}

    def train(self):
        for i in range(self.iter):
            if i % 10 == 0:
                train_accuracy = self.session.run(self.accuracy, feed_dict=self.train_feed)
                print("step %d, training accuracy %g" % (i, train_accuracy))
            self.train_step.run(feed_dict=self.test_feed)
        print("test accuracy %g" % self.session.run(self.accuracy, feed_dict=self.valid_feed))
