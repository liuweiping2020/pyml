from tensorflow.examples.tutorials.mnist import input_data

from modeler.mlpmodel import MLPModel
from trainer.tftrainer import TFTrainer


class MLPTrainer(TFTrainer):
    def __init__(self):
        pass

    def get_data(self):
        self.mnist = input_data.read_data_sets("/Volumes/D/lwp/project/python/tf/src/data/MNIST_data/",
                                               one_hot=True)
        self.batch_xs, self.batch_ys = self.mnist.train.next_batch(100)
        pass

    def get_model(self):
        mlpModel = MLPModel()
        self.train_step = mlpModel.train_step
        self.accuracy = mlpModel.accuracy
        self.x = mlpModel.x
        self.y_ = mlpModel.y_
        self.keep_prob = mlpModel.keep_prob

    def get_feed_data(self):
        self.train_feed = {self.x: self.batch_xs, self.y_: self.batch_ys, self.keep_prob: 0.75}
        self.test_feed = {self.x: self.mnist.test.images, self.y_: self.mnist.test.labels, self.keep_prob: 1.0}

        pass

    def train(self):
        for i in range(3000):
            self.session.run(self.train_step, feed_dict=self.train_feed)

        print(self.session.run(self.accuracy, feed_dict=self.test_feed))
        pass
