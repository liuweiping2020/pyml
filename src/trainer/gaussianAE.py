from tensorflow.examples.tutorials.mnist import input_data

from modeler.gaussianAE import GaussianAutoencoderModel
from trainer.tftrainer import TFTrainer
import sklearn.preprocessing as prep
import numpy as np


class GaussianAETrainer(TFTrainer):
    def __init__(self):
        self.training_epochs = 20
        self.batch_size = 128
        self.display_step = 1
        self.training_scale = 0.1
        pass

    def get_data(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.X_train, self.X_test = self.standard_scale(mnist.train.images, mnist.test.images)
        self.n_samples = int(mnist.train.num_examples)
        pass

    def get_model(self):
        gaussianAutoencoderModel = GaussianAutoencoderModel()
        self.cost = gaussianAutoencoderModel.cost
        self.optimizer = gaussianAutoencoderModel.optimizer
        self.scale = gaussianAutoencoderModel.scale
        self.x = gaussianAutoencoderModel.x
        self.hidden = gaussianAutoencoderModel.hidden
        self.weights = gaussianAutoencoderModel.weights
        self.reconstruction = gaussianAutoencoderModel.reconstruction
        pass

    def train(self):

        for epoch in range(self.training_epochs):
            avg_cost = 0.
            total_batch = int(self.n_samples / self.batch_size)
            for i in range(total_batch):
                batch_xs = self.get_random_block_from_data(self.X_train, self.batch_size)
                cost = self.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / self.n_samples * self.batch_size

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Total cost: " + str(self.calc_total_cost(self.X_test)))

        pass

    def standard_scale(self, X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test

    def get_random_block_from_data(self, data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def partial_fit(self, X):
        cost, opt = self.session.run((self.cost, self.optimizer),
                                     feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        return self.session.run(self.cost, feed_dict={self.x: X,
                                                      self.scale: self.training_scale
                                                      })

    def transform(self, X):
        return self.session.run(self.hidden, feed_dict={self.x: X,
                                                        self.scale: self.training_scale
                                                        })

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.session.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.session.run(self.reconstruction, feed_dict={self.x: X,
                                                                self.scale: self.training_scale
                                                                })

    def getWeights(self):
        return self.session.run(self.weights['w1'])

    def getBiases(self):
        return self.session.run(self.weights['b1'])
