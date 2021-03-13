from tensorflow.examples.tutorials.mnist import input_data

from modeler.birnnmodel import BiRNNModel
from trainer.tftrainer import TFTrainer


class BiRNNTrainer(TFTrainer):
    def __init__(self):
        self.max_samples = 400000
        self.learning_rate = 0.01
        self.batch_size = 128
        self.display_step = 10
        self.n_input = 28  # MNIST data input (img shape: 28*28)
        self.n_steps = 28  # timesteps
        self.n_hidden = 256  # hidden layer num of features
        self.n_classes = 10  # MNIST total classes (0-9 digits)
        self.test_len = 10000
        pass

    def get_data(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        batch_x, self.batch_y = mnist.train.next_batch(self.batch_size)
        self.batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
        self.test_data = mnist.test.images[:self.test_len].reshape((-1, self.n_steps, self.n_input))
        self.test_label = mnist.test.labels[:self.test_len]

    def get_model(self):
        biRNNModel = BiRNNModel()
        self.x = biRNNModel.x
        self.y = biRNNModel.y
        self.optimizer = biRNNModel.optimize
        self.accuracy = biRNNModel.accuracy
        self.cost = biRNNModel.cost

    def get_feed_data(self):
        self.train_feed = {self.x: self.batch_x, self.y: self.batch_y}
        self.test_feed = {self.x: self.test_data, self.y: self.test_label}
        pass

    def train(self):
        step = 1
        # Keep training until reach max iterations
        while step * self.batch_size < self.max_samples:
            # Reshape data to get 28 seq of 28 elements
            self.session.run(self.optimizer, feed_dict=self.train_feed)
            if step % self.display_step == 0:
                # Calculate batch accuracy
                acc, loss = self.session.run([self.accuracy, self.cost], feed_dict=self.train_feed)
                # Calculate batch loss
                print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        print("Testing Accuracy:", self.session.run(self.accuracy, feed_dict=self.test_feed))

    pass
