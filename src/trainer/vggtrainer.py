import tensorflow as tf

from modeler.vggmodel import VGGModel
from trainer.tftrainer import TFTrainer


class VGGTrainer(TFTrainer):
    def __init__(self):
        self.batch_size = 32
        self.num_batches = 100
        self.image_size = 224
        pass

    def get_model(self):
        vggModel = VGGModel()
        self.predictions = vggModel.predictions
        self.fc8 = vggModel.fc8
        self.p = vggModel.fc8
        self.keep_prob = vggModel.keep_prob

    def get_data(self):
        pass

    def get_feed_data(self):
        pass

    def train(self):
        feed_train = {self.keep_prob: 0.5}
        feed_test = {self.keep_prob: 1.0}
        for i in range(self.num_batches):
            self.session.run(self.predictions, feed_dict=feed_test)
        objective = tf.nn.l2_loss(self.fc8)
        grad = tf.gradients(objective, self.p)
        self.session.run(grad, feed_dict=feed_train)
        self.session.close()
