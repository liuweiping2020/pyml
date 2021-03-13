import tensorflow as tf

from modeler.inceptionnetmodel import InceptionNetModel
from trainer.tftrainer import TFTrainer


class InceptionNetTrainer(TFTrainer):
    def __init__(self):
        self.batch_size = 32
        self.height, self.width = 299, 299
        self.num_batches = 100
        self.slim = tf.contrib.slim
        pass

    def get_data(self):
        inputs = tf.random_uniform((self.batch_size, self.height, self.width, 3))
        pass

    def get_model(self):
        inceptionNetModel = InceptionNetModel()
        self.logits = inceptionNetModel.logits
        pass

    def train(self):
        for i in range(self.num_batches):
            self.session.run(self.logits)
        pass
