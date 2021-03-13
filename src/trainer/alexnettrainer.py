import tensorflow as tf

from modeler.alexnetmodel import AlexNetModeler
from trainer.tftrainer import TFTrainer


class AlexNetTrainer(TFTrainer):
    def __init__(self):
        self.image_size = 224
        self.batch_size = 64
        pass

    def get_data(self):
        pass

    def get_model(self):
        alexNetModeler = AlexNetModeler()
        self.model_out = alexNetModeler.model_out
        self.parameters=alexNetModeler.parameter
        pass

    def train(self):
        self.session.run(self.model_out)
        # Add a simple objective so we can calculate the backward pass.
        objective = tf.nn.l2_loss(self.model_out)
        # Compute the gradient with respect to all the parameters.
        grad = tf.gradients(objective, self.parameters)
        # Run the backward benchmark.
        self.session.run(grad)
