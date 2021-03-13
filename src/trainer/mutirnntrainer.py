import time

import numpy as np
import tensorflow as tf

from dater import reader
from modeler.multirnnmodel import MultiRNNModel
from trainer.tftrainer import TFTrainer


class MutiRNNTrainer(TFTrainer):
    def __init__(self):
        self.config = SmallConfig()
        self.eval_config = SmallConfig()
        self.eval_config.batch_size = 1
        self.eval_config.num_steps = 1
        pass

    def get_model(self):
        self.multiRNNModel = MultiRNNModel
        pass

    def get_data(self):
        raw_data = reader.ptb_raw_data('data/simple-examples.tar/data/')
        self.train_data, self.valid_data, self.test_data, _ = raw_data
        pass

    def train(self):
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-self.config.init_scale,
                                                        self.config.init_scale)

            with tf.name_scope("Train"):
                train_input = PTBInput(config=self.config, data=self.train_data, name="TrainInput")
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    m = self.multiRNNModel(is_training=True, config=self.config, input_=train_input)

            with tf.name_scope("Valid"):
                valid_input = PTBInput(config=self.config, data=self.valid_data, name="ValidInput")
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mvalid = self.multiRNNModel(is_training=False, config=self.config, input_=valid_input)

            with tf.name_scope("Test"):
                test_input = PTBInput(config=self.eval_config, data=self.test_data, name="TestInput")
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mtest = self.multiRNNModel(is_training=False, config=self.eval_config,
                                               input_=test_input)

            sv = tf.train.Supervisor()
            with sv.managed_session() as session:
                for i in range(self.config.max_max_epoch):
                    lr_decay = self.config.lr_decay ** max(i + 1 - self.config.max_epoch, 0.0)
                    m.assign_lr(session, self.config.learning_rate * lr_decay)

                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity = self.run_epoch(session, m, eval_op=m.train_op,
                                                      verbose=True)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                    valid_perplexity = self.run_epoch(session, mvalid)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

                test_perplexity = self.run_epoch(session, mtest)
                print("Test Perplexity: %.3f" % test_perplexity)

        pass

    def run_epoch(self, session, model, eval_op=None, verbose=False):
        """Runs the model on the given data."""
        costs = 0.0
        iters = 0
        state = session.run(model.initial_state)
        start_time = time.time()
        fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        for step in range(model.input.epoch_size):
            feed_dict = {}
            for i, (c, h) in enumerate(model.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += model.input.num_steps

            if verbose and step % (model.input.epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                       iters * model.input.batch_size / (time.time() - start_time)))

        return np.exp(costs / iters)


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)
