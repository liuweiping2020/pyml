import os
import sys

import tensorflow as tf

from mllogger.logger import Logger

sys.path.append("../")
from configer.dmnconfig import DMNConfig
from trainer.dynmemnettrainer import DynMemNetTrainer

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 15, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("sequence_length", 10, "max sentence length")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "embedding size")
tf.app.flags.DEFINE_integer("embed_size", 100, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("hidden_size", 100, "hidden size")
tf.app.flags.DEFINE_integer("story_length", 3, "story length")
tf.app.flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout_keep_prob")


def main(_):
    os.environ["LOG_CFG"] = "/Volumes/D/03project/python/tf/src/mllogger/logging.json"
    logger = Logger()
    tflogger = logger.getLogger("dmn")
    tflogger.info("start dmn model ....")
    algos_config = DMNConfig(FLAGS)
    tflogger.info(algos_config.__dict__)
    dmn = DynMemNetTrainer(algos_config)
    dmn.run()
    tflogger.info("end dmn model ....")
    pass


if __name__ == '__main__':
    tf.app.run(main)
