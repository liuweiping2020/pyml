class DMNConfig(object):
    def __init__(self, FLAGS=None, cfg_dir=None, cfg_obj=None):
        if FLAGS is not None:
            self.num_classes = FLAGS.num_classes
            self.learning_rate = FLAGS.learning_rate
            self.batch_size = FLAGS.batch_size
            self.decay_steps = FLAGS.decay_steps
            self.decay_rate = FLAGS.decay_rate
            self.sequence_length = FLAGS.sequence_length
            self.vocab_size = FLAGS.vocab_size
            self.embed_size = FLAGS.embed_size
            self.hidden_size = FLAGS.hidden_size
            self.is_training = FLAGS.is_training
            self.story_length = FLAGS.story_length
            self.dropout_keep_prob = FLAGS.dropout_keep_prob
        else:
            self.num_classes = 15
            self.learning_rate = 0.0005
            self.batch_size = 8
            self.decay_steps = 1000
            self.decay_rate = 0.9
            self.sequence_length = 10
            self.vocab_size = 10000
            self.embed_size = 100
            self.hidden_size = 100
            self.is_training = True
            self.story_length = 3
            self.dropout_keep_prob = 1.0

