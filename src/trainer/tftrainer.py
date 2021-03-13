import tensorflow as tf


class TFTrainer(object):
    def __init__(self):
        pass

    def run(self):
        self.get_data()
        self.get_model()
        self.get_fetch_list()
        self.get_feed_dict()
        self.before()
        self.train()
        self.after()

    def train(self):
        pass

    def before(self):
        self.init_session()
        pass

    def after(self):
        self.session.close()
        pass

    def init_session(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        self.session = tf.Session(config=config)
        self.session.run(init)
        self.saver=tf.train.Saver()

    def get_model(self):
        pass

    def get_data(self):
        pass

    def get_feed_dict(self):
        pass

    def get_fetch_list(self):
        pass
