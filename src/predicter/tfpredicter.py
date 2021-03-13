import tensorflow as tf


class TFPredicter(object):
    def __init__(self):
        self.ckpt_dir = None
        pass

    def run(self):
        self.before()
        self.get_model()
        self.get_fetch_list()
        self.get_data()
        self.get_feed_dict()
        self.predict()
        self.after()
        pass

    def predict(self):
        pass

    def get_data(self):
        pass

    def get_fetch_list(self):
        pass

    def get_feed_dict(self):
        pass

    def get_model(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver = tf.train.import_meta_graph(''.join([ckpt.model_checkpoint_path, '.meta']))
            self.saver.restore(self.session, tf.train.latest_checkpoint(self.ckpt_dir))
        pass

    def set_ckpt_dir(self):
        pass

    def init_session(self):
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.session = tf.Session(config=tfconfig)
        # self.sess.run(tf.global_variables_initializer())

    def before(self):
        self.set_ckpt_dir()
        self.init_session()

    def after(self):
        self.session.close()
