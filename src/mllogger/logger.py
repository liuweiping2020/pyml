import json
import logging.config
import os


class Logger(object):
    def __init__(self):
        self.setup_logging()

    def getLogger(self, module_name):
        return self.logger.getLogger(module_name)

    def setup_logging(self, default_path="logging.json", default_level=logging.INFO, env_key="LOG_CFG"):
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, "r") as f:
                config = json.load(f)
                logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
        self.logger = logging


if __name__ == "__main__":
    pass
