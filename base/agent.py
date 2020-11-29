class Agent:
    def __init__(self, config):
        self.config = config.copy()

    def load_checkpoint(self, filename):
        raise NotImplementedError

    def save_checkpoint(self, filename=None):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError