class BaseAgent:
    def __init__(self):
        pass

    def learn(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError
