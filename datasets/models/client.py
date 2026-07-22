import flwr as fl
import numpy as np

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        print("Training Client...")
        return parameters, 100, {}

    def evaluate(self, parameters, config):
        print("Evaluating Client...")
        return 0.1, 100, {"accuracy": 0.95}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)
