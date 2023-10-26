import torch


class GaussianScaler:
    def __init__(self, data: torch.Tensor = torch.randn(2, 1), dim: int = 0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.params = {"mean": data.mean(dim=dim), "std": data.std(dim=dim)}

    def to(self, device: str):
        for k, v in self.params.items():
            self.params[k] = v.to(device)
        return self

    def scale(self, x):
        return (x - self.params["mean"]) / self.params["std"]

    def unscale(self, x):
        return x * self.params["std"] + self.params["mean"]


class MinmaxScaler:
    def __init__(self, data: torch.Tensor = torch.randn(2, 1), dim: int = 0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.params = {"min": data.min(dim=dim).values, "max": data.max(dim=dim).values}

    def to(self, device: str):
        for k, v in self.params.items():
            self.params[k] = v.to(device)
        return self

    def load_params(self, params: dict):
        self.params = params

    def scale(self, x):
        return (x - self.params["min"]) / (self.params["max"] - self.params["min"])

    def unscale(self, x):
        return x * (self.params["max"] - self.params["min"]) + self.params["min"]
