import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.Linear(1024, 10)