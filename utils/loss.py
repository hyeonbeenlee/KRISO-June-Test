import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_components = 2
        self.eps = nn.Parameter(
            torch.full(
                [self.num_components], 0, dtype=torch.float32, device=self.device
            )
        )

    def forward(self, label, prediction):
        w = torch.sigmoid(self.eps)
        # Weighted MSE
        loss = torch.mean(torch.square(label - prediction)) * w[0]
        # Weighted corrcoef
        loss += -torch.corrcoef(torch.cat([label.T, prediction.T], axis=0))[0, 1] * w[1]
        return loss
