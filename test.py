import torch
import torch.nn as nn
import torch.nn.functional as F
from userkit.interface import UserInterface


def test_nn():
    datatypes = ["test"]
    freqs = [100]
    outs = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    for f in freqs:
        for o in outs:
            for dtype in datatypes:
                nn_ui = UserInterface(f"models/{f}hz_{o}.pt", use_gpu=True)
                # nn_ui.show_history()
                nn_ui.show_prediction(
                    dataset_type=dtype, show_ani_input=False, show_ani_output=False
                )
    

if __name__ == "__main__":
    test_nn()
