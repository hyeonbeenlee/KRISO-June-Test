import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import numpy as np
from functools import partial
from scipy.stats import spearmanr
from architectures.convlstm import ConvLSTM
from utils.snippets import *
from utils.metrics import *
from utils.trainer import Trainer
from .misc import *


class UserInterface(Trainer):
    def __init__(self, modelpath, use_gpu: bool = True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.modelpath = modelpath
        self.load_model(modelpath)
        self.show_props()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = self.scaler_i.scale(x)
        out = self.model(out)
        out = self.scaler_o.unscale(out)
        return out

    def load_model(self, modelpath: str):
        if self.device == "cuda":
            result = torch.load(modelpath)
        elif self.device == "cpu":
            result = torch.load(modelpath, map_location="cpu")
        self.metric_history = result["metric_history"]
        self.model_info = result["model_info"]  # hyperparameters
        self.model_init_args = result["model_init_args"]
        self.scaler_i = result["scaler_i"].to(self.device)
        self.scaler_o = result["scaler_o"].to(self.device)
        if "__class__" in self.model_init_args.keys():
            del self.model_init_args["__class__"]
        self.model = ConvLSTM(**self.model_init_args).to(self.device)
        self.model.load_state_dict(result["state_dict"])

    def show_props(self):
        print_line()
        print(f"{self.model_info['__class__']} instance from {self.modelpath}")
        print(f"Sampling rate: {self.model_info['sampling_rate']} Hz")
        print(
            f"Input shape: (N,C={len(self.model_info['input_columns'])},L={self.model_info['input_timesteps']})"
        )
        print(f"Input: {self.model_info['input_columns']}")
        print(f"Output: {self.model_info['output_columns']}")
        print_line()
        print()

    def show_history(self):
        plot_template(12)
        keys = list(self.metric_history.keys())
        n_rows = len(keys) // 2
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9))
        index = 0
        labels = ["train", "valid"]
        colors = ["black", "blue"]
        for i in range(n_rows):
            for j in range(n_cols):
                text = ""
                for k in range(2):
                    history = self.metric_history[keys[index]][:, k]
                    epochs = np.arange(len(history)) + 1
                    argmin = np.argmin(history)
                    min = np.min(history)
                    text += f"{labels[k]} min: {min:.4f} at epoch {argmin + 1}\n"
                    axes[i, j].plot(
                        epochs,
                        history,
                        color=colors[k],
                        label=labels[k],
                        marker="o",
                        markerfacecolor="none",
                    )
                    axes[i, j].set_ylabel(keys[index])
                text = text[:-1]
                add_textbox(axes[i, j], text, loc=1, fontsize=10)
                index += 1
        axes[0, 0].legend(loc=3)
        axes[-1, 0].set_xlabel("Epochs")
        axes[-1, 1].set_xlabel("Epochs")
        fig.tight_layout()
        plt.show()

    def build_dataset(self):
        assert "modelpath" in self.__dict__.keys(), "__init__ must be called first."
        self.load_data_Feb2023(
            self.model_info["input_columns"],
            self.model_info["output_columns"],
            self.model_info["input_timesteps"],
            self.model_info["sampling_rate"],
        )
        self.setup_dataloader(self.model_info["batch_size"], shuffle_train=False)

    def show_prediction(
        self,
        dataset_type: str = "test",
        show_ani_input: bool = False,
        show_ani_output: bool = False,
    ):
        assert dataset_type in [
            "train",
            "valid",
            "test",
        ], "dataset_type must be one of ['train', 'valid', 'test']"
        self.build_dataset()
        with torch.no_grad():
            self.model.eval()
            self.Test_O = []
            self.Test_Pred = []
            for batch in self.dataloaders[dataset_type]:
                batch_test_i, batch_test_o = batch
                batch_pred = self.predict(batch_test_i)
                self.Test_Pred.append(batch_pred.cpu())
                self.Test_O.append(batch_test_o.cpu())
            self.Test_O = torch.cat(self.Test_O, dim=0)
            self.Test_Pred = torch.cat(self.Test_Pred, dim=0)
        # Plot full-time result
        plot_template(14)
        fig1, axes1 = plt.subplots(figsize=(9, 4))
        style_label = dict(color="black", linewidth=1, linestyle="solid", label="Label")
        style_pred = dict(
            color="red", linewidth=0.7, linestyle=(0, (5, 10)), label="Prediction"
        )
        axes1.plot(self.Test_O, **style_label)
        axes1.plot(self.Test_Pred, **style_pred)
        axes1.set_xlabel("Timestamps")
        axes1.set_title(self.model_info["output_columns"][0])
        fig1.tight_layout()
        score = f"spearmanr: {spearmanr(self.Test_O, self.Test_Pred).correlation:.5f}\n"
        score += f"corrcoef: {crosscorr(self.Test_O, self.Test_Pred):.5f}\n"
        score += f"rmserr: {rmserr(self.Test_O, self.Test_Pred):.2f}%\n"
        score += f"peaktopeak: {peaktopeakerr(self.Test_O, self.Test_Pred):.2f}%\n"
        score += f"rel_meanerr: {rel_meanerr(self.Test_O, self.Test_Pred):.2f}%"
        add_textbox(axes1, score, loc=1, fontsize=10)

        # Animation variables
        self.__fstep = 10
        self.__playspeed = 3  # times faster than real-time
        self.__fps = round(
            self.model_info["sampling_rate"] / self.__fstep * self.__playspeed
        )
        self.__interval = 1 / self.__fps * 1000
        self.__n_cols = 3
        self.__n_rows = 5  # fixed n_rows
        # self._n_rows = len(self.model_info['input_columns']) // self._n_cols # adaptive n_rows
        self.__writer = animation.FFMpegWriter(fps=self.__fps)
        if show_ani_input:
            self.__fig2, self.__axes2 = plt.subplots(
                self.__n_rows, self.__n_cols, figsize=(10, 9)
            )
            self.__lines2 = []
            index_c = 0
            for i in range(self.__n_rows):
                for j in range(self.__n_cols):
                    try:
                        self.__lines2.append(
                            self.__axes2[i, j].plot(
                                [], [], marker="o", markerfacecolor="none", color="blue"
                            )[0]
                        )
                        self.__axes2[i, j].set_xlim(
                            0, self.model_info["input_timesteps"] - 1
                        )
                        self.__axes2[i, j].set_ylim(
                            self.data_tensors[f"{dataset_type}data_in"][:, index_c, :]
                            .flatten()
                            .unique()[1],
                            # second minimum
                            self.data_tensors[f"{dataset_type}data_in"][:, index_c, :]
                            .flatten()
                            .unique()[-2],
                        )  # second maximum
                        self.__axes2[i, j].set_title(
                            f"$C_{{{index_c + 1}}}$: {self.model_info['input_columns'][index_c]}"
                        )
                    except IndexError:
                        pass
                    index_c += 1
            self.__fig2.tight_layout()
            ani_i = FuncAnimation(
                self.__fig2,
                partial(self.__animate_i, dataset_type=dataset_type),
                frames=self.data_tensors[f"{dataset_type}data_in"].shape[0]
                // self.__fstep,
                interval=self.__interval,
                blit=True,
            )
        if show_ani_output:
            self.__fig3, self.__axes3 = plt.subplots(figsize=(9, 4))
            self.__lines3 = []
            styles = [style_label, style_pred]
            for n in range(2):  # label, pred
                self.__lines3.append(self.__axes3.plot(self.Test_O, **styles[n])[0])
            self.__axes3.set_xlabel("Timestamps")
            self.__axes3.set_title(self.model_info["output_columns"][0])
            self.__fig3.tight_layout()
            ani_o = FuncAnimation(
                self.__fig3,
                partial(self.__animate_o, dataset_type=dataset_type),
                frames=self.data_tensors[f"{dataset_type}data_in"].shape[0]
                // self.__fstep,
                interval=self.__interval,
                blit=True,
            )
        plt.show()
        # if show_ani_input:
        #     ani_i_path = f"figures/{self.model_info['sampling_rate']}hz_{self.model_info['output_columns'][0]}_{dataset_type}_i.mp4"
        #     start = time.perf_counter()
        #     ani_i.save(ani_i_path, writer=self.__writer)
        #     end = time.perf_counter()
        #     print_time(start, end)
        #     print(f"Animation saved: {ani_i_path}")
        # if show_ani_output:
        #     ani_o_path = f"figures/{self.model_info['sampling_rate']}hz_{self.model_info['output_columns'][0]}_{dataset_type}_o.mp4"
        #     start = time.perf_counter()
        #     ani_o.save(ani_o_path, writer=self.__writer)
        #     end = time.perf_counter()
        #     print_time(start, end)
        #     print(f"Animation saved: {ani_o_path}")

    def __animate_i(self, i, dataset_type: str = "test"):
        key = f"{dataset_type}data_in"
        xdata = np.arange(self.model_info["input_timesteps"])
        xticks = np.linspace(
            0, self.model_info["input_timesteps"], 6, endpoint=True, dtype=np.int8
        )
        xtickslabels = (
            np.linspace(
                0, self.model_info["input_timesteps"], 6, endpoint=True, dtype=np.int8
            )
            + i * self.__fstep
        )
        for lnum, line in enumerate(self.__lines2):
            try:
                line.set_data(xdata, self.data_tensors[key][i * self.__fstep, lnum, :])
            except IndexError:
                pass
        return self.__lines2

    def __animate_o(self, i, dataset_type: str = "test"):
        xdata_label = range(self.Test_O.shape[0])
        xdata_pred = range(self.Test_Pred[: self.__fstep * i].shape[0])
        ydata_label = self.Test_O
        ydata_pred = self.Test_Pred[: self.__fstep * i]
        xlist = [xdata_label, xdata_pred]
        ylist = [ydata_label, ydata_pred]
        for lnum, line in enumerate(self.__lines3):
            line.set_data(xlist[lnum], ylist[lnum])
        return self.__lines3
