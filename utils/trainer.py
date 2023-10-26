import torch.nn as nn
import time
import platform
import glob
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from ray import tune
from architectures.convlstm import ConvLSTM
from .scalers import GaussianScaler
from utils.snippets import *
from utils.metrics import *
from utils.loss import CorrLoss


class Trainer:
    def __init__(self, model: ConvLSTM):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

    def load_data_Feb2023(
        self, input_columns, output_columns, input_timesteps, sampling_rate
    ):
        self.model.update_attributes(locals())
        """
        data/Data_Feb2023/csv\Ground_1st_Test_1.csv (22407, 28) # Validation data
        data/Data_Feb2023/csv\Ground_1st_Test_2.csv (12652, 28)
        data/Data_Feb2023/csv\Ground_1st_Test_3.csv (15262, 28)
        data/Data_Feb2023/csv\Ground_1st_Test_4.csv (12077, 28)
        data/Data_Feb2023/csv\Ground_1st_Test_5.csv (14127, 28)
        data/Data_Feb2023/csv\Ground_1st_Test_6.csv (13230, 28)
        data/Data_Feb2023/csv\Ground_2nd_Test_1.csv (33578, 28)
        data/Data_Feb2023/csv\Ground_2nd_Test_2.csv (20084, 28)
        data/Data_Feb2023/csv\Ground_2nd_Test_3.csv (47872, 28) # Test data
        data/Data_Feb2023/csv\Ground_2nd_Test_4.csv (33178, 28)
        data/Data_Feb2023/csv\Ground_2nd_Test_5.csv (44013, 28)
        data/Data_Feb2023/csv\Ground_2nd_Test_6.csv (32937, 28)
        """
        # Data absloute path
        if platform.system() == "Windows":
            datapath = f"D:\GitHub\Grants\KRISO_DNN\data\Data_Feb2023\csv\preprocessed_{sampling_rate}hz"  # My PC
            # datapath = f'C:\\Users\ms-hyeonbeen\Desktop\Codes\KRISO_DNN\data\Data_Feb2023\csv\preprocessed_{sampling_rate}hz'  # WS
        elif platform.system() == "Linux":
            datapath = f"/home/mslab/Github/KRISO_DNN/data/Data_Feb2023/csv/preprocessed_{sampling_rate}hz"
        datapath = f"data/Data_Feb2023/csv/preprocessed_{sampling_rate}hz"
        # Data reading loop
        self.data_tensors = {}
        self.data_tensors["traindata_in"] = []
        self.data_tensors["traindata_out"] = []
        self.data_tensors["validdata_in"] = []
        self.data_tensors["validdata_out"] = []
        self.data_tensors["testdata_in"] = []
        self.data_tensors["testdata_out"] = []
        self.data_tensors["traindata_in"] = []
        self.data_tensors["traindata_in"] = []
        datalist = glob.glob(f"{datapath}/*.csv")
        for d in datalist:
            # Divide in/out
            d_in = df2tensor(pd.read_csv(d)[input_columns])
            d_out = df2tensor(pd.read_csv(d)[output_columns])
            # Zero-pad initial states
            pad = torch.zeros(input_timesteps - 1, d_in.shape[1])
            d_in = torch.cat([pad, d_in], dim=0)
            # Windowing
            windows_in = []
            windows_out = []
            for t in range(d_out.shape[0] - 1):  # except the first step
                windows_in.append(d_in[t : t + input_timesteps])
                windows_out.append(d_out[t + 1])  # (N,C)
            windows_in = torch.stack(windows_in).permute(0, 2, 1)  # (N,L,C)
            windows_out = torch.stack(windows_out)  # (N,C)
            # Divide train/valid/test
            if "Ground_1st_Test_1" in d:
                self.data_tensors["validdata_in"].append(windows_in)
                self.data_tensors["validdata_out"].append(windows_out)
            elif "Ground_2nd_Test_3" in d:
                self.data_tensors["testdata_in"].append(windows_in)
                self.data_tensors["testdata_out"].append(windows_out)
            else:
                self.data_tensors["traindata_in"].append(windows_in)
                self.data_tensors["traindata_out"].append(windows_out)
        self.data_tensors["traindata_in"] = torch.cat(
            self.data_tensors["traindata_in"], dim=0
        )
        self.data_tensors["traindata_out"] = torch.cat(
            self.data_tensors["traindata_out"], dim=0
        )
        self.data_tensors["validdata_in"] = self.data_tensors["validdata_in"][0]
        self.data_tensors["validdata_out"] = self.data_tensors["validdata_out"][0]
        self.data_tensors["testdata_in"] = self.data_tensors["testdata_in"][0]
        self.data_tensors["testdata_out"] = self.data_tensors["testdata_out"][0]
        # print(f"Training dataset: {self.data_tensors['traindata_in'].shape}, {self.data_tensors['traindata_out'].shape}")
        # print(f"Validation dataset: {self.data_tensors['validdata_in'].shape}, {self.data_tensors['validdata_out'].shape}")
        # print(f"Test dataset: {self.data_tensors['testdata_in'].shape}, {self.data_tensors['testdata_out'].shape}")

    def setup_dataloader(self, batch_size, shuffle_train=True):
        self.model.update_attributes(locals())
        self.scaler_i = GaussianScaler(self.data_tensors["traindata_in"]).to(
            self.device
        )
        self.scaler_o = GaussianScaler(self.data_tensors["traindata_out"]).to(
            self.device
        )
        self.dataloaders = {}
        self.dataloaders["train"] = DataLoader(
            TensorDataset(
                self.data_tensors["traindata_in"], self.data_tensors["traindata_out"]
            ),
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        self.dataloaders["valid"] = DataLoader(
            TensorDataset(
                self.data_tensors["validdata_in"], self.data_tensors["validdata_out"]
            ),
            batch_size=1024,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        self.dataloaders["test"] = DataLoader(
            TensorDataset(
                self.data_tensors["testdata_in"], self.data_tensors["testdata_out"]
            ),
            batch_size=1024,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

    def setup_optimizer(
        self, initial_lr: 1e-3, multioptim: bool = False, optim_alg: str = "radam"
    ):
        self.model.update_attributes(locals())
        optim_algs = {
            "radam": torch.optim.RAdam,
            "nadam": torch.optim.NAdam,
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
        }
        self.optimizer_group = []
        if multioptim:
            # RNNs
            self.optimizer_group.append(
                optim_algs[optim_alg](
                    [
                        {"params": self.model.RNN.parameters(), "lr": initial_lr},
                        {
                            "params": self.model.RNN_params.parameters(),
                            "lr": initial_lr,
                        },
                    ]
                )
            )
            # CNNs
            self.optimizer_group.append(
                optim_algs[optim_alg](
                    [{"params": self.model.CNNs.parameters(), "lr": initial_lr}]
                )
            )
            # MLPs
            Linears = [{"params": self.model.linear_in.parameters(), "lr": initial_lr}]
            if hasattr(self.model, "linear_hidden"):
                Linears.append(
                    {"params": self.model.linear_hidden.parameters(), "lr": initial_lr}
                )
            if hasattr(self.model, "linear_out"):
                Linears.append(
                    {"params": self.model.linear_out.parameters(), "lr": initial_lr}
                )
            self.optimizer_group.append(optim_algs[optim_alg](Linears))
            # Loss function parameters
            self.optimizer_group.append(
                optim_algs[optim_alg](
                    [{"params": self.loss_fn.parameters(), "lr": initial_lr}]
                )
            )
        else:
            self.optimizer_group.append(
                optim_algs[optim_alg](
                    [
                        {"params": self.model.parameters(), "lr": initial_lr},
                        {"params": self.loss_fn.parameters(), "lr": initial_lr},
                    ]
                )
            )  # All Layers

    def setup_loss_fn(self, loss_fn: str = "mse"):
        self.model.update_attributes(locals())
        loss_dict = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "smoothl1": nn.SmoothL1Loss(),
            "corrloss": CorrLoss(),
        }
        self.loss_fn = loss_dict[loss_fn]

    def fit(
        self,
        epochs: int,
        lr_halflife: int,
        input_columns: list,
        output_columns: list,
        input_timesteps: int,
        sampling_rate: int,
        reg_lambda: float = 1e-3,
        valid_metric: str = "corrcoef",
        save_filename: str = None,
        report_ray: bool = False,
        batch_size: int = 16,
        initial_lr: float = 1e-3,
        multioptim: bool = False,
        optim_alg: str = "radam",
        loss_fn: str = "mse",
    ):
        self.model.update_attributes(locals())
        self.model.print_model_info()
        self.valid_metric = valid_metric
        self.input_columns = input_columns
        self.output_columns = output_columns

        # Setup
        self.load_data_Feb2023(
            input_timesteps=input_timesteps,
            input_columns=input_columns,
            output_columns=output_columns,
            sampling_rate=sampling_rate,
        )
        self.setup_dataloader(batch_size=batch_size)
        self.setup_loss_fn(loss_fn)
        self.setup_optimizer(
            initial_lr=initial_lr, multioptim=multioptim, optim_alg=optim_alg
        )

        # Init
        model_best_parameter = None
        model_best_metric = torch.inf
        model_best_metric_index = 0
        large_value = np.inf
        self.metric_history = {}
        self.metric_history["spearmanr"] = np.full((epochs, 3), large_value)
        self.metric_history["corrcoef"] = np.full((epochs, 3), large_value)
        self.metric_history["r2"] = np.full((epochs, 3), large_value)
        self.metric_history["rms"] = np.full((epochs, 3), large_value)
        self.metric_history["peaktopeak"] = np.full((epochs, 3), large_value)
        self.metric_history["unscaled_mse"] = np.full((epochs, 3), large_value)
        self.metric_history["rel_meanerr"] = np.full((epochs, 3), large_value)
        self.metric_history["model_best_metric"] = np.full((epochs, 3), large_value)
        timer_start = time.perf_counter()
        # Iteration loop
        for epoch in range(epochs):
            # Training loop
            self.model.train()
            self.Train_O = []
            self.Train_Pred = []
            for idx_b, batch in enumerate(self.dataloaders["train"]):
                # print(f"{idx_b+1}/{len(self.dataloaders['train'])}")
                batch_train_i, batch_train_o = batch
                batch_train_i = batch_train_i.to(self.device)
                batch_train_o = batch_train_o.to(self.device)
                # Forward
                with torch.no_grad():
                    batch_train_i = self.scaler_i.scale(batch_train_i)
                    batch_train_o = self.scaler_o.scale(batch_train_o)
                batch_pred = self.model(batch_train_i)
                # Compute Loss
                loss = self.loss_fn(batch_train_o, batch_pred)
                reg = 0
                for n, p in self.model.named_parameters():
                    if "weight" in n:
                        reg += torch.sum(torch.square(p))
                loss = loss + reg_lambda * reg
                # Backward
                for p in self.model.parameters():
                    p.grad = None
                loss.backward()
                for o in self.optimizer_group:
                    o.step()
                # Append total
                self.Train_O.append(self.scaler_o.unscale(batch_train_o).detach().cpu())
                self.Train_Pred.append(self.scaler_o.unscale(batch_pred).detach().cpu())
            # Free memory
            del batch_train_i, batch_train_o, batch_pred
            torch.cuda.empty_cache()
            # Concatenate
            self.Train_O = torch.cat(self.Train_O, dim=0)
            self.Train_Pred = torch.cat(self.Train_Pred, dim=0)

            # Validation loop
            with torch.no_grad():
                self.model.eval()
                self.Valid_O = []
                self.Valid_Pred = []
                for idx_b, batch in enumerate(self.dataloaders["valid"]):
                    # print(f"{idx_b + 1}/{len(self.dataloaders['train'])}")
                    batch_valid_i, batch_valid_o = batch
                    batch_valid_i = batch_valid_i.to(self.device)

                    batch_valid_i = self.scaler_i.scale(batch_valid_i)
                    batch_pred = self.model(batch_valid_i)
                    self.Valid_O.append(batch_valid_o)
                    self.Valid_Pred.append(self.scaler_o.unscale(batch_pred).cpu())
            # Free memory
            del batch_valid_i, batch_valid_o, batch_pred
            torch.cuda.empty_cache()
            # Concatenate
            self.Valid_O = torch.cat(self.Valid_O, dim=0)
            self.Valid_Pred = torch.cat(self.Valid_Pred, dim=0)

            self.compute_metrics(epoch)
            self.print_status(epoch)
            if (epoch + 1) % lr_halflife == 0:
                self.decay_lr()

            # Remember the best score
            if self.metric_history[self.valid_metric][epoch, 1] < model_best_metric:
                """
                Must use deepcopy
                https://tutorials.pytorch.kr/beginner/saving_loading_models.html
                """
                model_best_parameter = deepcopy(self.model.state_dict())
                model_best_metric = self.metric_history[self.valid_metric][epoch, 1]
                model_best_metric_index = epoch

            # Report ray
            if report_ray:
                tune.report(**{k: v[epoch, 1] for k, v in self.metric_history.items()})
        # Report ray the best metric
        if report_ray:
            tune.report(**{k: np.min(v[:, 1]) for k, v in self.metric_history.items()})

        timer_end = time.perf_counter()
        hr, min, sec = sec2hms(timer_end - timer_start)
        print(f"Training time is {hr} hr {min} min {sec:.2f}sec.")
        print(
            f"Minimum validation set '{self.valid_metric}' {model_best_metric:.5f} at epoch {model_best_metric_index + 1}."
        )
        result = {
            "state_dict": model_best_parameter,
            "metric_history": self.metric_history,
            "model_info": self.model.model_info,
            "model_init_args": self.model.model_init_args,
            "scaler_i": self.scaler_i,
            "scaler_o": self.scaler_o,
        }
        if save_filename:
            torch.save(result, f"models/{save_filename}.pt")
        print(f"Saved model at 'models/{save_filename}.pt'")

    def decay_lr(self):
        assert "optimizer_group" in self.__dict__.keys()
        for o in self.optimizer_group:
            o.param_groups[0]["lr"] /= 2

    def print_status(self, current_epoch):
        assert "metric_history" in self.__dict__.keys()
        str_status = f"Epoch: {current_epoch + 1}\n"
        str_status += f"Input: {self.input_columns}\n"
        str_status += f"Output: {self.output_columns}\n"
        str_status += f"Training set / Validation set\n"
        for metric_name, history in self.metric_history.items():
            str_status += f"{metric_name}: {history[current_epoch, 0]:.5f} / {history[current_epoch, 1]:.5f}\n"
        print(str_status)

    def compute_metrics(self, current_epoch):
        assert "metric_history" in self.__dict__.keys()
        assert "valid_metric" in self.__dict__.keys()
        datasets = [[self.Train_O, self.Train_Pred], [self.Valid_O, self.Valid_Pred]]
        for i, d in enumerate(datasets):
            label, pred = d
            self.scaler_o.to("cpu")
            self.metric_history["spearmanr"][current_epoch, i] = -spearmanr(
                label, pred
            ).correlation
            self.metric_history["corrcoef"][current_epoch, i] = -torch.corrcoef(
                torch.cat([label.T, pred.T], dim=0)
            )[0, 1]
            self.metric_history["r2"][current_epoch, i] = -r2_score(label, pred)
            self.metric_history["rms"][current_epoch, i] = rmserr(label, pred)
            self.metric_history["peaktopeak"][current_epoch, i] = peaktopeakerr(
                label, pred
            )
            self.metric_history["unscaled_mse"][current_epoch, i] = torch.mean(
                torch.square(self.scaler_o.unscale(label) - self.scaler_o.unscale(pred))
            )
            self.metric_history["rel_meanerr"][current_epoch, i] = rel_meanerr(
                self.scaler_o.unscale(label), self.scaler_o.unscale(pred)
            )
            self.metric_history["model_best_metric"][current_epoch, i] = np.min(
                self.metric_history[self.valid_metric][current_epoch, i]
            )
            self.scaler_o.to(self.device)
