from architectures.convlstm import ConvLSTM
from utils.trainer import Trainer
from utils.datareader import *
from tune_hyperparams import load_result
import numpy as np
from variables import global_vars
import torch
import warnings


def preprocess():
    tocsv_Feb2023()
    preprocess_Feb2023(sampling_rate=20)
    preprocess_Feb2023(sampling_rate=100)
    transform_coordinates()


def Train(sampling_rate):
    for f_name in global_vars.output_features:
        # Correlation analysis
        output_columns = [f_name]
        datalist = glob.glob(
            f"data/Data_Feb2023/csv/preprocessed_{sampling_rate}hz/*.csv"
        )
        dataset = []
        for d in datalist:
            dataset.append(pd.read_csv(d))
        dataset = pd.concat(dataset, axis=0)[
            global_vars.input_features + output_columns
        ]

        # Load optimized configs
        optim_result_path = f"./models/RayTune/tune_20hz_{f_name}"
        config = load_result(
            optim_result_path, print_logs=False, return_best_config=True
        )

        # N correlated inputs
        correlation = dataset.corr("pearson")[f_name].drop(labels=output_columns)
        input_columns = correlation.nlargest(config["in_channels"]).index.tolist()

        # Train
        net = ConvLSTM(
            input_columns=input_columns,
            output_columns=config["output_columns"],
            input_timesteps=config["input_timesteps"],
            cnn_blocks=config["cnn_blocks"],
            cnn_kernel_size=config["cnn_kernel_size"],
            cnn_stride=config["cnn_stride"],
            cnn_activation=config["cnn_activation"],
            rnn_states=config["rnn_states"],
            rnn_layers=config["rnn_layers"],
            rnn_dropout=config["rnn_dropout"],
            rnn_bidirectional=config["rnn_bidirectional"],
            rnn_h_init=config["rnn_h_init"],
            linear_states=config["linear_states"],
            linear_layers=config["linear_layers"],
            linear_activation=config["linear_activation"],
        )
        trainer = Trainer(net)
        trainer.fit(
            epochs=round(100 / sampling_rate * 10),
            lr_halflife=9999,
            input_columns=input_columns,
            output_columns=config["output_columns"],
            input_timesteps=config["input_timesteps"],
            sampling_rate=sampling_rate,
            reg_lambda=config["reg_lambda"],
            valid_metric="corrcoef",
            save_filename=f"{sampling_rate}hz_{f_name}",
            # save_filename=None,
            report_ray=False,
            batch_size=config["batch_size"],
            initial_lr=config["initial_lr"],
            multioptim=config["multioptim"],
            optim_alg=config["optim_alg"],
            loss_fn="mse",
        )


def test_net_dim():
    warnings.simplefilter("ignore", UserWarning)
    N = 64
    C = 12
    L = 50
    input = torch.randn(N, C, L)
    net = ConvLSTM(
        input_columns=np.arange(C),
        output_columns=[1],
        input_timesteps=L,
        cnn_blocks=5,
        cnn_kernel_size=8,
        cnn_stride=2,
    )
    output = net(input)
    print(output.shape)


if __name__ == "__main__":
    # preprocess()
    # test_net_dim()
    Train(sampling_rate=20)
    # Train(sampling_rate=100)
