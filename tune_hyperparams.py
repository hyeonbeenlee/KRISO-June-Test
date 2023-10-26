import ray
import glob
import pandas as pd
import torch
import os
import shutil
import warnings
from architectures.convlstm import ConvLSTM
from utils.trainer import Trainer
from variables import global_vars
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import TuneConfig
from ray.air.config import RunConfig, CheckpointConfig, FailureConfig
from ray.tune import CLIReporter


def train_func(config=None):
    in_channels = config["in_channels"]
    sampling_rate = config["sampling_rate"]
    datapath = config["datapath"]
    output_columns = list(config["output_columns"])

    # Correlation analysis
    datalist = glob.glob(datapath)
    dataset = []
    for d in datalist:
        dataset.append(pd.read_csv(d))
    dataset = pd.concat(dataset, axis=0)[global_vars.input_features + output_columns]

    # N correlated inputs
    correlation = dataset.corr("pearson")[output_columns[0]].drop(labels=output_columns)
    input_columns = correlation.nlargest(in_channels).index.tolist()

    # Train
    config_net = dict(
        input_columns=input_columns,
        output_columns=output_columns,
        input_timesteps=config["input_timesteps"],
        cnn_blocks=config["cnn_blocks"],
        cnn_kernel_size=config["cnn_kernel_size"],
        cnn_stride=config["cnn_stride"],
        cnn_activation=config["cnn_activation"],
        rnn_states=config["rnn_states"],
        rnn_layers=config["rnn_layers"],
        rnn_dropout=config["rnn_dropout"],
        rnn_bidirectional=True,
        rnn_h_init=config["rnn_h_init"],
        linear_states=config["linear_states"],
        linear_layers=config["linear_layers"],
        linear_activation=config["linear_activation"],
    )
    config_fit = dict(
        epochs=config["epochs"],
        lr_halflife=config["lr_halflife"],
        input_columns=input_columns,
        output_columns=output_columns,
        input_timesteps=config["input_timesteps"],
        sampling_rate=sampling_rate,
        reg_lambda=config["reg_lambda"],
        valid_metric="corrcoef",
        save_filename=None,
        report_ray=True,
        batch_size=config["batch_size"],
        initial_lr=config["initial_lr"],
        multioptim=config["multioptim"],
        optim_alg=config["optim_alg"],
        loss_fn="mse",
    )

    net = ConvLSTM(**config_net)
    trainer = Trainer(net)
    trainer.fit(**config_fit)
    # Prevent CUDA out of memory
    del net, trainer, dataset
    torch.cuda.empty_cache()


def trial_str_creator(trial: ray.tune.experiment.trial.Trial):
    return f"{trial.trainable_name}_{trial.trial_id}"


def tune_nn():
    warnings.simplefilter("ignore", UserWarning)
    output_columns = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    for f_name in output_columns:
        # Correlation analysis parameters
        nn_config = dict(
            in_channels=tune.randint(3, 16), sampling_rate=20, output_columns=[f_name]
        )
        nn_config.update(
            dict(
                datapath=os.path.abspath(
                    f"data/Data_Feb2023/csv/preprocessed_{nn_config['sampling_rate']}hz/*.csv"
                )
            )
        )
        exp_name = f"tune_{nn_config['sampling_rate']}hz_{f_name}"
        # Delete existing ckpt folder
        if os.path.exists(f"{local_dir}/{exp_name}"):
            shutil.rmtree(f"{local_dir}/{exp_name}")
        else:
            pass

        # Network initialization
        nn_config.update(
            dict(
                input_timesteps=tune.randint(50, 200 + 1),
                cnn_blocks=tune.randint(1, 5 + 1),
                cnn_kernel_size=tune.randint(2, 8 + 1),
                cnn_stride=tune.choice([1, 2]),
                cnn_activation=tune.choice(["gelu", "selu", "elu"]),
                rnn_states=tune.randint(50, 400 + 1),
                rnn_layers=tune.randint(1, 4 + 1),
                rnn_dropout=tune.uniform(0, 0.5),
                rnn_bidirectional=True,
                rnn_h_init=tune.choice(["zeros", "randn", "learnable"]),
                linear_states=tune.randint(100, 600 + 1),
                linear_layers=tune.randint(1, 6 + 1),
                linear_activation=tune.choice(["gelu", "selu", "elu"]),
            )
        )
        # Training configuration
        nn_config.update(
            dict(
                epochs=15,
                lr_halflife=9999,
                # input_timesteps=config['input_timesteps'],
                # sampling_rate=20,
                reg_lambda=tune.loguniform(1e-4, 1e-2),
                valid_metric="corrcoef",
                save_filename=None,
                report_ray=True,
                batch_size=tune.choice([8, 16, 32, 64]),
                initial_lr=tune.loguniform(1e-4, 1e-2),
                multioptim=tune.choice([False, True]),
                optim_alg="radam",
                loss_fn="mse",
            )
        )

        # Configure tuning
        num_samples = 500  # number of search trials
        num_concurrency = 4  # number of parallel trials
        num_cpu_threads = 8 * num_concurrency  # total cpu usage
        # num_memory_gb = 6  # per trial # Deprecate, do not use
        search_alg = HyperOptSearch()  # Search algorithm
        search_alg = ConcurrencyLimiter(
            search_alg, max_concurrent=num_concurrency
        )  # Limits number of parallel runs, REQUIRED FOR STABILITY
        scheduler = ASHAScheduler()  # Trial early stopper
        reporter = CLIReporter(
            metric_columns=[
                "training_iteration",
                "spearmanr",
                "corrcoef",
                "r2",
                "rms",
                "peaktopeak",
                "unscaled_mse",
                "rel_meanerr",
                "model_best_metric",
            ],
            sort_by_metric=True,
            max_report_frequency=10,
            max_column_length=10,
            max_progress_rows=10,
            max_error_rows=5,
        )  # Console reporter
        # https://docs.ray.io/en/latest/ray-air/package-ref.html?highlight=TuneConfig#ray.tune.tune_config.TuneConfig
        # https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/logging_example.py
        tune_config = TuneConfig(
            metric="model_best_metric",
            mode="min",
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=num_samples,
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
        )
        # https://github.com/ray-project/ray/issues/10290
        # https://docs.ray.io/en/latest/ray-air/package-ref.html?highlight=CheckpointConfig#checkpoint
        run_config = RunConfig(
            name=exp_name,
            local_dir=local_dir,  # Checkpointing directory
            stop={
                "training_iteration": nn_config["epochs"],
            },  # Max epochs
            failure_config=FailureConfig(max_failures=2),
            checkpoint_config=CheckpointConfig(
                num_to_keep=None,
                checkpoint_score_attribute="model_best_metric",
                checkpoint_score_order="min",
                checkpoint_frequency=1,
            ),
            progress_reporter=reporter,
        )

        # trainable = train_func
        trainable = tune.with_resources(
            train_func,
            {"cpu": num_cpu_threads // num_concurrency, "gpu": 1 / num_concurrency},
        )  # GPU memory sharing
        # 'memory': num_memory_gb * 1024 ** 3})  # 1 GB = 1024^3 bytes
        tuner = tune.Tuner(
            trainable,
            param_space=nn_config,
            tune_config=tune_config,
            run_config=run_config,
        )
        # Run optimization
        results = tuner.fit()


def resume_tune(path: str = None):
    if path:
        tuner = tune.Tuner.restore(path=path)
        tuner.fit()
    elif not path:
        for i in range(6):
            output_columns = [OUTPUT[i]]
            prefix = output_columns[0].split("_")[-2]
            exp_name = f"Tuner_{prefix}"
            try:
                tuner = tune.Tuner.restore(path=f"{local_dir}/{exp_name}")
                tuner.fit()
            except:
                print(f"Resume on {local_dir}/{exp_name} unavailable. Continuing..")


def load_result(
    path,
    print_logs: bool = True,
    return_success: bool = False,
    return_best_config: bool = False,
):
    tuner = tune.Tuner.restore(path=path)
    results = tuner.get_results()
    results_df = results.get_dataframe(
        filter_metric="model_best_metric", filter_mode="min"
    )
    config_columns = [col for col in results_df.columns if "config" in col]
    config_columns_ = {
        col: col.split("/")[1] for col in results_df.columns if "config" in col
    }
    results_log = results_df[
        [
            "training_iteration",
            "corrcoef",
            "spearmanr",
            "r2",
            "rms",
            "peaktopeak",
            "model_best_metric",
        ]
    ]
    best_result_index = results_df["corrcoef"].argmin()
    results_best_config = results_df[config_columns].iloc[best_result_index]
    results_best_config = results_best_config.rename(index=config_columns_)
    if print_logs:
        print(results_log)
        print(results_best_config)
        print(results_best_config.shape)
    result_str = f"{path}: Maximum CrossCorr {results_df['corrcoef'].iloc[best_result_index]:.6f} at index {best_result_index}"
    print(result_str)
    if return_success:
        return results_df.shape[0]
    if return_best_config:
        return results_best_config.to_dict()


if __name__ == "__main__":
    """
    https://github.com/ray-project/ray/issues/27646
    윈도우에서 돌다가 Ctrl+C event 발생으로 터지는 이슈, 맨 마지막 참조 + import sys 추가 필요
    ConcurrencyLimiter + tune.with_resources(trainable, {'cpu': num_cpu_threads/num_concurrency, 'gpu':1/num_concurrency})
    1. Search space 가 error를 발생시키는 범위를 포함해서는 안된다. (또는 예외처리 필요)
    2. 병렬실행 수가 너무 많으면 안된다. (권장 4)
    3. Metric이 제대로 설정되었는지 확인
    4. 실행 도중 코드/변수명 수정 금지
    """
    local_dir = (
        "./models/RayTune"  # tensorboardX 관련 경로길이 제한이 있음...유의 -> trial_str_creator 로 해결
    )
    load_path = f"{local_dir}/tune_20hz_Fx"
    # tune_nn()
    # resume_tune(load_path)
    config = load_result(load_path, print_logs=True, return_best_config=True)
    print(config)
