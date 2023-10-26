import glob
import scipy
import os
import pandas as pd
import numpy as np
from scipy.interpolate import Akima1DInterpolator as akispl
from scipy.interpolate import interp1d


def readmat(filepath: str):
    mat = scipy.io.loadmat(filepath)
    del_keys = [
        "__header__",
        "__version__",
        "__globals__",
        "None",
        "__function_workspace__",
    ]
    for key in del_keys:
        try:
            del mat[key]
        except KeyError:
            continue
    return mat


def readmatset_Feb2023(datapath: str = "data/Data_Feb2023"):
    files = glob.glob(f"{datapath}\\**\\**.mat", recursive=True)
    datadict = {}
    for f in files:
        namePre = os.path.dirname(f).split("\\")[-1]
        namePre = "_".join(namePre.split("_")[:-1])
        namePost = os.path.basename(f).replace(".mat", "")
        if "test" in namePost.split("_"):
            namePost = "_".join(namePost.split("_")[5:])
        try:
            datadict[f"{namePre}"].update(readmat(f))
        except KeyError:
            datadict[f"{namePre}"] = {}
            datadict[f"{namePre}"].update(readmat(f))
    return datadict


def tocsv_Feb2023(datapath: str = "data/Data_Feb2023"):
    datadict = readmatset_Feb2023(datapath)
    # Drop non-filtered measurements
    for exp_name, exp_dataset in list(datadict.items()):
        for sensor_name, sensor_data in list(exp_dataset.items()):
            if not "Filtered" in sensor_name:
                continue
            elif "Force" in sensor_name:
                columns = ["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            elif "IMU" in sensor_name:
                columns = [
                    "time",
                    "imu_roll",
                    "imu_pitch",
                    "imu_yaw",
                    "imu_roll_angvel",
                    "imu_pitch_angvel",
                    "imu_yaw_angvel",
                    "imu_x_linacc",
                    "imu_y_linacc",
                    "imu_z_linacc",
                ]
            elif "JointDiffPressure" in sensor_name:
                columns = ["time", "diffp_j1", "diffp_j2", "diffp_j3", "diffp_j4"]
            elif "ToolPressure" in sensor_name:
                columns = ["time", "hydraulic_pressure"]
            elif "Depth" in sensor_name:
                columns = ["time", "depth"]
            elif "ToolPos" in sensor_name:
                columns = [
                    "time",
                    "pos_x",
                    "pos_y",
                    "pos_z",
                    "pos_roll",
                    "pos_pitch",
                    "pos_yaw",
                ]
            sensor_data_df = pd.DataFrame(sensor_data, columns=columns)
            sensor_data_df.to_csv(
                f"data/Data_Feb2023/csv/raw_data/{exp_name}_{sensor_name}.csv",
                index=False,
            )


def preprocess_Feb2023(datapath: str = "data/Data_Feb2023", sampling_rate=100):
    csv_list = glob.glob(f"{datapath}/csv/raw_data/*.csv")
    # Get names
    test_names = []
    sensor_names = []
    for csv in csv_list:
        file_name = ".".join(os.path.basename(csv).split(".")[:-1])
        test_name = "_".join(file_name.split("_")[:4])
        sensor_name = "_".join(file_name.split("_")[4:])
        test_names.append(test_name)
        sensor_names.append(sensor_name)
    test_names = list(dict.fromkeys(test_names))
    sensor_names = list(dict.fromkeys(sensor_names))

    # Analyze with nested loop
    dataset_props = {}
    for test_name in test_names:
        test_props = {}
        for sensor_name in sensor_names:
            data_fixed = pd.read_csv(
                f"{datapath}/csv/raw_data/{test_name}_{sensor_name}.csv"
            )
            time_bounds = (data_fixed["time"].iloc[0], data_fixed["time"].iloc[-1])
            freq = data_fixed.shape[0] / (time_bounds[1] - time_bounds[0])
            shape = data_fixed.shape
            time_vec = data_fixed.to_numpy()[:, 0]
            measurements = data_fixed.to_numpy()[:, 1:]
            interpolator = interp1d(
                time_vec, measurements, axis=0, kind="zero", fill_value="extrapolate"
            )
            test_props[sensor_name] = {
                "time_bounds": time_bounds,
                "freq": freq,
                "shape": shape,
                "time_vec": time_vec,
                "interpolator": interpolator,
            }
        dataset_props[test_name] = test_props

    # Interpolate and concatenate along tests
    for test_name in test_names:
        if sampling_rate == 100:
            test_global_time_vec = dataset_props[test_name]["Filtered_Force_Data"][
                "time_vec"
            ]  # 100hz
        elif sampling_rate == 20:
            test_global_time_vec = dataset_props[test_name][
                "Filtered_ToolPressure_Data"
            ][
                "time_vec"
            ]  # 20hz
        columns = ["time"]
        interp_values = [test_global_time_vec]
        for sensor_name in sensor_names:
            columns += (
                pd.read_csv(f"{datapath}/csv/raw_data/{test_name}_{sensor_name}.csv")
                .columns[1:]
                .tolist()
            )
            interp_values.append(
                dataset_props[test_name][sensor_name]["interpolator"](
                    test_global_time_vec
                )
            )
        interp_values[0] = interp_values[0].reshape(
            -1, 1
        )  # Reshape time vector (2D interpolator)
        interp_values = np.concatenate(interp_values, axis=1)
        interp_values = pd.DataFrame(interp_values, columns=columns)
        # Save
        interp_values.to_csv(
            f"{datapath}/csv/preprocessed_{sampling_rate}hz/{test_name}.csv",
            index=False,
        )


def transform_coordinates(datapath: str = "data/Data_Feb2023"):
    # Match all coordinates to global origin coordiate
    # IMU Sensors == Global coordinate
    csv_list_new = glob.glob(f"{datapath}/csv/preprocessed_100hz/*.csv")
    csv_list_new += glob.glob(f"{datapath}/csv/preprocessed_20hz/*.csv")
    for csv in csv_list_new:
        data_original = pd.read_csv(csv)
        data_fixed = data_original.copy()
        # FT Sensors
        data_fixed["Fx"] = data_original["Fy"]
        data_fixed["Fy"] = -data_original["Fx"]
        data_fixed["Fz"] = data_original["Fz"]
        data_fixed["Mx"] = data_original["My"]
        data_fixed["My"] = -data_original["Mx"]
        data_fixed["Mz"] = data_original["Mz"]
        # Tool Sensors
        data_fixed["pos_x"] = data_original["pos_x"]
        data_fixed["pos_y"] = -data_original["pos_y"]
        data_fixed["pos_z"] = -data_original["pos_z"]
        data_fixed["pos_roll"] = data_original["pos_roll"]
        data_fixed["pos_pitch"] = -data_original["pos_pitch"]
        data_fixed["pos_yaw"] = -data_original["pos_yaw"]
        # Save
        data_fixed.to_csv(csv, index=False)
