import torch
import numpy as np
from copy import deepcopy


def mse(label, pred):
    return torch.mean(torch.square(label - pred))


def rmserr(label, pred):
    label = label.numpy().copy()
    pred = pred.numpy().copy()
    # centering
    label = label - label.mean()
    pred = pred - pred.mean()
    # compute
    rms_label = np.sqrt(np.mean(np.square(label)))
    rms_prediction = np.sqrt(np.mean(np.square(pred)))
    rms_error = np.abs(rms_label - rms_prediction) / rms_label * 100
    return rms_error


def crosscorr(label, pred):
    label = label.numpy()
    pred = pred.numpy()
    max, min = label.max(), label.min()
    label = (label - min) / (max - min)
    pred = (pred - min) / (max - min)
    coeff = np.corrcoef(label, pred, rowvar=False)
    return np.abs(coeff[0, 1])  # exp to sim


def peaktopeakerr(label, pred):
    label = label.numpy()
    pred = pred.numpy()
    max, min = label.max(), label.min()
    label = (label - min) / (max - min)
    pred = (pred - min) / (max - min)
    label_err = np.max(label) - np.min(label)
    pred_err = np.max(pred) - np.min(pred)
    err = np.abs(label_err - pred_err) / label_err * 100
    return err


def rel_meanerr(label, pred):
    label = label.numpy().flatten()
    pred = pred.numpy().flatten()
    # normalize
    min, max = np.min(np.concatenate([label, pred])), np.max(
        np.concatenate([label, pred])
    )
    label = (label - min) / (max - min)
    pred = (pred - min) / (max - min)
    # compute mean value error
    err = np.mean(np.abs((label - pred) / label.mean())) * 100
    return err


def mae(label, pred):
    label = label.numpy().flatten()
    pred = pred.numpy().flatten()
    # compute MAE
    err = np.mean(np.abs(label - pred))
    return err


def mape(label, pred):
    label = label.numpy().flatten()
    pred = pred.numpy().flatten()
    # compute MAPE
    err = np.mean(np.abs((label - pred) / label)) * 100
    return err


def mase(label, pred):
    label = label.numpy().flatten()
    pred = pred.numpy().flatten()
    # compute non-seasonal MASE
    n = np.mean(np.abs(label - pred))  # N points
    d = np.mean(np.abs(np.diff(label)))  # N-1 points
    err = n / d
    return err


def smape(label, pred):
    label = label.numpy().flatten()
    pred = pred.numpy().flatten()
    # compute sMAPE
    n = np.abs(label - pred)
    d = np.abs(label) + np.abs(pred)
    err = np.mean(n / d) * 100
    return err
