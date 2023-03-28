import os
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
from torch import nn
from torch.nn import LayerNorm, Linear, ReLU
from torch import Tensor
import torch.nn.functional as F
import torch
from utils.libs import architecture_map
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from StringEmbedding.stringmodel import StringModel
from SimilarityCalibration import CalibrationModel


def evaluate_vulhuk_calibration(FunctionName, VectorTable, FunctionMap, DetailData, FunctionName2, VectorTable2,
                                FunctionMap2, DetailData2, seed=10, poolsize=10):
    calibrationModel = CalibrationModel()

    np.random.seed(seed)
    second_keys = np.random.randint(0, len(FunctionMap2), 10000).tolist()
    keys = [i for i in range(len(FunctionMap))]
    pbar = tqdm(keys)
    prefix = FunctionName2[0][1].split(os.path.sep)[:-1]

    Y_pred = []
    Y = []
    for key in pbar:
        funcname = FunctionName[key][1].split(os.path.sep)[-1:]
        anchor_vector = VectorTable[key]
        anchor_func = DetailData[key]

        positive_sample_name = os.path.sep.join(prefix + funcname)

        if positive_sample_name not in FunctionMap2:
            continue
        positive_sample_vector = VectorTable2[FunctionMap2[positive_sample_name]]
        s = anchor_vector.dist(positive_sample_vector).numpy()

        positive_sample_id = FunctionMap2[positive_sample_name]
        positive_sample_func = DetailData2[positive_sample_id]
        calibration_s = calibrationModel.calibrationSimilarity(anchor_func, positive_sample_func, s)
        Y_pred.append(calibration_s.cpu())
        Y.append(1)
        n = poolsize
        while True:
            random_key = second_keys.pop()
            neg_sample_vector = VectorTable2[random_key]
            neg_sample_name = FunctionName2[random_key][1]
            if os.path.basename(neg_sample_name) == os.path.basename(positive_sample_name):
                continue
            s = anchor_vector.dist(neg_sample_vector).numpy()
            neg_sample_id = FunctionMap2[neg_sample_name]
            neg_sample_func = DetailData2[neg_sample_id]
            calibration_s = calibrationModel.calibrationSimilarity(anchor_func, neg_sample_func, s)
            Y_pred.append(calibration_s.cpu())
            Y.append(0)
            n -= 1

            if n < 1:
                break
    print("AUC:\t", round(roc_auc_score(Y, Y_pred), 3))


if __name__ == '__main__':
    model_path = "VulHawk_store/adapter/"

    FunctionName, VectorTable, FunctionMap, DetailData = pickle.load(open("example/inputBinaries/O1/b2sum.emb", "rb"))
    FunctionName2, VectorTable2, FunctionMap2, DetailData2 = pickle.load(open("example/inputBinaries/O3/b2sum.emb", "rb"))

    ret = evaluate_vulhuk_calibration(FunctionName, VectorTable, FunctionMap, DetailData,
                                      FunctionName2, VectorTable2, FunctionMap2, DetailData2,
                                      seed=0)
