# Top Network

from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import itertools
from torch.nn import LayerNorm, Linear, ReLU
import torch
import math
import os
import torch.nn.functional as F
from torch import nn
import time


from utils.runtime import EntropyModel, EntropyHead, arch_map

def build_file_so_exe(data, isLib=False):
    X = []
    Y = []
    Arch = []
    for i in data:
        if i["isLib"] == isLib:
            X.append(i["x"])
            Y.append(i["y"])
            Arch.append(i["archCode"])
    return X, Y, Arch

def build_file_size_dataset_max(data, MAX_SIZE=1024):
    X = []
    Y = []
    Arch = []
    for i in data:
        if i["fileSize"] <= MAX_SIZE:
            X.append(i["x"])
            Y.append(i["y"])
            Arch.append(i["archCode"])
    return X, Y, Arch

def build_file_size_dataset_min(data, MIN_SIZE=1024):
    X = []
    Y = []
    Arch = []
    for i in data:
        if i["fileSize"] > MIN_SIZE:
            X.append(i["x"])
            Y.append(i["y"])
            Arch.append(i["archCode"])
    return X, Y, Arch

def build_arch_dataset(data, arch="x86"):
    X = []
    Y = []
    Arch = []
    for i in data:
        if arch in i["arch"]:
            X.append(i["x"])
            Y.append(i["y"])
            Arch.append(i["archCode"])
    return X, Y, Arch

def print_accuracy(acuuracy_list):
    for i in acuuracy_list:
        for j in i:
            print(str(round(j, 3)).ljust(5, "0"), end="\t")
        print()

def evaluation(model, X, Y, Arch):
    model.eval()
    Y_pred = model(input_x=torch.stack(X), arch=Arch).detach().cpu()
    Y_pred = torch.argmax(Y_pred, axis=1)

    accuracy = accuracy_score(y_pred=Y_pred, y_true=Y)
    evaluate_(Y, Y_pred)
    return round(accuracy, 3)

def evaluate_(Y, Y_pred):
    results = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)]
    for i, y in enumerate(Y_pred):
        results[Y[i]][y] += 1
    results = np.array(results[0:4])+np.array(results[4:])
    results = results[:, 0:4] + results[:, 4:]
    results_percent = [[] for _ in range(4)]
    for _, i in enumerate(results):
        for j in i:
            results_percent[_].append(j/sum(i))
    print_accuracy(results_percent)

def evaluation_compiler(model, X, Y, Arch):
    model.eval()
    Y_pred = model(input_x=torch.stack(X), arch=Arch).detach().cpu()
    Y_pred = torch.argmax(Y_pred, axis=1)
    accuracy = accuracy_score(y_pred=Y_pred>4, y_true=torch.Tensor(Y)>4)
    return round(accuracy, 3)


def load_model(load_path):
    model = torch.load(load_path + "model.bin")
    return model


model_path = "VulHawk_store/entropy/"
eta_model = load_model(model_path)

data = pickle.load(open("example/FileEnvironmentsData.pkl", "rb"))

for arch in ["x86", "mips", "arm", "-"]:
    X, Y, Arch = build_arch_dataset(data, arch)
    print("\nTesting for {}".format(arch))
    accuracy = evaluation(eta_model, X, Y, Arch)
    print("accuracy:\t", accuracy)

for isLib in [True, False]:
    X, Y, Arch = build_file_so_exe(data, isLib)
    exe_or_so = "so" if isLib else "exe"
    print("\nTesting for {}".format(exe_or_so))
    accuracy = evaluation(eta_model, X, Y, Arch)
    print("accuracy:\t", accuracy)

print("\nTesting for small binaries")
X, Y, Arch = build_file_size_dataset_max(data, 1024)
accuracy = evaluation(eta_model, X, Y, Arch)
print("accuracy:\t", accuracy)

print("\nTesting for big binaries")
X, Y, Arch = build_file_size_dataset_min(data, 1024)
accuracy = evaluation(eta_model, X, Y, Arch)
print("accuracy:\t", accuracy)

print("\nTesting for compilers")
X, Y, Arch = build_arch_dataset(data, "-")
accuracy = evaluation_compiler(eta_model, X, Y, Arch)
print("accuracy:\t", accuracy)





