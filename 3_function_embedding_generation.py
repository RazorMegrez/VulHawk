import torch
from utils.runtime import BlockEmbedding, ControlGraphEmbedding, Adapters, AdapterBlock, architecture_map
from Tokenizer.InstructionTokenizer import InstructionTokenizer
from utils.FunctionNet import Net, MyData
from utils.libs import S2VGraph, EntropyHead, EntropyModel, search_read_pickle
import traceback
import pickle
from torch_geometric.loader import DataLoader as FunctionLoader
from tqdm import tqdm
import os

def build_for_one_list(inputs=[], min_blk=3,  add_block_features=False):
    FunctionName = []
    VectorTable = []
    DetailData = {}
    FunctionMap = {}
    global CfgEmbedding
    id = 0


    input_loader = FunctionLoader([MyData(_.node_features, _.edge_mat, torch.tensor(_.entropy+8*architecture_map[_.architecture])) for _ in inputs],
                           batch_size=1200)
    pbar = tqdm(input_loader)
    representations = []
    for batch_func in pbar:
        file_environments = batch_func.edge_attr
        batch_output = CfgEmbedding.generate(batch_func, file_environments)
        representations.extend(batch_output)

    for i, func in enumerate(inputs):
        funcName = func.label
        funcName = func.binary_name + os.path.sep + funcName
        funcName = funcName
        if len(func) < min_blk:
            continue

        func_features = representations[i].cpu()
        FunctionName.append((id, funcName))
        DetailData[id] = {"binary_name": func.binary_name,
                          "funcname": funcName,
                          "callers": list(set(func.callers)),
                          "callees": list(set(func.callees)),
                          "cg_hash": func.cg_hash,
                          "strings": func.strings,
                          "imports": func.imports,
                          "architecture": func.architecture,
                          "file_environments": func.entropy,
                          }
        if add_block_features:
            DetailData[id]["block"] = func.node_features
        FunctionMap[funcName] = id
        id += 1
        VectorTable.append(func_features)

    return FunctionName, VectorTable, FunctionMap, DetailData

def generate_embedding(filename):
    BlockData = []
    try:
        func_dict = search_read_pickle(filename, BlkEmbedding=BlkEmbedding, dim=256, min_blk=3)
        BlockData.extend(list(func_dict.values()))
    except Exception as e:
        traceback.print_exc()

    FunctionName, VectorTable, FunctionMap, DetailData = build_for_one_list(BlockData, min_blk=3, add_block_features=True)
    pickle.dump((FunctionName, VectorTable, FunctionMap, DetailData), open(filename.replace(".pkl", ".emb"), "wb"))


if __name__ == '__main__':
    # cmd64 = "path\\to\\ida64.exe"
    # script = "path\\to\\script.py"
    model_path = "VulHawk_store/"
    blkEmbedding = "./VulHawk_store/block"
    tokenizer_path = "Tokenizer/model_save/tokenizer.model"
    device = torch.device('cuda:0')
    DIM = 256

    BlkEmbedding = BlockEmbedding(model_directory=blkEmbedding, tokenizer=tokenizer_path, device=device)
    CfgEmbedding = ControlGraphEmbedding(pretrained_model=model_path, use_adapter=True)
    generate_embedding("example/inputBinaries/O3/b2sum.pkl")
    generate_embedding("example/inputBinaries/O2/b2sum.pkl")
    generate_embedding("example/inputBinaries/O1/b2sum.pkl")