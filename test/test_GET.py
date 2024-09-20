import torch 
import numpy as np
import pandas as pd
import time
import json

import sys
sys.path.append('./src/')

from dataset import loadDataset 
from treeFunc import readTreePath, objv_cost
from warmStart import CART_Reg_warmStart
from GET_alpApprx import multiStartTreeOptbyGRAD_withC


if __name__ == "__main__":
    ################## main Code ##################
    ## Args 
    data_num_start = int(sys.argv[1])
    data_num_end = int(sys.argv[2])
    runs_num_start = int(sys.argv[3])
    runs_num_end = int(sys.argv[4])
    treeDepth = int(sys.argv[5])                    
    epochNum = int(sys.argv[6])                      
    device_arg =  str(sys.argv[7])                  
    device = torch.device(device_arg)
    startNum = int(sys.argv[8]) 

    ##  data
    datasetPath = "./data/"
    # all datasets (all n>1000)
    Datasets_names = ["airfoil-self-noise", "space-ga", "abalone", "gas-turbine-co-emission-2015", "gas-turbine-nox-emission-2015",  "puma8NH",  "cpu-act", "cpu-small", "kin8nm", "delta-elevators", "combined-cycle-power-plant", "electrical-grid-stability", "condition-based-maintenance_compressor", "condition-based-maintenance_turbine", "ailerons", "elevators", "houses", "house-8L", "house-16H", "friedman-artificial", "protein-tertiary-structure",  "nasa-phm2008-1",  "power-consumption-tetouan-zone1", "power-consumption-tetouan-zone2", "power-consumption-tetouan-zone3"]

    ## read the treePath from the HDF5 file
    indices_flags_dict = readTreePath(treeDepth, device)

    datasetNum = len(Datasets_names)
    print("Starting: Total {} datasets".format(datasetNum))

    for datasetIdx in range(data_num_start-1, data_num_end):
        print("############# Dataset[{}]: {} #############".format(datasetIdx+1, Datasets_names[datasetIdx]))
        for run in range(runs_num_start, runs_num_end+1):
            print("####### Run: {} #######".format(run))
            torch.manual_seed(run)
            np.random.seed(run)
            data_train, data_valid, data_test = loadDataset(Datasets_names[datasetIdx], run, datasetPath)
            p = data_train.shape[1] - 1
            X_train = torch.from_numpy(data_train[:, 0:p] * 1.0).float()
            Y_train = torch.from_numpy(data_train[:, p] * 1.0).float()
            X_valid = torch.from_numpy(data_valid[:, 0:p] * 1.0).float()
            Y_valid = torch.from_numpy(data_valid[:, p] * 1.0).float()
            X_test = torch.from_numpy(data_test[:, 0:p] * 1.0).float()
            Y_test = torch.from_numpy(data_test[:, p] * 1.0).float()
            X = torch.cat((X_train, X_valid), 0)
            Y = torch.cat((Y_train, Y_valid), 0)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            X_test = X_test.to(device, non_blocking=True)
            Y_test = Y_test.to(device, non_blocking=True)

            if run == runs_num_start:
                print("dataset:{};    n_train:{};    n_valid:{};    n_test:{};    p:{}\n".format(Datasets_names[datasetIdx], X_train.shape[0], X_valid.shape[0], X_test.shape[0], X_train.shape[1]))

            startTime = time.perf_counter()

            # cart warm start
            a_init, b_init, c_init = CART_Reg_warmStart(X, Y, treeDepth, device)
            cart_warmStart_dict = {"a": a_init, "b": b_init, "c": c_init}
            treeCART = {"a": torch.tensor(a_init, device=device), "b": torch.tensor(-b_init, device=device), "c": torch.tensor(c_init, device=device)}
            objvMSECART, r2CART = objv_cost(X, Y, treeDepth, treeCART)
            objvMSECART_test, r2CART_test = objv_cost(X_test, Y_test, treeDepth, treeCART)
            print("objvMSECART: {};   r2CART: {}".format(objvMSECART, r2CART))
            print("objvMSECART_test: {};   r2CART_test: {}".format(objvMSECART_test, r2CART_test))

            warmStart = [cart_warmStart_dict]
            objv_MulitStart, tree = multiStartTreeOptbyGRAD_withC(X, Y, treeDepth, indices_flags_dict, epochNum, device, warmStart, startNum)

            objv_mse_train, r2_train = objv_cost(X, Y, treeDepth, tree)
            objv_mse_test, r2_test = objv_cost(X_test, Y_test, treeDepth, tree)
            elapsedTime = time.perf_counter() - startTime

            print("\nFinal Results...")
            print("\nobjvMseTrain: {};    objvMseTest: {}".format(objv_mse_train, objv_mse_test))
            print("r2Train: {};   r2Test: {}".format(r2_train, r2_test))
            print("\nelapsedTime: {}\n".format(elapsedTime))









