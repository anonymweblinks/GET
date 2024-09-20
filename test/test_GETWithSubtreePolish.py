import torch 
torch.set_default_dtype(torch.float32)
import numpy as np
import pandas as pd
import time
import json 

import sys
sys.path.append('./src/')

from dataset import loadDataset
from treeFunc import readTreePath, objv_cost
from warmStart import CART_Reg_warmStart
from SubtreePolish import RT as MH3BaseInitRT



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
            X_train = torch.from_numpy(data_train[:, 0:p] * 1.0).float()            # casts the tensor to a float32 data type
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

            objv_TreeGPSub, TreeGPSub, objv_TreeGWT, TreeGWT, time_GWTElapsed = MH3BaseInitRT(X, Y, treeDepth, indices_flags_dict, epochNum, device, startNum)
            TimeGPSub = time.perf_counter()-startTime

            objv_MSE_train_GWT, r2TrainGWT = objv_cost(X, Y, treeDepth, TreeGWT)
            objv_MSE_test_GWT, r2TestGWT = objv_cost(X_test, Y_test, treeDepth, TreeGWT)
            objv_MSE_train_GPSub, r2TrainGPSub = objv_cost(X, Y, treeDepth, TreeGPSub)
            objv_MSE_test_GPSub, r2TestGPSub = objv_cost(X_test, Y_test, treeDepth, TreeGPSub)

            print("Final Results...")
            print("\nobjvMSETrainGET: {};    objvMSETestGET: {}".format(objv_MSE_train_GWT, objv_MSE_test_GWT))
            print("r2TrainGET: {};    r2TestGET: {}".format(r2TrainGWT, r2TestGWT))
            print("\nobjvMSETrainGETwithSubTree: {};    objvMSETestGETwithSubTree: {}".format(objv_MSE_train_GPSub, objv_MSE_test_GPSub))
            print("r2TrainGETwithSubTree: {};    r2TestGETwithSubTree: {}".format(r2TrainGPSub, r2TestGPSub))
            print("\nelapsedTime: {}\n".format(TimeGPSub))





