import torch 
import sys 
import copy 
import time

sys.path.append('./src/')

from warmStart import CART_Reg_warmStart
from treeFunc import update_c, objv_cost, get_branch_nodes
from GET_alpApprx import  multiStartTreeOptbyGRAD_withC

def RT(X, y, treeDepth, indices_flags_dict, epochNum, device, startNum):

    time_GWTStart = time.perf_counter()
    a_init, b_init, c_init = CART_Reg_warmStart(X, y, treeDepth, device)
    cart_warmStart_dict = {"a": a_init, "b": b_init, "c": c_init}
    warmStart = [cart_warmStart_dict]
    objvMseBase, TreeCombBase = multiStartTreeOptbyGRAD_withC(X, y, treeDepth, indices_flags_dict, epochNum, device, warmStart, startNum)

    time_GWTElapsed = time.perf_counter() - time_GWTStart

    objv_TreeCombBase, r2_TreeCombBase = objv_cost(X, y, treeDepth, TreeCombBase)
    TreeOpt = copy.deepcopy(TreeCombBase)

    ## Polish the base tree by MH
    n, p = X.shape
    Tb = 2 ** treeDepth - 1                            # branch node size                     
    Tleaf = 2 ** treeDepth                             # leaf node size        
    a = torch.zeros((Tb, p), device= device, dtype=torch.float32)
    b = torch.zeros((Tb), device= device, dtype=torch.float32)
    updateIndList = []
    a, b, TreeOpt, objv_TreeOpt  = RT_inner(X, y, X, y, 1, treeDepth, indices_flags_dict, a, b, 1, epochNum, device, startNum, TreeOpt, objv_TreeCombBase, updateIndList)
    treeDepth_tensor = torch.tensor(treeDepth, dtype=torch.float32, device=device)
    TreeGPSub = {"a": a, "b": b, "c": torch.zeros(Tleaf, dtype=torch.float32, device=device), "D": treeDepth_tensor}
    TreeGPSub = update_c(X, y, treeDepth_tensor, TreeGPSub)
    objv_TreeGPSub, _ = objv_cost(X, y, treeDepth, TreeGPSub)

    if objv_TreeOpt < objv_TreeGPSub:
        return objv_TreeOpt, TreeOpt, objv_TreeCombBase, TreeCombBase, time_GWTElapsed
    else:
        return objv_TreeGPSub, TreeGPSub, objv_TreeCombBase, TreeCombBase, time_GWTElapsed




def RT_inner(X, y, X_all, y_all, dc, Dmax, indices_flags_dict, a_s, b_s, ind, epochNum, device, startNum, TreeOpt, objvOpt, updateIndList):

    TreeBasePolNode = copy.deepcopy(TreeOpt)
    H_ind = Dmax-dc+1

    BranchNodesList = get_branch_nodes(ind, H_ind)
    a_branchNodesWS = TreeOpt["a"][BranchNodesList, :]
    b_branchNodesWS = TreeOpt["b"][BranchNodesList]              
    Tleaf = 2 ** H_ind
    tree_branchNodeWS =  {"a": a_branchNodesWS, "b": b_branchNodesWS, "c": torch.zeros(Tleaf, device=device, dtype=torch.float32)}
    tree_branchNodeWS = update_c(X_all, y_all, H_ind, tree_branchNodeWS)
    tree_branchNodeWS = {"a": tree_branchNodeWS["a"], "b": -tree_branchNodeWS["b"], "c": tree_branchNodeWS["c"]}   # negative b here 

    a_init, b_init, c_init = CART_Reg_warmStart(X, y, H_ind, device)
    cart_warmStart_dict = {"a": a_init, "b": b_init, "c": c_init}
    warmStart = [cart_warmStart_dict, tree_branchNodeWS]
    objv_iter, tree = multiStartTreeOptbyGRAD_withC(X, y, H_ind, indices_flags_dict, epochNum, device, warmStart, startNum)

    a_s[ind-1, :] = tree["a"][0, :]
    b_s[ind-1] = tree["b"][0]

    TreeBasePolNode["a"][BranchNodesList, :] = tree["a"]    
    TreeBasePolNode["b"][BranchNodesList] = tree["b"]
    TreeBasePolNode = update_c(X_all, y_all, Dmax, TreeBasePolNode)
    objv_TreeBasePolNode, r2_TreeBasePolNode = objv_cost(X_all, y_all, Dmax, TreeBasePolNode)

    if objv_TreeBasePolNode < objvOpt:
        TreeOpt = copy.deepcopy(TreeBasePolNode)
        objvOpt = objv_TreeBasePolNode
        updateIndList.append(ind)
    else:
        pass
    
    stopDepthForMH = 4
    if Dmax > stopDepthForMH: 
        if dc == stopDepthForMH: 
            a_s[BranchNodesList, :] = tree["a"]
            b_s[BranchNodesList] = tree["b"]
            return a_s, b_s, TreeOpt, objvOpt
    else:
        # reach the depth 
        if dc == Dmax:
            return a_s, b_s, TreeOpt, objvOpt

    # node index for next layer 
    node_l = 2 * ind
    node_r = 2 * ind + 1
    yes = torch.matmul(X, a_s[ind-1, :]) <= b_s[ind-1]
    if y[yes].shape[0] > 1 and torch.unique(y[yes]).numel() > 1:
        a_s, b_s, TreeOpt, objvOpt = RT_inner(X[yes, :], y[yes], X_all, y_all, dc + 1, Dmax, indices_flags_dict, a_s, b_s, node_l, epochNum, device, startNum, TreeOpt, objvOpt, updateIndList)
    else:
        # print("y[yes].shape[0] <= 1")
        pass
    if y[~yes].shape[0] > 1 and torch.unique(y[~yes]).numel() > 1:
        a_s, b_s, TreeOpt, objvOpt = RT_inner(X[~yes, :], y[~yes], X_all, y_all, dc + 1, Dmax, indices_flags_dict, a_s, b_s, node_r, epochNum, device, startNum, TreeOpt, objvOpt, updateIndList)
    else:
        # print("y[~yes].shape[0] <= 1")
        pass
    
    return a_s, b_s, TreeOpt, objvOpt


