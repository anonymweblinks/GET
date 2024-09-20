
from sklearn.tree import _tree
from sklearn import tree
import numpy as np
import sklearn.metrics as metrics
import torch

## accelerate the get_branch_nodes function
def get_nodes_id(ind, Hind):
    ind -= 1                  # index starts from 0
    branchNodes = [ind]
    current_nodes = [ind]
    for _ in range(Hind-1):
        next_nodes = [2*node + j for node in current_nodes for j in [1, 2]]
        branchNodes.extend(next_nodes)
        current_nodes = next_nodes
    leftLeaf = 2*(ind+1) if Hind == 1 else 2*(next_nodes[0]+1)      # read index start from 1. eg, 32 means the 32th node, the first leaf node and index should be 31
    return branchNodes, leftLeaf


## retrieve the parameters abc of the trained tree model
def regTreeWarmStart(model, treeDepth):
    tree_ = model.tree_                  
    branchNode_inputDepth = 2**(treeDepth) - 1
    leafNode_inputDepth = 2**(treeDepth)
    a = [0]*branchNode_inputDepth
    b = [0]*branchNode_inputDepth
    c = [0]*leafNode_inputDepth

    ab0indList = []     
    def warmStartPara(node, ind):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            featureIdx = tree_.feature[node]
            threshold = tree_.threshold[node]
            a[ind-1] = featureIdx
            b[ind-1] = -threshold
            
            node_l = 2 * ind
            node_r = 2 * ind + 1
            warmStartPara(tree_.children_left[node], node_l)
            warmStartPara(tree_.children_right[node], node_r)
        
        else:
            if ind <= branchNode_inputDepth:
                currDepthForInd = int(np.log2(ind))
                diffDepthInbd = treeDepth - currDepthForInd
                ab0NodeListForEachInd, leftLeaf = get_nodes_id(ind, diffDepthInbd)
                ab0indList.extend(ab0NodeListForEachInd)
                c[leftLeaf-1-branchNode_inputDepth] = tree_.value[node].squeeze()
            else:
                c[ind-1-branchNode_inputDepth] = (tree_.value[node].squeeze())

    warmStartPara(0, 1)
    return a, b, c, ab0indList


def CART_Reg_warmStart(X, Y, treeDepth, device):
    model = tree.DecisionTreeRegressor(max_depth=treeDepth, min_samples_leaf=1, random_state=0)
    if device == torch.device('cuda'):
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    else:
        X_np, Y_np = X, Y

    if X_np.shape[0] < 1:
        branchNodeNum = 2**(treeDepth) - 1
        leafNodeNum = 2**(treeDepth)
        p = X_np.shape[1]
        b = [0]*branchNodeNum
        c = [0]*leafNodeNum
        a = np.zeros((branchNodeNum, p), dtype="float32")
        return a, np.asarray(b,dtype="float32"), np.asarray(c, dtype="float32")
        
    else:
        model = model.fit(X_np, Y_np)
        p = X.shape[1]
        a, b, c, ab0indList = regTreeWarmStart(model,treeDepth)
        # print("ab0indList", ab0indList)
        a_all = np.eye(p, dtype="float32")[a]                   # float32
        a_all[ab0indList] = 0

        return a_all, np.asarray(b,dtype="float32"), np.asarray(c, dtype="float32")



