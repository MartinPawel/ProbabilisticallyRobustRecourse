import datetime
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from carla import log
from carla.recourse_methods.processing import reconstruct_encoding_constraints

DECISION_THRESHOLD = 0.5


def dice_recourse(
    torch_model,
    x: np.ndarray,
    cat_feature_indices: List[int],
    binary_cat_features: bool = True,
    feature_costs: Optional[List[float]] = None,
    lr: float = 0.01,
    lambda_param: float = 5,
    y_target: List[int] = [0, 1],
    n_iter: int = 1000,
    t_max_min: float = 0.5,
    norm: int = 1,
    clamp: bool = True,
    loss_type: str = "MSE",
    total_cfs: int = 2,
    setting='classification',
    diversity_loss_type: str = "avg_dist",
    cfs: List = [],
    max_iter=1000,
    step=0.25
) -> np.ndarray:
    """

    Parameters
    ----------
    torch_model: black-box-model to discover
    x: factual to explain
    cat_feature_indices: list of positions of categorical features in x
    binary_cat_features: If true, the encoding of x is done by drop_if_binary
    feature_costs: List with costs per feature
    lr: learning rate for gradient descent
    lambda_param: weight factor for feature_cost
    y_target: List of one-hot-encoded target class
    n_iter: maximum number of iteration
    t_max_min: maximum time of search
    norm: L-norm to calculate cost
    clamp: If true, feature values will be clamped to (0, 1)
    loss_type: String for loss function (MSE or BCE)

    Returns
    -------
    Counterfactual example as np.ndarray
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # returns counterfactual instance
    torch.manual_seed(0)

    if feature_costs is not None:
        feature_costs = torch.from_numpy(feature_costs).float().to(device)

    x = torch.from_numpy(x).float().to(device)
    y_target = torch.tensor(y_target).float().to(device)
    lamb = torch.tensor(lambda_param).float().to(device)
    # x_new is used for gradient search in optimizing process
    x_new = Variable(x.clone(), requires_grad=True)
    # x_new_enc is a copy of x_new with reconstructed encoding constraints of x_new
    # such that categorical data is either 0 or 1
    x_new_enc = reconstruct_encoding_constraints(
        x_new, cat_feature_indices, binary_cat_features
    )

    optimizer = torch.optim.Adam([x_new], lr, amsgrad=True)
    softmax = nn.Softmax()

    if loss_type == "MSE":
        loss_fn = torch.nn.MSELoss()
        f_x_new = softmax(torch_model(x_new))[1]
    else:
        loss_fn = torch.nn.BCELoss()
        f_x_new = torch_model(x_new)[:, 1]

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)

    # These methods are inspired by the original DICE code
    cfs = do_cf_initializations(dim=x.shape[1], total_cfs=total_cfs)
    cfs = initialize_cfs(cfs, total_cfs, x, init_near_query_instance=True)

    xprime = []
    for i in range(total_cfs):
        x = cfs[i].clone().detach()
        x.requires_grad_(True)
        xprime.append(x)

    if optimizer == 'adam':
        optim = torch.optim.Adam(xprime, lr)
    else:
        optim = torch.optim.RMSprop(xprime, lr)

    # Timer
    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)

    outputs = []
    for i in range(total_cfs):
        output = _call_model(torch_model, x.reshape(1, -1), setting)[1]
        outputs.append(output)

    lam = lamb
    counterfactuals = []
    while not _check_cf_valid(outputs, 1):
    
        iter = 0
    
        distances = []
        all_loss = []
    
        while not _check_cf_valid(outputs, 1) and iter < max_iter:
            optim.zero_grad()
            total_loss, loss_distance = compute_loss(torch_model,
                                                     _lambda=lam,
                                                     counterfactuals=xprime,
                                                     original_instance=x,
                                                     target=torch.tensor(1),
                                                     setting=setting,
                                                     diversity_loss_type=diversity_loss_type
                                                     )
            total_loss.backward()
            optim.step()
        
            outputs = []
            for i in range(total_cfs):
                output = _call_model(torch_model, xprime[i].reshape(1, -1), setting)[1]
                outputs.append(output)
                # self.cfs[i].requires_grad_(True)
        
            # print(output)
            if _check_cf_valid(outputs, 1):
                coin = np.random.binomial(n=1, p=0.5)
                cf = xprime[int(coin)]
                counterfactuals.append(cf.detach())
                distances.append(loss_distance.detach())
                all_loss.append(total_loss.detach().numpy())
        
            iter = iter + 1
    
        # print("output:", output)
        if datetime.datetime.now() - t0 > t_max:
            # print('Timeout - No counterfactual explanation found')
            break
        elif _check_cf_valid(outputs, 1):
            # print('Counterfactual explanation found')
            pass
    
        if step == 0.0:  # Don't search over lambdas
            break
        else:
            lam -= step

    if not len(counterfactuals):
        print('No counterfactual explanation found')
        return None

    # Choose the nearest counterfactual
    counterfactuals = torch.stack(counterfactuals)
    distances = torch.stack(distances)
    distances = distances.detach().numpy()
    index = np.argmin(distances)
    counterfactuals = counterfactuals.detach()

    return counterfactuals[index]
  
  
def do_cf_initializations(dim, total_cfs):
    """Intializes CFs and other related variables."""
    
    # CF initialization
    cfs = []
    for ix in range(total_cfs):
        one_init = []
        for jx in range(dim):
            one_init.append(np.random.uniform(0, 1))
        cfs.append(torch.tensor(one_init).float())
    return cfs
 
    
def initialize_cfs(cfs, total_cfs, query_instance, init_near_query_instance=True):
    """Initialize counterfactuals."""
    for n in range(total_cfs):
        for i in range(query_instance.shape[1]):
            if init_near_query_instance:
                cfs[n].data[i] = query_instance[0, i] + (n * 0.01)
            else:
                cfs[n].data[i] = np.random.uniform(0, +1)
    return cfs
    
    
def compute_loss(torch_model,
                 _lambda: float,
                 original_instance: torch.tensor,
                 counterfactuals: list,
                 target: torch.tensor,
                 setting,
                 diversity_loss_type) -> torch.tensor:
    
    outputs = []
    loss_classification = 0.0
    
    for i in range(2):
        output = _call_model(torch_model, counterfactuals[i].reshape(1, -1), setting)[1]
        # print('output', output)
        outputs.append(output)
        # output.backward()
    # classification loss
    if setting == "classification":
        bce_loss = nn.BCELoss()
        for i in range(2):
            loss_classification += (1 / 2) * bce_loss(outputs[i].reshape(-1), target.reshape(-1).float())
    
    elif setting == "regression":
        mse_loss = nn.MSELoss()
        for i in range(2):
            loss_classification += (1 / 2) * mse_loss(outputs[i][0].reshape(-1), target.reshape(-1))
    else:
        raise ValueError("Illegal setting. Only classification and regression are supported.")
    
    # distance loss
    loss_distance = proximity_loss(original_instance, counterfactuals)
    # diversity loss
    diversity_loss = compute_diversity_loss(counterfactuals, diversity_loss_type)
    # total loss
    total_loss = loss_classification + _lambda * loss_distance + diversity_loss
    # print("total loss:", total_loss)
    
    return total_loss, loss_distance
 
    
def proximity_loss(original_instance, counterfactuals):
    proximity_loss = 0.0
    for i in range(2):
        proximity_loss += torch.norm((counterfactuals[i] - original_instance), 1)
    return proximity_loss / 2


def _call_model(model, cf_candidate, setting):
    if setting == "classification":
        output = model(cf_candidate)[0]
    
    elif setting == "regression":
        output = model.predict_with_logits(cf_candidate).reshape(1, -1)
    
    else:
        raise ValueError("Illegal setting. Only classification and regression are supported.")
    return output
    
    
def _check_cf_valid(outputs, target_class):
    """ Check if the output constitutes a sufficient CF-example.
        target_class = 1 in general means that we aim to improve the score,
        whereas for target_class = 0 we aim to decrese it.
    """
    if target_class == 1:
        checks = []
        for output in outputs:
            check = output >= 0.5
            checks.append(check)
        # print(checks)
        return all(checks)
    else:
        checks = []
        for output in outputs:
            check = output <= 0.5
            checks.append(check)
        return all(checks)
    
    
def dpp_style(submethod, counterfactuals):
    """Computes the DPP of a matrix."""
    det_entries = torch.ones((2, 2))
    if submethod == "inverse_dist":
        for i in range(2):
            for j in range(2):
                det_entries[(i, j)] = 1.0 / (1.0 + torch.norm(counterfactuals[i] - counterfactuals[j], 1))
                if i == j:
                    det_entries[(i, j)] += 0.0001
        
    elif submethod == "exponential_dist":
        for i in range(2):
            for j in range(2):
                det_entries[(i, j)] = 1.0 / (
                    torch.exp(torch.norm(counterfactuals[i] - counterfactuals[j], 1)))
                if i == j:
                    det_entries[(i, j)] += 0.0001
    
    diversity_loss = torch.det(det_entries)
    return diversity_loss


def compute_diversity_loss(counterfactuals, diversity_loss_type, total_cfs=2):
    """Computes the third part (diversity) of the loss function."""
    if total_cfs == 1:
        return torch.tensor(0.0)
    
    if "dpp" in diversity_loss_type:
        submethod = diversity_loss_type.split(':')[1]
        return dpp_style(submethod, counterfactuals)
    elif diversity_loss_type == "avg_dist":
        diversity_loss = 0.0
        count = 0.0
        # computing pairwise distance and transforming it to normalized similarity
        for i in range(total_cfs):
            for j in range(i + 1, total_cfs):
                count += 1.0
                diversity_loss += 1.0 / (1.0 + torch.norm(counterfactuals[i] - counterfactuals[j], 1))
    
    return 1.0 - (diversity_loss / count)
