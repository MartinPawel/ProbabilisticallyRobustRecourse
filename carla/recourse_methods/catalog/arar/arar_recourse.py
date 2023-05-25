# utils
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import grad
from scipy.optimize import linprog
import torch
import numpy as np


class ArArRecourse:
    def __init__(self,
                 W=None,
                 W0=None,
                 y_target=1,
                 delta_max=0.1,
                 feature_costs=None,
                 pW=None,
                 pW0=None):
        
        self.set_W(W)
        self.set_W0(W0)
        
        self.set_pW(pW)
        self.set_pW0(pW0)
        
        self.y_target = torch.tensor(y_target).float()
        self.delta_max = delta_max
        self.feature_costs = feature_costs
        if self.feature_costs is not None:
            self.feature_costs = torch.from_numpy(feature_costs).float()
    
    def set_W(self, W):
        self.W = W
        if W is not None:
            self.W = torch.from_numpy(W).float()
    
    def set_W0(self, W0):
        self.W0 = W0
        if W0 is not None:
            self.W0 = torch.from_numpy(W0).float()
    
    def set_pW(self, pW):
        self.pW = pW
        if pW is not None:
            self.pW = torch.from_numpy(pW).float()
    
    def set_pW0(self, pW0):
        self.pW0 = pW0
        if pW0 is not None:
            self.pW0 = torch.from_numpy(pW0).float()
    
    def l1_cost(self, x_new, x):
        cost = torch.dist(x_new, x, 1)
        return cost
    
    def pfc_cost(self, x_new, x):
        cost = torch.norm(self.feature_costs * (x_new - x), 1)
        return cost

    # FGSM attack code
    def fgsm_attack(self, img, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = img + epsilon * sign_data_grad
        # Return the perturbed image
        return perturbed_image
    
    def calc_delta_opt(self, recourse):
        """
        calculate the optimal delta using linear program
        :returns: torch tensor with optimal delta value
        """
        W = self.W
        # recourse = recourse.clone().detach()
        
        loss_fn = torch.nn.BCELoss()
        
        A_eq = np.empty((0, len(W)), float)
        b_eq = np.array([])
        
        recourse.requires_grad = True
        f_x_new = torch.nn.Sigmoid()(torch.matmul(W, recourse) + self.W0.item())
        rec_loss = loss_fn(f_x_new, self.y_target)
        gradient_rec_loss = grad(rec_loss, recourse)[0]
        
        c = list(np.array(gradient_rec_loss) * np.array([-1] * len(gradient_rec_loss)))
        bound = (-self.delta_max, self.delta_max)
        bounds = [bound] * len(gradient_rec_loss)
        
        res = linprog(c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='simplex')
        delta_opt = res.x  # the delta value that maximizes the function
        delta_rec = np.array(delta_opt[:])
        return delta_rec
    
    def get_recourse(self, x, lamb=0.01):
        torch.manual_seed(0)
        
        # returns x'
        x = torch.from_numpy(x).float()
        lamb = torch.tensor(lamb).float()
        x_new = Variable(x.clone(), requires_grad=True)
        optimizer = optim.Adam([x_new])
        # loss_fn = torch.nn.BCELoss()
        
        # Placeholders
        loss = torch.tensor(1)
        loss_diff = 1

        while loss_diff > 1e-3:
            loss_prev = loss.clone().detach()
            
            # Zero all existing gradients
            optimizer.zero_grad()
            
            # Calculate the inner loss
            # Use optimal solution from: https://adversarial-ml-tutorial.org/linear_models/
            pred = torch.matmul(self.W, x_new) + self.W0
            z = self.y_target * pred - self.delta_max * torch.norm(self.W, 2)
            Lz = torch.log(1 + torch.exp(-z))

            cost = self.l1_cost(x_new, x)
            loss = Lz + lamb * cost
            loss.backward()
            optimizer.step()
            loss_diff = torch.dist(loss_prev, loss, 2)
        return x_new.detach().numpy()
