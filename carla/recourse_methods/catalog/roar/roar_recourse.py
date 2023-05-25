# utils
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import grad
from scipy.optimize import linprog
from tqdm import tqdm
import torch
import numpy as np


class RoarRecourse:
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

    def calc_delta_opt(self, recourse):
        """
        calculate the optimal delta using linear program
        :returns: torch tensor with optimal delta value
        """
        W = torch.cat((self.W, self.W0), 0)  # Add intercept to weights
        recourse = torch.cat((recourse, torch.ones(1)), 0)  # Add 1 to the feature vector for intercept

        loss_fn = torch.nn.BCELoss()

        A_eq = np.empty((0, len(W)), float)
        b_eq = np.array([])

        W.requires_grad = True
        f_x_new = torch.nn.Sigmoid()(torch.matmul(W, recourse))
        w_loss = loss_fn(f_x_new, self.y_target)
        gradient_w_loss = grad(w_loss, W)[0]

        c = list(np.array(gradient_w_loss) * np.array([-1] * len(gradient_w_loss)))
        bound = (-self.delta_max, self.delta_max)
        bounds = [bound] * len(gradient_w_loss)

        res = linprog(c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='simplex')
        delta_opt = res.x  # the delta value that maximizes the function
        delta_W, delta_W0 = np.array(delta_opt[:-1]), np.array([delta_opt[-1]])
        return delta_W, delta_W0

    def get_recourse(self, x, lamb=0.25):
        torch.manual_seed(0)

        # returns x'
        x = torch.from_numpy(x).float()
        lamb = torch.tensor(lamb).float()

        x_new = Variable(x.clone(), requires_grad=True)
        optimizer = optim.Adam([x_new])

        loss_fn = torch.nn.BCELoss()

        # Placeholders
        loss = torch.tensor(1)
        loss_diff = 1

        while loss_diff > 1e-4:
            loss_prev = loss.clone().detach()
            
            delta_W, delta_W0 = self.calc_delta_opt(x_new)
            delta_W, delta_W0 = torch.from_numpy(delta_W).float(), torch.from_numpy(delta_W0).float()

            optimizer.zero_grad()
            if self.pW is not None:
                dec_fn = torch.matmul(self.W + delta_W, x_new) + self.W0
                f_x_new = torch.nn.Sigmoid()(torch.matmul(self.pW, dec_fn.unsqueeze(0)) + self.pW0)[0]
            else:
                f_x_new = torch.nn.Sigmoid()(torch.matmul(self.W + delta_W, x_new) + self.W0 + delta_W0)[0]

            if self.feature_costs is not None:
                cost = self.pfc_cost(x_new, x)
            else:
                cost = self.l1_cost(x_new, x)

            loss = loss_fn(f_x_new, self.y_target) + lamb * cost
            loss.backward()
            optimizer.step()

            loss_diff = torch.dist(loss_prev, loss, 2)
        return x_new.detach().numpy()  # np.concatenate((delta_W.detach().numpy(), delta_W0.detach().numpy()))

    # Heuristic for picking hyperparam lambda
    def choose_lambda(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        lambdas = [0.1, 0.25, 0.5, 0.75]

        v_old = 0
        for i, lamb in enumerate(lambdas):
            print("Testing lambda:%f" % lamb)
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train,
                                                               x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    r, _ = self.get_recourse(x, lamb)

                    self.set_W(None)
                    self.set_W0(None)
                else:
                    r, _ = self.get_recourse(x, lamb)
                recourses.append(r)

            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            if v >= v_old:
                v_old = v
            else:
                li = max(0, i - 1)
                return lambdas[li]

        return lamb

    def choose_delta(self, recourse_needed_X, predict_fn, X_train=None,
                     predict_proba_fn=None, lamb=0.1):
        deltas = [0.1, 0.25, 0.5, 0.75]

        v_old = 0
        for i, d in enumerate(deltas):
            print("Testing delta:%f" % d)
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train, x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    self.delta_max = d

                    r, _ = self.get_recourse(x, lamb)

                    self.set_W(None)
                    self.set_W0(None)
                else:
                    self.delta_max = d
                    r, _ = self.get_recourse(x, lamb)
                recourses.append(r)

            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            if v >= v_old:
                v_old = v
            else:
                di = max(0, i - 1)
                return deltas[di]

        return d

    def choose_params(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        def get_validity(d, l, recourse_needed_X, predict_proba_fn):
            print("Testing delta %f, lambda %f" % (d, l))
            recourses = []
            for xi, x in tqdm(enumerate(recourse_needed_X)):

                self.delta_max = d

                # Call lime if nonlinear
                if self.W0 is None and self.W is None:
                    # set seed for lime
                    np.random.seed(xi)
                    coefficients, intercept = lime_explanation(predict_proba_fn,
                                                               X_train, x)
                    coefficients, intercept = np.round_(coefficients, 4), np.round_(intercept, 4)
                    self.set_W(coefficients)
                    self.set_W0(intercept)

                    r, _ = self.get_recourse(x, l)

                    self.set_W(None)
                    self.set_W0(None)

                else:
                    r, _ = self.get_recourse(x, l)

                recourses.append(r)
            v = recourse_validity(predict_fn, recourses, target=self.y_target.numpy())
            return v

        deltas = [0.01, 0.25, 0.5, 0.75]
        lambdas = [0.1, 0.25, 0.5, 0.75]

        m1_validity = np.zeros((4, 4))
        costs = np.zeros((4, 4))

        delta = None
        lamb = None
        for li, l in enumerate(lambdas):
            if li == 0:
                for di, d in enumerate(deltas):
                    d = deltas[di]
                    v = get_validity(d, l, recourse_needed_X, predict_proba_fn)
                    if v < m1_validity[max(0, di - 1)][li]:
                        di = max(0, di - 1)
                        delta = deltas[di]
                        break
                    m1_validity[di][li] = v

                if delta is None:
                    delta = d
            else:
                v = get_validity(delta, l, recourse_needed_X, predict_proba_fn)
                m1_validity[di][li] = v
                if v < m1_validity[di][max(0, li - 1)]:
                    li = max(0, li - 1)
                    lamb = lambdas[li]
                    break
        if lamb is None:
            lamb = l

        print(m1_validity)
        return delta, lamb
