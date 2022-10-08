# This code is a modified version of https://github.com/nimarb/pytorch_influence_functions
# adapted to work on graphs
import torch
import time
import copy
import logging
from pathlib import Path
from torch.autograd import grad
from utils import display_progress
import torch.nn.functional as F
import numpy as np

class ShadowInfluence():
    def __init__(self, model, data, device, recursion_depth=1, r_averaging=1):
        self.device = device
        self.data = data.to(self.device)
        self.model = model
        self.verbose = True
        self.recursion_depth = recursion_depth
        self.r_averaging = r_averaging

    def calc_loss(self, out, y):
        y = torch.tensor([y]).to(self.device)
        out = out.view(1, -1)
        loss = F.cross_entropy(out, y)
        return loss

    def hvp(self, y, w, v):
        """Multiply the Hessians of y and w by v.
        Uses a backprop-like approach to compute the product between the Hessian
        and another vector efficiently, which even works for large Hessians.
        Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
        which evaluates to the same values as (A + A.t) v.

        Arguments:
            y: scalar/tensor, for example the output of the loss function
            w: list of torch tensors, tensors over which the Hessian
                should be constructed
            v: list of torch tensors, same shape as w,
                will be multiplied with the Hessian

        Returns:
            return_grads: list of torch tensors, contains product of Hessian and v.

        Raises:
            ValueError: `y` and `w` have a different length."""
        if len(w) != len(v):
            raise(ValueError("w and v must have the same length."))

        # First backprop
        first_grads = grad(y, w, retain_graph=True, create_graph=True)

        # Elementwise products
        elemwise_products = 0
        for grad_elem, v_elem in zip(first_grads, v):
            elemwise_products += torch.sum(grad_elem * v_elem)

        # Second backprop
        return_grads = grad(elemwise_products, w, create_graph=True)

        return return_grads


    def grad_z(self, pos):
        """Calculates the gradient z. One grad_z should be computed for each
        training sample."""

        self.model.eval()

        y = self.model(self.data.x, self.data.edge_index, self.data.batch,
                self.data.root_n_id)

        loss = self.calc_loss(y[pos], self.data.y[pos])

        # Compute sum of gradients from model parameters to loss
        params = [p for p in self.model.parameters() if p.requires_grad]
        g = grad(loss, params, create_graph=True)

        return list(g)


    def s_test(self, pos, damp=0.1, scale=85.0, recursion_depth=50):
        """s_test can be precomputed for each test point of interest, and then
        multiplied with grad_z to get the desired value for each training point.
        Here, strochastic estimation is used to calculate s_test. s_test is the
        Inverse Hessian Vector Product.  """
        
        v = self.grad_z(pos)
        h_estimate = v.copy()

        for i in range(recursion_depth):
            # y = self.model(self.data.x, self.data.edge_index)
            y = self.model(self.data.x, self.data.edge_index, self.data.batch,
                    self.data.root_n_id)

            loss = self.calc_loss(y[0], self.data.y[0])

            params = [ p for p in self.model.parameters() if p.requires_grad ]
            hv = self.hvp(loss, params, h_estimate)

            # Recursively calculate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]

#            display_progress("Calc. s_test recursions: ", i, recursion_depth)
            
        return h_estimate


    def calc_s_test_single(self, pos, damp=0.01, scale=25, recursion_depth=5000, r=1):
        """Calculates s_test for a single test image taking into account the whole
        training dataset. s_test = invHessian * nabla(Loss(test_img, model params)) """

        s_test_vec_list = []

        for i in range(r):
            s_test_vec_list.append(self.s_test(pos, damp=damp, scale=scale,
                                        recursion_depth=recursion_depth))
#            display_progress("Averaging r-times: ", i, r)

        s_test_vec = s_test_vec_list[0]

        for i in range(1, r):
            s_test_vec += s_test_vec_list[i]

        s_test_vec = [i / r for i in s_test_vec]

        return s_test_vec

    def calc_influence_single(self, recursion_depth, r, s_test_vec=None, time_logging=False):
        train_dataset_size = len(self.data.y)
        influences = []

        for i,_ in enumerate(self.data.y):
            s_test_vec = self.calc_s_test_single(i, recursion_depth=recursion_depth, r=r)
            grad_z_vec = self.grad_z(i)

            tmp_influence = -sum(
                [
                    torch.sum(k * j).cpu().data.numpy() for k, j in zip(grad_z_vec, s_test_vec)
                ]) / train_dataset_size

            influences.append(tmp_influence)
            display_progress("Calc. influence function: ", i, train_dataset_size)

        harmful = np.argsort(influences)
        helpful = harmful[::-1]

        return influences, harmful.tolist(), helpful.tolist()

    def calculate(self):
        influence, harmful, helpful = self.calc_influence_single(
                self.recursion_depth, self.r_averaging)

        influences = {}
        infl = [x.tolist() for x in influence]
        influences['influences'] = infl
        influences['harmful'] = harmful[:500]
        influences['helpful'] = helpful[:500]

        return influences
