from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
from typing import Dict, Any, Callable
import time
from jax.flatten_util import ravel_pytree
from itertools import cycle

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
import flax
import optax
import jaxopt

from . import deepxde as dde

from .ntk import NTKHelper
from .al import PointSelector, AL_CONSTRUCTOR
from .icbc_patch import generate_residue, get_corresponding_y
from .utils import to_cpu

from .multiadam_jax import multiadam
from .multiadam_jax_alt import multiadam_optimizer as alt_multiadam
import gc


from torch.utils.tensorboard import SummaryWriter



class ModifiedTrainLoop:
    
    def __init__(self, model: dde.Model, inverse_problem: bool = False, log_dir: str = None, 
                 point_selector_method: str = None, point_selector_args: Dict = dict(), 
                 optim_method: str = 'adam', optim_lr: float = 1e-3, optim_args: Dict = dict(),
                 train_steps: int = 10000, al_every: int = 10000, snapshot_every: int = 1000,
                 select_anchors_every: int = 10000, anchor_budget: int = 0, mem_pts_total_budget: int = 1000,
                 anc_point_filter: Callable = None, anc_measurable_idx: int = None,
                 loss_w_bcs: float = 1.0, loss_w_pde: float = 1.0, loss_w_anc: float = 1.0, autoscale_loss_w_bcs: bool = False, autoscale_first: bool = False,
                 save_grads: bool = True, al_loss_weights: bool = False,
                 random_points_for_weights: bool = False, ntk_ratio_threshold: float = None, check_budget: int = 200, tensorboard_plots = False,
                 sample_each_round: bool = False, lra_loss_w_bcs: bool = False
                 ):
        #for recording gradient weight distribution
        # self.pde_grads = None  # Changed to dict
        # self.bc_grads = None   # Changed to dict

        self.autoscale_first = autoscale_first
        self.random_points_for_weights = random_points_for_weights

        self.save_grads = save_grads
        self.model = model
        self.inverse_problem = inverse_problem
        self.log_dir = log_dir
        self.net = model.net
        self.data = model.data
        self.x_test = model.data.test_x
        self.y_test = model.data.test_y
        
        self.point_selector_method = point_selector_method
        self.point_selector_args = point_selector_args
        self.anchor_budget = anchor_budget
        self.select_anchors = (anchor_budget > 0)
        self.mem_pts_total_budget = mem_pts_total_budget
        self.anc_point_filter = anc_point_filter
        self.anc_measurable_idx = anc_measurable_idx

        self.autoscale_first = autoscale_first
        self.random_points_for_weights = random_points_for_weights
        self.sample_each_round = sample_each_round
        self.lra_loss_w_bcs = lra_loss_w_bcs
        
        self.optim_method = optim_method
        self.optim_lr = optim_lr
        self.optim_args = optim_args
        
        self.loss_w_bcs = loss_w_bcs if hasattr(loss_w_bcs, "__len__") else [loss_w_bcs for _ in self.data.bcs]
        self.loss_w_pde = loss_w_pde
        self.loss_w_anc = loss_w_anc

        self.al_loss_weights = al_loss_weights
        if self.al_loss_weights:
            self.al_loss_w_bcs = loss_w_bcs
            self.al_loss_w_pde = loss_w_pde
            self.al_loss_w_anc = loss_w_anc

        self.autoscale_loss_w_bcs = autoscale_loss_w_bcs
        self.ntk_ratio_threshold = ntk_ratio_threshold

        self.train_steps = train_steps
        self.al_every = al_every
        self.snapshot_every = snapshot_every
        self.select_anchors_every = select_anchors_every
        self._last_al_step = 0
                
        self.pde_residue_fn = self._generate_pde_res()
        self.icbc_error_fns = self._generate_icbc_err()
        self.soln_error_fn = self._generate_function_error()
        
        check_budget = 200
        d = {
            'res': jnp.array(self.data.geom.random_points(check_budget, random='pseudo')),
            'bcs': [],
        }
        for i, bc in enumerate(self.data.bcs):
            if isinstance(bc, dde.icbc.IC):
                # print(f'BCS_{i+1} is IC type, adding in test_pts_init')
                d['bcs'].append(jnp.array(self.data.geom.random_initial_points(check_budget)))
            elif isinstance(bc, dde.icbc.PointSetBC):
                # print(f'BCS_{i+1} is PointSet type, adding in {self.num_candidates_init} candidate points from BC class')
                pts_subset_idx = np.random.choice(
                    a=bc.points.shape[0],
                    size=min(check_budget, bc.points.shape[0]),
                    replace=False
                )
                d['bcs'].append(jnp.array(bc.points[jnp.array(pts_subset_idx)]))
            else:
                # print(f'BCS_{i+1} is BC type, adding in test_pts_bc')
                d['bcs'].append(jnp.array(self.data.geom.random_boundary_points(check_budget)))
        pts_subset_idx = np.random.choice(
            a=self.x_test.shape[0],
            size=min(check_budget, self.x_test.shape[0]),
            replace=False
        )
        d['anc'] = self.x_test[jnp.array(pts_subset_idx)]
        self._ntk_check_pts = d
        self._ntk_fn = NTKHelper(self.model, inverse_problem=self.inverse_problem)

        # for debugging
        self.opt_state = None
        self.tensorboard_plots = tensorboard_plots

            
        self.reset_training()
        
    def reset_training(self):
        
        self.current_train_step = 0
        self.current_params = None
        self.al = None
        self.current_samples = None
        self.loss_fn = None
        self.loss_fn_grad = None

        self.loss_steps = []
        self.loss_train = []
        self.loss_pde = []
        self.loss_bc = []
        self.test_res = []
        self.test_err = []
        self.snapshot_data = {
            None: {
                'x_test': to_cpu(self.x_test),
                'y_test': to_cpu(self.y_test),
                'al_method': self.point_selector_method,
                'al_args': self.point_selector_args,
                'check_pts': to_cpu(self._ntk_check_pts),
            }
        }
        self.al_data_round = dict()
        
    def _generate_function_error(self, idx=None):
        
        if idx is None:
            def function_error(nn_params, xs, ys):
                f_ = lambda xs: self.net.apply(nn_params, xs, training=True)
                ypred = f_(xs)
                assert ys.shape == ypred.shape
                return jnp.abs(ypred - ys)
            
        elif isinstance(idx, int):
            def function_error(nn_params, xs, ys):
                f_ = lambda xs: self.net.apply(nn_params, xs, training=True)
                ypred = f_(xs)
                assert ys.shape == ypred.shape
                assert len(ypred.shape) == 2
                return jnp.abs(ypred[:, idx:idx+1] - ys[:, idx:idx+1])
            
        else:
            def function_error(nn_params, xs, ys):
                f_ = lambda xs: self.net.apply(nn_params, xs, training=True)
                ypred = f_(xs)
                assert ys.shape == ypred.shape
                assert len(ypred.shape) == 2
                return jnp.abs(ypred[:, idx] - ys[:, idx])
        
        return function_error
    
    def _generate_pde_res(self):
        def _pde_residue_fn(params, xs):
            if self.inverse_problem:
                nn_params, ext_params = params 
                f_ = lambda xs: self.net.apply(nn_params, xs, training=True)
                return jnp.abs(self.data.pde(xs, (f_(xs), f_), ext_params)[0])
            else:
                # print(f'xs is {xs}')
                f_ = lambda xs: self.net.apply(params[0], xs, training=True)
                # print(f'f_ is {f_}')
                out = self.data.pde(xs, (f_(xs), f_))[0]
                return jnp.abs(out)
                # return jnp.abs(self.data.pde(xs, f_(xs))[0])
        return _pde_residue_fn
    
    def _generate_icbc_err(self):

        # def gen_loss_fn(i):
        #     return lambda params, xs: self.data.bcs[i].error(
        #         X=xs, 
        #         inputs=xs, 
        #         outputs=(
        #             self.net.apply(params, xs, training=True),
        #             lambda xs: self.net.apply(params, xs, training=True)
        #         ), 
        #         beg=0, 
        #         end=xs.shape[0], 
        #         aux_var=None
        #     )
        
        # return [gen_loss_fn(i) for i in range(len(self.data.bcs))]
        return [generate_residue(bc, net_apply=self.net.apply) for bc in self.data.bcs]
    
    def _do_active_learning(self, do_anchor: bool = False):

        # if self.autoscale_loss_w_bcs and self.autoscale_first:
        #     print('autoscaling')
        #     if self.random_points_for_weights:
        #         d = self._get_random_samples()
        #     else:
        #         d = self.current_samples
            
        #     ntk_pde = self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['res'], code=-1))
        #     eigvals_pde = jnp.linalg.eigvalsh(ntk_pde)
        #     tr_pde = jnp.sum(eigvals_pde)
        #     print(f'PDE Cl top eigvals = {eigvals_pde[-min(5, eigvals_pde.shape[0]):]}')
        #     print(f'PDE Cl trace = {tr_pde}')
            
            
        #     ntk_bcs_list = [self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['bcs'][i], code=i)) for i in range(len(d['bcs']))]
        #     eigvals_bcs = [jnp.linalg.eigvalsh(ntkbc) for ntkbc in ntk_bcs_list]
        #     tr_bcs = [jnp.sum(eigbc) for eigbc in eigvals_bcs]
        #     for i, bc in enumerate(eigvals_bcs):
        #         print(f'BC {i} Cl top eigvals = {bc[-min(5, bc.shape[0]):]}')
        #         print(f'BC {i} Cl trace = {tr_bcs[i]}')
            
        #     if 'anc' in d.keys():
        #         ntk_anc = self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['anc'], code=-2))
        #         eigvals_anc = jnp.linalg.eigvalsh(ntk_anc)
        #         print(f'Exp top eigvals = {eigvals_pde[-min(5, eigvals_anc.shape[0]):]}')
        #         tr_anc = jnp.sum(eigvals_anc)
        #         print(f'Exp trace = {tr_anc}')
        #     else:
        #         tr_anc = 0.
                
        #     total = tr_pde + tr_anc + sum(tr_bcs)
        #     self.loss_w_pde = total / tr_pde if tr_pde > 0. else 0.
        #     self.loss_w_bcs = [total/bc if bc > 0. else 0. for bc in tr_bcs]
        #     self.loss_w_anc = total / tr_anc if tr_anc > 0. else 0.
            
        #     print('loss_w_pde =', self.loss_w_pde)
        #     print('loss_w_bcs =', self.loss_w_bcs)
        #     print('loss_w_anc =', self.loss_w_anc)

        #     del d, ntk_pde, eigvals_pde, tr_pde, ntk_bcs_list, eigvals_bcs, tr_bcs, ntk_anc, eigvals_anc, tr_anc, total
        
        self._last_al_step = self.current_train_step
        
        if self.point_selector_method is None:
            self.point_selector_method = 'random'
            
        if (('eig' in self.point_selector_method) or ('residue' in self.point_selector_method)) and do_anchor:
            
            point_sel_args_d = self.point_selector_args.copy()
            
            if self.optim_method == 'lbfgs':
                steps = 5
            else:
                steps = 100
            print(f'Training for {steps} steps to get pseudo-values for anchor')
            
            #MIGHT NEED CHANGING
            if self.loss_fn is None:
                def _loss(params):
                    loss = jnp.mean(self.pde_residue_fn(params, self._ntk_check_pts['res']) ** 2)
                    for i in range(len(self._ntk_check_pts['bcs'])):
                        loss += jnp.mean(self.icbc_error_fns[i](params[0], self._ntk_check_pts['bcs'][i]) ** 2)
                    return loss
                self.loss_fn = _loss
                self.loss_fn_grad = jax.value_and_grad(_loss) 
                if self.optim_method == 'multiadam':
                    raise NotImplementedError('Did not implement this case yet for the new multiadam')
                print('Pseudo-training using new function')
                print(f'Mock train set has {self._ntk_check_pts["res"].shape[0]} res pts. and {[bcs.shape[0] for bcs in self._ntk_check_pts["bcs"]]} ICBC pts.')
            else:
                print('Pseudo-training using existing train function')
            
            # do GD a few more steps to get pseudo data
            if self.optim_method == 'multiadam':
                solver = self._generate_solver(value_and_grad=self.loss_fn_grad)
            else:
                solver = self._generate_solver(value_and_grad=self.loss_fn_grad)

            target_fn_param = self.model.params
            if self.opt_state is None:
                if self.optim_method == 'multiadam':
                    opt_state = solver.init(target_fn_param)
                else:
                    opt_state = solver.init_state(target_fn_param)
            else:
                opt_state = self.opt_state
            for r_inside in range(steps):
                if self.optim_method == 'multiadam':
                    grads = [jax.grad(self.pde_loss_fn)(target_fn_param), jax.grad(self.bc_loss_fn)(params)]
                    updates, opt_state = solver.update(grads, opt_state, params)
                    target_fn_param = optax.apply_updates(target_fn_param, updates)
                else:
                    target_fn_param, opt_state = solver.update(target_fn_param, opt_state)
                
            point_sel_args_d['target_fn_param'] = target_fn_param
            
            print('Pseudo-training done.')
            
            self.loss_fn = None
            self.loss_fn_grad = None
            
        else:
            point_sel_args_d = self.point_selector_args
            
        if self.anc_measurable_idx is None:
            anc_idx = 0
        elif isinstance(self.anc_measurable_idx, int):
            anc_idx = self.anc_measurable_idx
        else:
            anc_idx = self.anc_measurable_idx[0]


        self.al = AL_CONSTRUCTOR[self.point_selector_method](
            model=self.model,
            inverse_problem=self.inverse_problem,
            loss_w_bcs=self.al_loss_w_bcs if self.al_loss_weights else self.loss_w_bcs,
            loss_w_pde=self.al_loss_w_pde if self.al_loss_weights else self.loss_w_pde,
            loss_w_anc=self.loss_w_anc if self.al_loss_weights else self.loss_w_anc,
            optim_lr=self.optim_lr,
            current_samples=self.current_samples,
            mem_pts_total_budget=self.mem_pts_total_budget,
            anchor_budget=self.anchor_budget if do_anchor else 0,
            anc_point_filter=self.anc_point_filter,
            anc_idx=anc_idx,
            **point_sel_args_d
        )
        self.current_samples, self._sample_intermediates = self.al.generate_samples()
        # if self.autoscale_loss_w_bcs and ('new_loss_w_bcs' in self._sample_intermediates.keys()):
        #     old_lwbcs = self.loss_w_bcs
        #     self.loss_w_bcs = self._sample_intermediates['new_loss_w_bcs']
        #     print(f'loss_w_bcs changed from {old_lwbcs} to {self.loss_w_bcs} instead')
        #     old_lwpde = self.loss_w_pde
        #     self.loss_w_pde = self._sample_intermediates['new_loss_w_pde']
        #     print(f'loss_w_pde changed from {old_lwpde} to {self.loss_w_pde} instead')
        #     if self.select_anchors:
        #         old_lwanc = self.loss_w_anc
        #         self.loss_w_anc = self._sample_intermediates['new_loss_w_anc']
        #         print(f'loss_w_anc changed from {old_lwanc} to {self.loss_w_anc} instead')

        # self.al = AL_CONSTRUCTOR[self.point_selector_method](model=self.model, **self.point_selector_args, loss_w_bcs = self.loss_w_bcs, current_samples = self.current_samples)
        # logging_dict = self.al.generate_samples()
        # self.current_samples = logging_dict['chosen_pts']
        
        if 'anc' in self.current_samples.keys():
            anc_x = self.current_samples['anc']
            anc_y = get_corresponding_y(anc_x, self.data.test_x, self.data.test_y)
            
        self.al_data_round[self.current_train_step] = self.current_samples
        
        soln_train_loss = self._generate_function_error(idx=self.anc_measurable_idx)
        
        # N = float(
        #     self.current_samples['res'].shape[0] +
        #     sum([xs.shape[0] for xs in self.current_samples['bcs']]) +
        #     (self.current_samples['anc'].shape[0] if 'anc' in self.current_samples.keys() else 0)
        # )
        
        # compute kernel
        # if self.autoscale_loss_w_bcs and not self.autoscale_first:
        #     print('autoscaling')
        #     if self.random_points_for_weights:
        #         d = self._get_random_samples()
        #     else:
        #         d = self.current_samples
            
        #     ntk_pde = self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['res'], code=-1))
        #     eigvals_pde = jnp.linalg.eigvalsh(ntk_pde)
        #     tr_pde = jnp.sum(eigvals_pde)
        #     print(f'PDE Cl top eigvals = {eigvals_pde[-min(5, eigvals_pde.shape[0]):]}')
        #     print(f'PDE Cl trace = {tr_pde}')
            
            
        #     ntk_bcs_list = [self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['bcs'][i], code=i)) for i in range(len(d['bcs']))]
        #     eigvals_bcs = [jnp.linalg.eigvalsh(ntkbc) for ntkbc in ntk_bcs_list]
        #     tr_bcs = [jnp.sum(eigbc) for eigbc in eigvals_bcs]
        #     for i, bc in enumerate(eigvals_bcs):
        #         print(f'BC {i} Cl top eigvals = {bc[-min(5, bc.shape[0]):]}')
        #         print(f'BC {i} Cl trace = {tr_bcs[i]}')
            
        #     if 'anc' in d.keys():
        #         ntk_anc = self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['anc'], code=-2))
        #         eigvals_anc = jnp.linalg.eigvalsh(ntk_anc)
        #         print(f'Exp top eigvals = {eigvals_pde[-min(5, eigvals_anc.shape[0]):]}')
        #         tr_anc = jnp.sum(eigvals_anc)
        #         print(f'Exp trace = {tr_anc}')
        #         del ntk_anc, ntk_anc
        #     else:
        #         tr_anc = 0.
                
        #     total = tr_pde + tr_anc + sum(tr_bcs)
        #     self.loss_w_pde = total / tr_pde if tr_pde > 0. else 0.
        #     self.loss_w_bcs = [total/bc if bc > 0. else 0. for bc in tr_bcs]
        #     self.loss_w_anc = total / tr_anc if tr_anc > 0. else 0.
            
        #     print('loss_w_pde =', self.loss_w_pde)
        #     print('loss_w_bcs =', self.loss_w_bcs)
        #     print('loss_w_anc =', self.loss_w_anc)

        #     del d, ntk_pde, eigvals_pde, tr_pde, ntk_bcs_list, eigvals_bcs, tr_bcs, tr_anc, total
        
        # if not self.optim_method == 'multiadam':
        #     def new_loss(params):
            
        #         # Calculate loss
        #         # TODO to add in the loss of the active learning samples and also include weights
        #         loss = self.loss_w_pde * jnp.mean(self.pde_residue_fn(params, self.current_samples['res']) ** 2)
        #         # Adjusting to use the new loss_w_bcs array
        #         for i in range(len(self.current_samples['bcs'])):
        #             loss += self.loss_w_bcs[i] * jnp.mean(self.icbc_error_fns[i](params[0], self.current_samples['bcs'][i]) ** 2)
        #         if 'anc' in self.current_samples.keys():
        #             loss += self.loss_w_anc * jnp.mean(soln_train_loss(params[0], anc_x, anc_y) ** 2)
        #         return loss
        #     self.loss_fn = jax.jit(new_loss)
        #     self.loss_fn_grad = jax.value_and_grad(self.loss_fn) 
        self.update_functions()

    def update_functions(self):
        def new_pde_loss(params, xs=None):
            # Calculate losses separately
            pde_loss = self.loss_w_pde * jnp.mean(self.pde_residue_fn(params, self.current_samples['res']) ** 2)
            # Adjusting to use the new loss_w_bcs array
            return pde_loss
        def new_bc_loss(params, xs=None):
            bc_loss = 0
            for i in range(len(self.current_samples['bcs'])):
                bc_loss += self.loss_w_bcs[i] * jnp.mean(self.icbc_error_fns[i](params[0], self.current_samples['bcs'][i]) ** 2)
            if 'anc' in self.current_samples.keys():
                raise NotImplementedError("Multiadam optimization with anchor points not implemented") 
            return bc_loss
        def unweighted_bc_loss(params, xs=None):
            bc_loss = 0
            for i in range(len(self.current_samples['bcs'])):
                bc_loss += jnp.mean(self.icbc_error_fns[i](params[0], self.current_samples['bcs'][i]) ** 2)
            if 'anc' in self.current_samples.keys():
                raise NotImplementedError("Multiadam optimization with anchor points not implemented") 
            return bc_loss
        def new_loss(params):
            # Calculate losses separately
            pde_loss = new_pde_loss(params)
            bc_loss = new_bc_loss(params)
            loss = pde_loss + bc_loss
            return loss #, {'pde_loss': pde_loss, 'bc_loss': bc_loss}
        
        self.loss_fn = jax.jit(new_loss)
        self.loss_fn_grad = jax.value_and_grad(self.loss_fn) 
        self.pde_loss_fn = jax.jit(new_pde_loss)
        self.pde_loss_fn_grad = jax.value_and_grad(self.pde_loss_fn) 
        self.bc_loss_fn = jax.jit(new_bc_loss)
        self.bc_loss_fn_grad = jax.value_and_grad(self.bc_loss_fn) 
        self.unw_bc_loss_fn = jax.jit(unweighted_bc_loss)
        self.unw_bc_loss_fn_grad = jax.value_and_grad(self.unw_bc_loss_fn) 
        
    def _generate_solver(self, value_and_grad):
        # # TODO to add in other optimizers like L-BFGS
        # return {
        #     'adam': optax.adam
        # }[self.optim_method]
        if self.optim_method == 'multiadam':
            # Configure MultiAdam with 2 groups - PDE and boundary
            # opt = multiadam(
            #     learning_rate=self.optim_lr,
            #     group_weights=[0.5, 0.5],  # Weight PDE vs boundary
            #     loss_group_idx=[0, 1], # 0=PDE, 1=BC
            #     aux_loss=aux_loss,
            #     **self.optim_args
            # )
            # solver = jaxopt.OptaxSolver(opt=opt, fun=value_and_grad, value_and_grad=True)
            solver = alt_multiadam(
                lr=self.optim_lr,
                group_weights=[0.5, 0.5],  # Weight PDE vs boundary
                **self.optim_args
            )

        elif self.optim_method == 'adam':
            opt = optax.adam(learning_rate=self.optim_lr, **self.optim_args)
            solver = jaxopt.OptaxSolver(opt=opt, fun=value_and_grad, value_and_grad=True)
            
        elif self.optim_method == 'lbfgs':
            solver = jaxopt.LBFGS(fun=value_and_grad, value_and_grad=True, jit=True, **self.optim_args)
            
        else:
            raise ValueError(f'Invalid optim_method: {self.optim_method}')
        
        return solver
        
    def _record(self, writer: SummaryWriter = None, al_step = False):

        if not self.point_selector_method.startswith('eig'):
            skip = True
        else:
            skip = False
        
        params = self.current_params
        self.loss_steps.append(self.current_train_step)
        self.loss_train.append(self.loss_fn(params) if self.loss_fn else jnp.nan)
        self.loss_pde.append(self.pde_loss_fn(params)/self.loss_w_pde if self.loss_fn else jnp.nan)
        self.loss_bc.append(self.unw_bc_loss_fn(params) if self.loss_fn else jnp.nan)
        
        test_pred_list = self.net.apply(params[0], self.x_test, training=False)
        
        fn2 = lambda x_: self.pde_residue_fn(params, x_.reshape(1, -1))[0]  # version for single dims
        test_res_list = jax.vmap(fn2)(self.x_test)
        # test_res_list = self.pde_residue_fn(params, self.x_test)
        self.test_res.append(jnp.mean(test_res_list ** 2))
        
        soln_err_list = self.soln_error_fn(params[0], self.x_test, self.y_test)
        self.test_err.append(jnp.nanmean(soln_err_list ** 2))
        
        # Compute separate jacs
        if skip:
            self.snapshot_data[self.current_train_step] = {
            'al_intermediate': self._sample_intermediates if al_step else None,
            'samples': to_cpu(self.current_samples),
            'params': to_cpu(self.current_params),
            'loss_train': to_cpu(self.loss_train[-1]),
            'loss_pde': to_cpu(self.loss_pde[-1]),
            'loss_bc': to_cpu(self.loss_bc[-1]),
            'residue_test_mean': to_cpu(self.test_res[-1]),
            'error_test_mean': to_cpu(self.test_err[-1]),
            'pred_test': to_cpu(test_pred_list),
            'residue_test': to_cpu(test_res_list),
            'error_test': to_cpu(soln_err_list),
            'loss_w_bcs': self.loss_w_bcs,
            'loss_w_pde': self.loss_w_pde,
            'loss_w_anc': self.loss_w_anc
            }   
        else:
            d = self._ntk_check_pts
            jac_pde = self._ntk_fn.get_jac(d['res'], code=-1)
            jac_bcs_list = [self._ntk_fn.get_jac(d['bcs'][i], code=i) for i in range(len(d['bcs']))] # separate eigvals for each bc component
            jac_anc = self._ntk_fn.get_jac(d['anc'], code=-2)
            jacs_sep = [jac_pde] + jac_bcs_list + [jac_anc]
            jacs = {k: jnp.concatenate([jc[k] for jc in jacs_sep], axis=0) for k in jac_pde.keys()}
            K_check_pts = self._ntk_fn.get_ntk(jac1=jacs, jac2=jacs)
                
            self.snapshot_data[self.current_train_step] = {
                'al_intermediate': self._sample_intermediates if al_step else None,
                'samples': to_cpu(self.current_samples),
                'params': to_cpu(self.current_params),
                'loss_train': to_cpu(self.loss_train[-1]),
                'loss_pde': to_cpu(self.loss_pde[-1]),
                'loss_bc': to_cpu(self.loss_bc[-1]),
                'residue_test_mean': to_cpu(self.test_res[-1]),
                'error_test_mean': to_cpu(self.test_err[-1]),
                'pred_test': to_cpu(test_pred_list),
                'residue_test': to_cpu(test_res_list),
                'error_test': to_cpu(soln_err_list),
                'K_check_pts': to_cpu(K_check_pts),
                'loss_w_bcs': self.loss_w_bcs,
                'loss_w_pde': self.loss_w_pde,
                'loss_w_anc': self.loss_w_anc
            }

        # Force garbage collection after recording
        gc.collect()

        print(f'Step {self.loss_steps[-1]: 6d} : train_loss = {self.loss_train[-1]:.8f}, '
              f'loss_pde = {self.loss_pde[-1]:.8f}, loss_bc = {self.loss_bc[-1]:.8f}, '
              f'test_res = {self.test_res[-1]:.8f}, test_err = {self.test_err[-1]:.8f}'
              + (f', ext_params = {[float(f"{float(x):.4f}") for x in self.current_params[1]]}' if self.inverse_problem else ''))

        # -------- Plotting functions for tensorboard ----------
        # Eigenbasis plots differentiating between residual and predict

        if not skip:
            def func_kernel_pred(train_loop,x_test,step_idx,idx=-1,code = -2, use_const_res=True):
                
                jacs_x_test = self._ntk_fn.get_jac(x_test, code=code)
                # jacs_train = self._ntk_fn.get_jac(train_loop._sample_intermediates['old_points'])
                jacs_train = train_loop._sample_intermediates['jac_train']
                K_train_x_test = self._ntk_fn.get_ntk(jac1=jacs_train, jac2=jacs_x_test)

                # K_train_x_test = train_loop._sample_intermediates['K_train_test']
                eigvects = train_loop._sample_intermediates['eigvects']
                eigvals = train_loop._sample_intermediates['eigvals']
                residual_old = train_loop._sample_intermediates['residual_old']
                
                # # Selecting specific eigenvectors
                if use_const_res == True:
                    # output1 = jnp.ones(residual_old.shape) @eigvects[:,idx]
                    # Correct approach is to not use use residuals. 
                    output1 = 1
                else:
                    output1 = residual_old@eigvects[:,idx]
                output = output1 * 1/eigvals[idx]* eigvects[:,idx].T @ K_train_x_test
                return output

            def plot_eigenbasis(train_loop,step_idx=0, num_plots=10, plots_per_level = 5, use_const_res=True):
                # Plotting eigenbasis

                # Plotting grid settings
                res = 30
                xs = train_loop.x_test
                xi, yi = [np.linspace(np.min(xs[:,i]), np.max(xs[:,i]), res) for i in range(2)]
                grid = np.meshgrid(xi, yi)
                pool_pts = jnp.array(grid).reshape(2, -1).T
                x_test = pool_pts

                fig, axs = plt.subplots(2, plots_per_level, figsize=(30, 15))
                for i in range(plots_per_level):
                    idx = -1-i
                    # First level is for predict, second is for residual
                    x_train = train_loop.snapshot_data[step_idx]['samples']

                    samples = x_train
                    for level in range(2):
                        axs[level,i%plots_per_level].plot(samples['res'][:, 0], samples['res'][:, 1], 'o')
                        for bc_pts in samples['bcs']:
                            axs[level,i%plots_per_level].plot(bc_pts[:, 0], bc_pts[:, 1], '^')
                        if level == 0:
                            axs[level,i% plots_per_level].set_title(f'Top {-idx} Eigenvector for predict, step {step_idx}, Eigval = {train_loop.snapshot_data[step_idx]["al_intermediate"]["eigvals"][idx]:.2f}', wrap = True)
                            T = func_kernel_pred(train_loop,x_test = x_test,step_idx=step_idx, idx=-idx, use_const_res=use_const_res, code = -2).reshape(res, res)
                        else:
                            axs[level,i% plots_per_level].set_title(f'Top {-idx} Eigenvector for residual, step {step_idx}, Eigval = {train_loop.snapshot_data[step_idx]["al_intermediate"]["eigvals"][idx]:.2f}', wrap=True)
                            T = func_kernel_pred(train_loop,x_test = x_test,step_idx=step_idx, idx=-idx, use_const_res=use_const_res, code = -1).reshape(res, res)

                        cb = axs[level,i%plots_per_level].pcolormesh(*grid, T, cmap='RdBu_r')

                        fig.colorbar(cb, ax=axs[level,i%plots_per_level])
                return fig
        
            def gen_eigenbasis(step_idx=0, eigval_count=10):
                res = 30
                xs = self.x_test
                xi, yi = [np.linspace(np.min(xs[:,i]), np.max(xs[:,i]), res) for i in range(2)]
                grid = np.meshgrid(xi, yi)
                pool_pts = jnp.array(grid).reshape(2, -1).T
                pred_eigvects = [func_kernel_pred(self, x_test=pool_pts, step_idx=step_idx, idx=i, use_const_res=True, code=-2).reshape(res, res)
                                for i in range(eigval_count)]
                res_eigvects = [func_kernel_pred(self, x_test=pool_pts, step_idx=step_idx, idx=i, use_const_res=True, code=-1).reshape(res, res)
                                for i in range(eigval_count)]
                return pool_pts, pred_eigvects, res_eigvects
        
            if al_step and ('jac_train' in self._sample_intermediates.keys()) and (self.data.test_x.shape[1] == 2):
                pool_pts, pred_eigvects, res_eigvects = gen_eigenbasis(step_idx=self.current_train_step)
            else:
                pool_pts, pred_eigvects, res_eigvects = None, None, None
            self.snapshot_data[self.current_train_step]['eig_pool_pts'] = to_cpu(pool_pts)
            self.snapshot_data[self.current_train_step]['pred_eigvects'] = to_cpu(pred_eigvects)
            self.snapshot_data[self.current_train_step]['res_eigvects'] = to_cpu(res_eigvects)

        def _plot_grid_timesteps(step_idxs, xs, zs_arrs, train_loop, res=100, plot_training_data=True):
            xi, yi = [np.linspace(np.min(xs[:,i]), np.max(xs[:,i]), res) for i in range(2)]
            grid = np.meshgrid(xi, yi)
            fig, axs = plt.subplots(nrows=1, ncols=len(step_idxs), figsize=(plt.rcParams['figure.figsize'][0] * (len(step_idxs) + 0.25), plt.rcParams['figure.figsize'][0]))
            if len(step_idxs) == 1:
                axs = [axs]
            levels = np.linspace(np.min(zs_arrs), np.max(zs_arrs), num=2*res)
            
            if plot_training_data:
                for ax, step_idx in zip(axs, step_idxs):
                    train_loop.plot_training_data(step_idx=step_idx, ax=ax)
                
            triang = tri.Triangulation(xs[:,0], xs[:,1])
            for ax, step_idx, z in zip(axs, step_idxs, zs_arrs):
                interpolator = tri.LinearTriInterpolator(triang, z.flatten())
                Xi, Yi = np.meshgrid(xi, yi)
                zi = interpolator(Xi, Yi)
                cb = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
                ax.set_title(f'Step {step_idx}')
                
            fig.colorbar(cb, ax=list(axs))
            return fig

        def plot_residue_loss(train_loop, step_idxs, res=100, plot_training_data=True):
            zs_arrs = [np.abs(train_loop.pde_residue(xs=train_loop.x_test, step_idx=s)) for s in step_idxs]
            return _plot_grid_timesteps(step_idxs=step_idxs, xs=train_loop.x_test, zs_arrs=zs_arrs, train_loop=train_loop, res=res, plot_training_data=plot_training_data)
        
        def plot_prediction(train_loop, step_idxs=None, res=100, plot_training_data=True):
            has_step_idxs = ((step_idxs is not None) and (len(step_idxs) > 0))
            if not has_step_idxs:
                step_idxs = []
            
            xs = train_loop.x_test
            z_actual = train_loop.y_test[:,0].flatten()
            if has_step_idxs:
                zs_arrs = [train_loop.solution_prediction(xs, step_idx=s)[:,0].flatten() for s in step_idxs]
            else:
                zs_arrs = []
            
            xi, yi = [np.linspace(np.min(xs[:,i]), np.max(xs[:,i]), res) for i in range(2)]
            grid = np.meshgrid(xi, yi)
            fig, axs = plt.subplots(nrows=1, ncols=len(step_idxs) + 1, figsize=(plt.rcParams['figure.figsize'][0] * (len(step_idxs) + 1.25), plt.rcParams['figure.figsize'][0]))
            if not has_step_idxs:
                axs = [axs]
            
            triang = tri.Triangulation(xs[:,0], xs[:,1])
            
            levels = np.linspace(np.min([z_actual] + zs_arrs), np.max([z_actual] + zs_arrs), num=2*res)
            if has_step_idxs:
                
                if plot_training_data:
                    for ax, step_idx in zip(axs[1:], step_idxs):
                        train_loop.plot_training_data(step_idx=step_idx, ax=ax)
                
                for ax, step_idx, z in zip(axs[1:], step_idxs, zs_arrs):
                    interpolator = tri.LinearTriInterpolator(triang, z.flatten())
                    Xi, Yi = np.meshgrid(xi, yi)
                    zi = interpolator(Xi, Yi)
                    cb = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
                    ax.set_title(f'Step {step_idx}')
                
            interpolator = tri.LinearTriInterpolator(triang, z_actual)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            cb = axs[0].contourf(xi, yi, zi, levels=levels, cmap="RdBu_r")
            axs[0].set_title(f'True solution')
                
            if has_step_idxs:
                axs = axs.ravel().tolist()
            fig.colorbar(cb, ax=axs)
            return fig

            # Plot eigenvalues
        # def plot_eigvals(train_loop,step_idx):
        #     fig = plt.plot(jnp.log10(train_loop._sample_intermediates['eigvals']))
        #     plt.title(f'Eigenvalues of NTK at step {step_idx}')
        #     plt.xlabel('Index')
        #     plt.ylabel('Log10 of eigenvalues')
        #     return fig

        # ------------------- Writing to tensorboard ------------------- #


        # only record at al_step to save time
        if writer is not None:
            
            writer.add_scalar('Loss/train', float(self.loss_train[-1]), self.current_train_step)
            writer.add_scalar('Loss/pde', float(self.loss_pde[-1]), self.current_train_step)
            writer.add_scalar('Loss/bc', float(self.loss_bc[-1]), self.current_train_step)
            writer.add_scalar('Loss/test_res', float(self.test_res[-1]), self.current_train_step)
            writer.add_scalar('Loss/test_err', float(self.test_err[-1]), self.current_train_step)

            if self.autoscale_loss_w_bcs or self.lra_loss_w_bcs:
                # Loss function weights
                # for i in range(len(self._sample_intermediates['chosen_pts']['bcs'])):
                writer.add_scalar('Weights/loss_w_bcs', float(jnp.mean(jnp.array(self.loss_w_bcs))), self.current_train_step)
                writer.add_scalar('Weights/loss_w_pde', float(self.loss_w_pde), self.current_train_step)
                writer.add_scalar('Weights/loss_w_anc', float(self.loss_w_anc), self.current_train_step)
            # if self.optim_method == 'multiadam' and self.opt_state is not None:
            #     # Loss function weights
            #     print(self.opt_state['exp_avg_sq'])
            #     writer.add_scalar('Weights/loss_w_bcs', float(1/jnp.sqrt(float(self.opt_state['exp_avg_sq'][1]) + 1e-8)), self.current_train_step)
            #     writer.add_scalar('Weights/loss_w_pde', float(1/jnp.sqrt(float(self.opt_state['exp_avg_sq'][0]) + 1e-8)), self.current_train_step)
            #     writer.add_scalar('Weights/loss_w_anc', float(self.loss_w_anc), self.current_train_step)



            # Add gradient histograms
            if hasattr(self, 'pde_grads'):
                writer.add_histogram(
                    'Gradients/pde_grads',
                    np.array(self.pde_grads.flatten()),
                    self.current_train_step
                )
            if hasattr(self, 'bc_grads'):
                writer.add_histogram(
                    'Gradients/bc_grads',
                    np.array(self.bc_grads.flatten()),
                    self.current_train_step
                )

            # Choice of points
            writer.add_scalar('Points/num_res', self._sample_intermediates['chosen_pts']['res'].shape[0], self.current_train_step)
            for i in range(len(self._sample_intermediates['chosen_pts']['bcs'])):
                writer.add_scalar(f'Points/num_bcs_{i}', self._sample_intermediates['chosen_pts']['bcs'][i].shape[0], self.current_train_step)
            if 'anc' in self._sample_intermediates['chosen_pts'].keys():
                writer.add_scalar('Points/num_anc', self._sample_intermediates['chosen_pts']['anc'].shape[0], self.current_train_step)
            # Inverse problem parameters
            if self.inverse_problem:
                writer.add_scalar('Params/pde_param', np.array(self.current_params[1][-1]), self.current_train_step)
            # GAAF values
            if 'scale' in self.current_params[0]['params']:
                writer.add_scalar('Params/GAAF_scale', np.array(self.current_params[0]['params']['scale']), self.current_train_step)
            # a-value
            if not skip:
                writer.add_scalar('Coef_eigvects/new_pts', np.array(self._sample_intermediates['label_info_new_pts']['a_norm']), self.current_train_step)
                writer.add_scalar('Coef_eigvects/returned_pts', np.array(self._sample_intermediates['label_info_returned_pts']['a_norm']), self.current_train_step)


            if self.tensorboard_plots and al_step:
                
                predict_plot = plot_prediction(self,step_idxs = [self.current_train_step])
                writer.add_figure('predict_plots', predict_plot, global_step=self.current_train_step)
                res_error_plot = plot_residue_loss(self,step_idxs = [self.current_train_step])
                writer.add_figure('res_error_plots', res_error_plot, global_step=self.current_train_step)

                if not skip and ('jac_train' in self._sample_intermediates.keys()) and ('eigvals' in self._sample_intermediates.keys()) and (self.data.test_x.shape[1] == 2):
                    
                    eigbasis_plots = plot_eigenbasis(self,step_idx = self.current_train_step,plots_per_level=5, use_const_res = True)
                    writer.add_figure('eigbasis_plots', eigbasis_plots, global_step=self.current_train_step)
                    # eigval_plots = plot_eigvals(self,step_idx = self.current_train_step)
                    # writer.add_figure('eigval_plots', eigval_plots, global_step=self.current_train_step)
            
            if al_step:
                writer.flush()
            
        # # remove jacobians to save memory
        # if self.snapshot_data[self.current_train_step]['al_intermediate'] is not None:
        #     self.snapshot_data[self.current_train_step]['al_intermediate'].pop('jac_train', None)
        #     self.snapshot_data[self.current_train_step]['al_intermediate'].pop('jac_candidates', None)


    def need_to_redo_active_learning(self, writer=None):
        if self.ntk_ratio_threshold is not None:
            curr_ntk = self.snapshot_data[self.current_train_step]['K_check_pts']
            prev_ntk = self.snapshot_data[self._last_al_step]['K_check_pts']
            F_diff = np.sqrt(np.mean((curr_ntk - prev_ntk)**2))
            F_prev = np.sqrt(np.mean((prev_ntk)**2))
            ratio = F_diff / F_prev
            if writer is not None:
                writer.add_scalar('Coef_eigvects/F_ratio', ratio, self.current_train_step)
            if ratio > self.ntk_ratio_threshold:
                print(f'Compare with step {self._last_al_step}, F_norm ratio is {F_diff:.7f} / {F_prev:.7f} = {ratio:.7f} > {self.ntk_ratio_threshold} - will redo active learning')
                return True
            else:
                print(f'Compare with step {self._last_al_step}, F_norm ratio is {F_diff:.7f} / {F_prev:.7f} = {ratio:.7f} - F_norm low enough value')
                return False
        else:
            # don't do this bit of checking
            return False
    

    def plot_gradient_distributions(self, writer=None):
        """Plot gradient distributions for current timestep"""
        print('plotting gradient distributions')
        # Get current gradient values
        curr_pde_grads = self.pde_grads.flatten()  # Changed indexing
        curr_bc_grads = self.bc_grads.flatten()    # Changed indexing
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PDE gradients
        ax1.hist(curr_pde_grads, 
                 bins=200, 
                 alpha=0.5, 
                 density=True,
                 color='blue')
        
        # BC gradients
        ax2.hist(curr_bc_grads, 
                 bins=200, 
                 alpha=0.5, 
                 density=True,
                 color='blue')

        ax1.set_title(f'PDE Loss Gradients (Step {self.current_train_step})')
        ax2.set_title(f'BC Loss Gradients (Step {self.current_train_step})')
        ax1.set_xlabel('Gradient Value')
        ax2.set_xlabel('Gradient Value')
        ax1.set_ylabel('Frequency')
        ax2.set_ylabel('Frequency')
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        
        if self.log_dir:
            plt.savefig(os.path.join(self.log_dir, f'gradient_distributions_step_{self.current_train_step}.png'))
        
        if writer is not None:
            writer.add_figure(f'Gradients/distributions_step_{self.current_train_step}', fig, global_step=self.current_train_step)
        
        plt.close()

    def lra_update_weights(self, writer):
        if self.current_train_step % 500 == 0:
            print('updating weights...')
            plot, _ = self.plot_training_data(self.current_samples)
            writer.add_figure('weight_samples', plot, global_step=self.current_train_step)
            print(f"Sample sizes are: N_res={len(self.current_samples['res'])}, N_b={[len(self.current_samples['bcs'][i]) for i in range(len(self.current_samples['bcs']))]}")
            
        def pde_loss(params, res):
            loss = jnp.mean(self.pde_residue_fn(params,res) ** 2)
            return loss
        
        pde_grad = jax.grad(pde_loss)(self.current_params, self.current_samples['res'])
        pde_grad = jnp.concatenate([leaf.ravel() for leaf in jax.tree_util.tree_leaves(pde_grad)])
        # Get mean absolute value of gradients, flattening parameter tree first
        max_pde_grad = jnp.max(jnp.abs(pde_grad))

        loss_w_bcs = []
        for i in range(len(self.current_samples['bcs'])):
            def bc_loss(params, bcs):
                loss = jnp.mean(self.icbc_error_fns[i](params[0], bcs) ** 2)
                return loss
            
            bc_grad = jax.grad(bc_loss)(self.current_params, self.current_samples['bcs'][i])
            bc_grad = jnp.concatenate([leaf.ravel() for leaf in jax.tree_util.tree_leaves(bc_grad)])
            lambda_bc = max_pde_grad / jnp.mean(jnp.abs(bc_grad))
            loss_w_bcs.append(lambda_bc)
        
        self.loss_w_bcs = [0.9 * self.loss_w_bcs[i] + (1 - 0.9) * loss_w_bcs[i] for i in range(len(self.loss_w_bcs))]

        if self.current_train_step % 500 == 0:
            print('updated weights:')
            print('loss_w_pde =', self.loss_w_pde)
            print('loss_w_bcs =', self.loss_w_bcs)
            #print('loss_w_anc =', self.loss_w_anc)
        
        del pde_grad, max_pde_grad, bc_grad, lambda_bc

    def ntk_update_weights(self, writer):

        if self.random_points_for_weights:
            res_proportion = 0.5
            n_res = int(res_proportion * self.mem_pts_total_budget) if (len(self.model.data.bcs) > 0) else (self.mem_pts_total_budget)
            n_per_bc = (self.mem_pts_total_budget - n_res) // len(self.model.data.bcs) if (len(self.model.data.bcs) > 0) else 0
            
            returned_pts = {
                'res': jnp.array(self.model.data.geom.random_points(n_res, random='pseudo')),
                'bcs': []
            }
            for i, bc in enumerate(self.model.data.bcs):
                if isinstance(bc, dde.icbc.boundary_conditions.PointSetBC):
                    idxs = jnp.array(np.random.choice(a=bc.points.shape[0], size=n_per_bc, replace=False))
                    xs = bc.points[idxs, :]
                elif isinstance(bc, dde.icbc.initial_conditions.IC):
                    xs = jnp.array(self.model.data.geom.random_initial_points(n_per_bc))
                else:
                    # xs = jnp.array(self.data.geom.random_boundary_points(n_per_bc))
                    xs = jnp.array(bc.geom.random_boundary_points(n_per_bc))
                returned_pts['bcs'].append(xs)
            d = returned_pts
            print(f'Randomly selected {n_res} residual points and {n_per_bc} boundary points per BC')
        else:
            d = self.current_samples

        if self.current_train_step % 50 == 0:
            print('updating weights...')
            plot, _ = self.plot_training_data(d)
            writer.add_figure('weight_samples', plot, global_step=self.current_train_step)
            print(f"Sample sizes are: N_res={len(d['res'])}, N_b={[len(d['bcs'][i]) for i in range(len(d['bcs']))]}")
            
        ntk_pde = self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['res'], code=-1))
        eigvals_pde = jnp.linalg.eigvalsh(ntk_pde)
        tr_pde = jnp.sum(eigvals_pde)
        if self.current_train_step % 50 == 0:
            print(f'PDE Cl top eigvals = {eigvals_pde[-min(5, eigvals_pde.shape[0]):]}')
            print(f'PDE Cl trace = {tr_pde}')
        
        ntk_bcs_list = [self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['bcs'][i], code=i)) for i in range(len(d['bcs']))]
        eigvals_bcs = [jnp.linalg.eigvalsh(ntkbc) for ntkbc in ntk_bcs_list]
        tr_bcs = [jnp.sum(eigbc) for eigbc in eigvals_bcs]
        if self.current_train_step % 50 == 0:
            for i, bc in enumerate(eigvals_bcs):
                print(f'BC {i} Cl top eigvals = {bc[-min(5, bc.shape[0]):]}')
                print(f'BC {i} Cl trace = {tr_bcs[i]}')
        
        if 'anc' in d.keys():
            ntk_anc = self._ntk_fn.get_ntk(jac1=self._ntk_fn.get_jac(xs=d['anc'], code=-2))
            eigvals_anc = jnp.linalg.eigvalsh(ntk_anc)
            print(f'Exp top eigvals = {eigvals_pde[-min(5, eigvals_anc.shape[0]):]}')
            tr_anc = jnp.sum(eigvals_anc)
            print(f'Exp trace = {tr_anc}')
        else:
            tr_anc = 0.
        if self.current_train_step % 50 == 0:
            print('trace computation complete')
        

        if 'anc' in d.keys():
            raise NotImplemented("Weight update for anchor not implemented")
        else:
            tr_anc = 0.
            
        total = tr_pde + tr_anc + sum(tr_bcs)
        self.loss_w_pde = total / tr_pde if tr_pde > 0. else 0.
        self.loss_w_bcs = [total/bc if bc > 0. else 0. for bc in tr_bcs]
        self.loss_w_anc = total / tr_anc if tr_anc > 0. else 0.
        
        if self.current_train_step % 500 == 0:
            print('updated weights:')
            print('loss_w_pde =', self.loss_w_pde)
            print('loss_w_bcs =', self.loss_w_bcs)
            #print('loss_w_anc =', self.loss_w_anc)

        del d, ntk_pde, eigvals_pde, tr_pde, ntk_bcs_list, eigvals_bcs, tr_bcs, tr_anc, total

    def train(self, train_steps=None):
        
        assert self.select_anchors_every % self.al_every == 0
        
        train_steps = train_steps if train_steps else self.train_steps
        al_rounds = train_steps // self.al_every
        
        # Logging tensorboard
        if self.log_dir:
            writer = SummaryWriter(self.log_dir)
            writer.add_text('args',f'Run with the following args: {vars(self)}',global_step=0)
        else:
            writer = None

        params = self.model.params
        opt_state = self.opt_state
                
        self.current_params = params
        
        if self.current_train_step == 0:
            do_anchor = self.select_anchors and (self.current_train_step % self.select_anchors_every == 0)
            print(f'======= Step {self.current_train_step} - performing active learning{" with anchor" if do_anchor else ""} =======')
            self._do_active_learning(do_anchor=do_anchor)
            print(f'======= Done active learning =======')
            self._record(writer=writer, al_step=True)
                
        # if self.optim_method == 'multiadam':
        #     solver = self._generate_solver(value_and_grad=self.loss_fn_grad)
        # else:
        solver = self._generate_solver(value_and_grad=self.loss_fn_grad)

        for r in range(al_rounds):
            
            start = time.time()  
                
            if opt_state is None:
                if self.optim_method == 'multiadam':
                    opt_state = solver.init(params)
                else:
                    opt_state = solver.init_state(params)
                                                
            for r_inside in range(self.al_every):
                
                self.current_train_step += 1
                # print(f'now on iteration {self.current_train_step}')

                if self.sample_each_round:
                    self.current_samples, _ =  self.al.generate_samples(verbose=False)

                if self.autoscale_loss_w_bcs and self.current_train_step % 10 == 0:
                    self.ntk_update_weights(writer=writer)
                    self.update_functions()
                    solver = self._generate_solver(value_and_grad=self.loss_fn_grad) #check if this is necessary
                elif self.lra_loss_w_bcs and self.current_train_step % 10 == 0:
                    self.lra_update_weights(writer=writer)
                    self.update_functions()
                    solver = self._generate_solver(value_and_grad=self.loss_fn_grad) #check if this is necessary


                # if self.current_train_step  % 50 == 0:
                #     print(f'Current train step: {self.current_train_step}')

                # l_, grad = self.loss_fn_grad(params)
                # updates, opt_state = opt.update(grad, opt_state)
                # params = optax.apply_updates(params, updates)
                if self.optim_method == 'multiadam':
                    grads = [jax.grad(self.pde_loss_fn)(params), jax.grad(self.bc_loss_fn)(params)]
                    updates, opt_state = solver.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                else:
                    params, opt_state = solver.update(params, opt_state)
                
                self.net.params = params[0] 
                self.model.params = params
                self.opt_state = opt_state
                self.current_params = params
                
                if self.current_train_step % self.snapshot_every == 0:
                    print(f'======= Step {self.current_train_step} - recording =======')
                    
                    def pde_loss(params, res):
                        loss = self.loss_w_pde * jnp.mean(self.pde_residue_fn(params,res) ** 2)
                        return loss
                            
                    def bc_loss(params, bcs):
                        loss = 0
                        for i in range(len(bcs)):
                            loss += self.loss_w_bcs[i] * jnp.mean(self.icbc_error_fns[i](params[0], bcs[i]) ** 2)
                        return loss
                    
                    print(f'self.save_grads = {self.save_grads}')
                    if self.save_grads:
                        # Collect PDE gradients
                        pde_grad = jax.grad(pde_loss)(params, self.current_samples['res'])
                        self.pde_grads = jnp.concatenate([leaf.ravel() for leaf in jax.tree_util.tree_leaves(pde_grad)])  # Changed to dict

                        # Collect BC gradients
                        bc_grad = jax.grad(bc_loss)(params, self.current_samples['bcs'])
                        self.bc_grads = jnp.concatenate([leaf.ravel() for leaf in jax.tree_util.tree_leaves(bc_grad)])  # Changed to dict
                        
                        # Plot gradient distributions periodically
                        #self.plot_gradient_distributions(writer=writer)
                        #print('saved graident distribution at {self.current_train_step}')
                        
                        del pde_grad, bc_grad

                    self._record(writer=writer, al_step=(self.current_train_step % self.al_every == 0))

                    print(f'completed snapshot at step {self.current_train_step}')

                     # Clean up after each round
                    gc.collect()
                    jax.clear_caches()
                    
                    # either last iteration of round (have to do AL), or trigger to redo AL from NTK value
                    if not self.sample_each_round and ((r_inside == self.al_every - 1) or self.need_to_redo_active_learning(writer=writer)):
                        do_anchor = self.select_anchors and (self.current_train_step % self.select_anchors_every == 0)
                        print(f'======= Step {self.current_train_step} - performing active learning =======')
                        self._do_active_learning(do_anchor=do_anchor)
                        print(f'======= Done active learning =======')
                        # if self.reset_sampler_at_al:
                        solver = self._generate_solver(value_and_grad=self.loss_fn_grad) #TODO: not sure what this does
                    elif (r_inside == self.al_every - 1):
                        self.al_data_round[self.current_train_step] = self.current_samples 


                # self.net.params = params[0]
                # self.model.params = params

        end = time.time()
        print(f"Time required for last {self.al_every} steps = {end - start:.6f} seconds")
                                            
        if writer is not None:
            writer.close()
        # Record plots to tensorboard
        # if self.log_dir:
        #     fig_loss_curve, _ = self.plot_losses()
        #     fig_predict = plot_prediction(train_loop=self, step_idxs=[0, max(self.snapshot_data.keys())])
        #     fig_error = plot_error(train_loop=self, step_idxs=[0, max(self.snapshot_data.keys())]);
        #     fig_loss = plot_residue_loss(train_loop=self, step_idxs=[0, max(self.snapshot_data.keys())])
            
        #     writer.add_figure('Loss/train', fig_loss_curve, self.current_train_step)
        #     writer.add_figure('Loss/test_res', fig_predict, self.current_train_step)
        #     writer.add_figure('Loss/test_err', fig_error, self.current_train_step)
        #     writer.add_figure('Loss/residue', fig_loss, self.current_train_step)
        #     writer.flush()
        #     writer.close()

    def solution_prediction(self, xs, step_idx=None):
        step_idx = list(self.snapshot_data.keys())[-1] if step_idx is None else step_idx
        params = self.snapshot_data[step_idx]['params'][0]
        return self.net.apply(params, xs, training=True)
    
    def pde_residue(self, xs, step_idx=None):
        step_idx = list(self.snapshot_data.keys())[-1] if step_idx is None else step_idx
        params = self.snapshot_data[step_idx]['params']
        return self.pde_residue_fn(params, xs)
    
    def solution_error(self, xs, ys=None, step_idx=None):
        assert not ((ys is None) and (self.data.soln is None))
        step_idx = list(self.snapshot_data.keys())[-1] if step_idx is None else step_idx
        params = self.snapshot_data[step_idx]['params'][0]
        ys = self.data.soln(xs) if ys is None else ys
        return self.soln_error_fn(params, xs, ys)

    def plot_training_data(self, samples=None, step_idx=None, ax=None, marker_size=8):
        colours = ['cyan','orange', 'purple', 'red', 'purple', 'brown', 'pink', 'grey', 'green', 'magenta']
        colours = cycle(colours)
        if step_idx is None and samples is None:
            samples = self.current_samples
        else:
            step_idx = list(self.al_data_round.keys())[-1] if step_idx is None else max(k for k in self.al_data_round.keys() if k <= step_idx)
            if samples is None:
                samples = self.al_data_round[step_idx]
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.plot(samples['res'][:, 0], samples['res'][:, 1], 'o', color='black', markersize=marker_size)
        if 'anc' in samples.keys():
            ax.plot(samples['anc'][:, 0], samples['anc'][:, 1], '^', color='blue', markersize=marker_size)
        for i, bc_pts in enumerate(samples['bcs']):
            ax.plot(bc_pts[:, 0], bc_pts[:, 1], ls='', marker=(i+4,0,0), color=next(colours), markersize=marker_size)
        if fig is not None:
            plt.close(fig)
        return fig, ax

    def plot_losses(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.semilogy(self.loss_steps, self.loss_train, label='Train loss')
        ax.semilogy(self.loss_steps, self.test_res, label='Residue')
        # if self.soln_error_fn is not None:
        ax.semilogy(self.loss_steps, self.test_err, label='Error')
        ax.legend()
        # if fig is not None:
        #     plt.close(fig)
        return fig, ax


