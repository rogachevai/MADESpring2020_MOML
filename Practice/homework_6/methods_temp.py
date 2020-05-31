import numpy as np
import numpy.linalg as npla
import scipy
from collections import defaultdict, deque
from datetime import datetime
from numpy.linalg import LinAlgError
from scipy.optimize.linesearch import scalar_search_wolfe2
import time
from numpy.linalg import norm
import oracles
from scipy.linalg import cho_factor, cho_solve


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c
        elif self._method == 'Armijo':
            alpha_0 = previous_alpha if previous_alpha is not None else self.alpha_0
            return self.armijo_search(oracle, x_k, d_k, alpha_0)
        elif self._method == 'Wolfe':
            alpha = scalar_search_wolfe2(
                phi=lambda step: oracle.func_directional(x_k, d_k, step),
                derphi=lambda step: oracle.grad_directional(x_k, d_k, step),
                c1=self.c1,
                c2=self.c2
            )[0]
            if alpha is None:
                return self.armijo_search(oracle, x_k, d_k, self.alpha_0)
            else:
                return alpha

        return None

    def armijo_search(self, oracle, x_k, d_k, alpha_0):
        phi = lambda step: oracle.func_directional(x_k, d_k, step)
        alpha = alpha_0
        coef = self.c1 * oracle.grad_directional(x_k, d_k, 0)
        while phi(alpha) > phi(0) + alpha * coef:
            alpha = alpha / 2
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


# class History(object):
#     def __init__(self, trace):
#         self.trace = trace
#         self.t = time.time()
#         if not trace:
#             self.history = None
#             return
#         self.history = defaultdict(list)
#
#     def add_record_to_history(self, func_val, grad_norm, x):
#         if not self.trace:
#             return
#         self.history['time'].append(time.time() - self.t)
#         self.history['func'].append(func_val)
#         self.history['grad_norm'].append(grad_norm)
#         if x.size <= 2:
#             self.history['x'].append(x)
#         self.history['x_star'] = x


# def add_record_to_history(self):
#     now = datetime.now()
#     self.time += (now - self._absolute_time).total_seconds()
#     self._absolute_time = now
#     self.hist['func'].append(self.oracle.func(self.x_k))
#     self.hist['time'].append(self.time)
#     if not hasattr(self, 'grad_k'):
#         self.grad_k = self.oracle.grad(self.x_k)
#     self.hist['grad_norm'].append(npla.norm(self.grad_k))


class Newton(object):
    def __init__(self, oracle, x_0, tolerance=1e-4, line_search_options=None):
        self.oracle = oracle
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.hist = defaultdict(list)
        self.x_0 = x_0

    def add_record_to_history_(self):
        now = datetime.now()
        self.time += (now - self.s_time).total_seconds()
        self.s_time = now
        self.hist['func'].append(self.oracle.func(self.x_k))
        self.hist['time'].append(self.time)
        if not hasattr(self, 'grad_k'):
            self.grad_k = self.oracle.grad(self.x_k)
        self.hist['grad_norm'].append(norm(self.grad_k))
        if self.x_k.size <= 2:
            self.hist['x'].append(self.x_k)

    def run(self, max_iter=100):
        def get_d(x, grad, oracle):
            upper_triangle, _ = cho_factor(oracle.hess(x), lower=False, overwrite_a=True, check_finite=True)
            direction = cho_solve((upper_triangle, False), -grad, overwrite_b=False, check_finite=True)
            return direction

        message = 'ok'
        self.x_k = np.copy(self.x_0)
        benchmark = self.tolerance * norm(
            self.oracle.grad(self.x_k)) ** 2
        self.time = 0.
        self.s_time = datetime.now()
        for i in range(max_iter + 1):
            self.grad_k = self.oracle.grad(self.x_k)
            self.add_record_to_history_()
            if norm(self.grad_k) ** 2 <= benchmark:
                message = "success"
                break
            d = get_d(self.x_k, self.grad_k, self.oracle)
            alpha = self.line_search_tool.line_search(self.oracle, self.x_k, d)
            self.x_k = self.x_k + d * alpha
        self.hist['x_star'] = self.x_k.copy()

        return self.x_k, message, self.hist


class BFGS(object):
    def __init__(self, oracle, x_0, tolerance=1e-4, line_search_options=None):
        self.oracle = oracle
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.hist = defaultdict(list)
        # maybe more of your code here
        self.x_k = np.copy(x_0)
        self.grad_norm_0 = np.linalg.norm(self.oracle.grad(x_0))
        self.H_k = np.eye(x_0.shape[0], dtype=x_0.dtype)
        if self.line_search_tool._method == 'Constant':
            self.alpha_0 = self.line_search_tool.c
        else:
            self.alpha_0 = 1.
        self.grad_k = self.oracle.grad(self.x_k)

    def add_record_to_history_(self):
        now = datetime.now()
        self.time += (now - self.s_time).total_seconds()
        self.s_time = now
        self.hist['func'].append(self.oracle.func(self.x_k))
        self.hist['time'].append(self.time)
        if not hasattr(self, 'grad_k'):
            self.grad_k = self.oracle.grad(self.x_k)
        self.hist['grad_norm'].append(norm(self.grad_k))
        if self.x_k.size <= 2:
            self.hist['x'].append(self.x_k)

    def update_H(self):
        self.s_k = (self.x_k - self.x_k_).reshape((self.x_k.shape[0], 1))
        self.y_k = (self.grad_k - self.grad_k_).reshape((self.x_k.shape[0], 1))
        ml = self.s_k.flatten().dot(self.y_k.flatten())
        c = 1e-8
        self.H_k -= self.H_k.dot(self.y_k).dot(self.s_k.T) / (ml + c)
        self.H_k -= (self.H_k.dot(self.y_k).dot(self.s_k.T) / (ml + c)).T
        self.H_k += self.s_k.dot(self.s_k.T) / (ml + c)

    def run(self, max_iter=100):
        message = 'ok'
        benchmark = self.tolerance * norm(
            self.oracle.grad(self.x_k)) ** 2
        self.time = 0.
        self.s_time = datetime.now()
        for _ in range(max_iter + 1):
            self.grad_k = self.oracle.grad(self.x_k)
            self.add_record_to_history_()
            if norm(self.grad_k) ** 2 <= benchmark:
                message = "success"
                break
            self.x_k_ = self.x_k.copy()
            self.grad_k_ = self.grad_k.copy()
            d_k = -self.H_k.dot(self.grad_k)
            alpha_k = self.line_search_tool.line_search(self.oracle, self.x_k, d_k, self.alpha_0)
            self.x_k += alpha_k * d_k
            self.grad_k = self.oracle.grad(self.x_k)
            self.update_H()
        self.hist['x_star'] = self.x_k.copy()
        return self.x_k, message, self.hist




class LBFGS(object):
    def __init__(self, oracle, x_0, tolerance=1e-4, memory_size=10,
                 line_search_options=None):
        self.oracle = oracle
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.hist = defaultdict(list)
        self.memory_size = memory_size
        # maybe more of your code here
        self.x_k = np.copy(x_0)
        self.grad_norm = norm(self.oracle.grad(x_0))
        self.grad_k = self.oracle.grad(self.x_k)
        if self.line_search_tool._method == 'Constant':
            self.alpha_0 = self.line_search_tool.c
        else:
            self.alpha_0 = 1.
        self.lbfgs_queue = deque(maxlen=memory_size)

    def add_record_to_history_(self):
        now = datetime.now()
        self.time += (now - self.s_time).total_seconds()
        self.s_time = now
        self.hist['func'].append(self.oracle.func(self.x_k))
        self.hist['time'].append(self.time)
        if not hasattr(self, 'grad_k'):
            self.grad_k = self.oracle.grad(self.x_k)
        self.hist['grad_norm'].append(npla.norm(self.grad_k))
        if self.x_k.size <= 2:
            self.hist['x'].append(self.x_k)

    def lbfgs_mul(self, v, memory, gamma):
        if len(memory) == 0:
            return gamma * v
        s, y = memory[-1]
        v1 = v - s.dot(v) / y.dot(s) * y
        z = self.lbfgs_mul(v1, memory[:-1], gamma)
        return z + (s.dot(v) - y.dot(z)) / y.dot(s) * s

    def run(self, max_iter=100):
        message = 'ok'
        benchmark = self.tolerance * norm(
            self.oracle.grad(self.x_k)) ** 2
        self.time = 0.
        self.s_time = datetime.now()
        for _ in range(max_iter + 1):
            self.grad_k = self.oracle.grad(self.x_k)
            self.add_record_to_history_()
            if norm(self.grad_k) ** 2 <= benchmark:
                message = "success"
                break
            self.x_k_ = self.x_k
            self.grad_k_ = self.grad_k
            try:
                s, y = self.lbfgs_queue.pop()
                self.lbfgs_queue.append((s, y))
                gamma_0 = y.dot(s) / y.dot(y)
                d_k = self.lbfgs_mul(-self.grad_k, list(self.lbfgs_queue), gamma_0)
            except IndexError:
                d_k = -self.grad_k
            alpha_k = self.line_search_tool.line_search(self.oracle, self.x_k, d_k, self.alpha_0)
            if alpha_k is None:
                return
            self.x_k = self.x_k + alpha_k * d_k
            self.lbfgs_queue.append((self.x_k - self.x_k_, self.oracle.grad(self.x_k) - self.grad_k_))
        self.hist['x_star'] = self.x_k.copy()
        return self.x_k, message, self.hist

#
# def lbfgs_direction(H, grad):
#     s, y = H[0]
#     gamma = y.dot(s) / y.dot(y)
#     return bfgs_multiply(-grad, H, 0, gamma)
#
#
# def bfgs_multiply(v, H, index, gamma):
#     if index == len(H):
#         return gamma * v
#     s, y = H[index]
#     s_dot_v = s.dot(v)
#     y_dot_s = y.dot(s)
#     v_next = v - s_dot_v / y_dot_s * y
#     z = bfgs_multiply(v_next, H, index + 1, gamma)
#     return z + (s_dot_v - y.dot(z)) / y_dot_s * s
# your code here
