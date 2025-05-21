import numpy as np
cimport numpy as np
from libc.math cimport pow

cdef class MonteCarloPricer:
    cdef double r, p, u, d
    cdef int steps
    cdef double S0

    def __init__(self, double r, double p, double u, double d, int steps, double S0):
        self.r = r
        self.p = p
        self.u = u
        self.d = d
        self.steps = steps
        self.S0 = S0

    cpdef double _evaluate_mc(self, func, int n, list trajectory=None, int t=0):
        if trajectory is None:
            trajectory = [self.S0]
        if t == self.steps:
            return func(trajectory)
        cdef double u_rand = np.random.uniform(0, 1)
        if u_rand < self.p:
            return self._evaluate_mc(func, n, trajectory + [self.u * trajectory[-1]], t + 1)
        else:
            return self._evaluate_mc(func, n, trajectory + [self.d * trajectory[-1]], t + 1)

    cpdef list get_price_of_asset_mc(self, func, int n):
        cdef int i
        cdef double discount = pow(1 + self.r, self.steps)
        prices = [self._evaluate_mc(func, n) for i in range(n)]
        sum_mc = [sum(prices[:i + 1]) / (discount * (i + 1)) for i in range(n)]
        return sum_mc
