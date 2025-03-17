import numpy as np
from kernels import Kernel
import heapq

class Monomial(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __lt__(self, other):
        assert isinstance(other, Monomial)
        return self.degree() < other.degree()

    def degree(self):
        if len(self) == 0:
            return 0
        return np.sum(self.values())

    def copy(self):
        return Monomial(super().copy())

    def __repr__(self):
        if self.degree() == 0:
            return "1"
        monostr = ""
        for idx, exp in self.items():
            expstr = f"^{exp}" if exp > 1 else ""
            monostr += f"x{idx}{expstr}."
        return monostr[:-1]

def generate_fra_monomials(data_covar_eigvals, num_monomials, kernel_class):
    assert isinstance(num_monomials, int) and num_monomials > 0, "num_monomials must be positive integer"
    d = len(data_covar_eigvals)

    monomials = [Monomial({})]
    fra_eigvals = [kernel_class.get_level_coeff_fn(monomials[0])]
    # Each entry in the priority queue is (-fra_eigval, Monomial({idx:exp, ...}))
    pq = [(-get_fra_eigval(data_covar_eigvals, Monomial({0:1}), kernel_class), Monomial({0:1}))]
    heapq.heapify(pq)
    while pq and len(monomials) < num_monomials:
        neg_fra_eigval, monomial = heapq.heappop(pq)
        fra_eigvals.append(-neg_fra_eigval)
        monomials.append(monomial)

        last_idx = max(monomial.keys())
        if last_idx + 1 < d:
            left_monomial = monomial.copy()
            left_monomial[last_idx] -= 1
            if left_monomial[last_idx] == 0:
                del left_monomial[last_idx]
            left_monomial[last_idx + 1] = left_monomial.get(last_idx + 1, 0) + 1

            fra_eigval = get_fra_eigval(data_covar_eigvals, left_monomial, kernel_class)
            heapq.heappush(pq, (-fra_eigval, left_monomial))

        right_monomial = monomial.copy()
        right_monomial[last_idx] += 1
        fra_eigval = get_fra_eigval(data_covar_eigvals, right_monomial, kernel_class)
        heapq.heappush(pq, (-fra_eigval, right_monomial))

    return np.array(fra_eigvals), monomials

def lookup_monomial_idx(monomials, monomial):
    for i, mon in enumerate(monomials):
        if mon == monomial:
            return i
    return None

def get_fra_eigval(eigvals, monomial, kernel_class: Kernel):
    fra_eigval = kernel_class.get_level_coeff_fn(monomial)
    for i, exp in monomial.items():
        fra_eigval *= eigvals[i].item() ** exp
    return fra_eigval

