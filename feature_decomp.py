import heapq
import numpy as np

class Monomial(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __lt__(self, other):
        assert isinstance(other, Monomial)
        return self.degree() < other.degree()

    def degree(self):
        if len(self) == 0:
            return 0
        return sum(self.values())

    def copy(self):
        return Monomial(super().copy())

    def __repr__(self):
        if self.degree() == 0:
            return "1"
        monostr = ""
        for idx, exp in self.items():
            expstr = f"^{exp}" if exp > 1 else ""
            monostr += f"x_{{{idx}}}{expstr}"
        return f"${monostr}$"


def get_fra_eigval(data_eigvals, monomial, eval_level_coeff):
    fra_eigval = eval_level_coeff(monomial.degree())
    for i, exp in monomial.items():
        fra_eigval *= data_eigvals[i].item() ** exp
    return fra_eigval


def generate_fra_monomials(data_covar_eigvals, num_monomials, eval_level_coeff):
    assert isinstance(num_monomials, int) and num_monomials > 0, "num_monomials must be positive integer"
    d = len(data_covar_eigvals)

    monomials = [Monomial({})]
    fra_eigvals = [get_fra_eigval(data_covar_eigvals, monomials[0], eval_level_coeff)]
    # Each entry in the priority queue is (-fra_eigval, Monomial({idx:exp, ...}))
    pq = [(-get_fra_eigval(data_covar_eigvals, Monomial({0:1}), eval_level_coeff), Monomial({0:1}))]
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

            fra_eigval = get_fra_eigval(data_covar_eigvals, left_monomial, eval_level_coeff)
            heapq.heappush(pq, (-fra_eigval, left_monomial))

        right_monomial = monomial.copy()
        right_monomial[last_idx] += 1
        fra_eigval = get_fra_eigval(data_covar_eigvals, right_monomial, eval_level_coeff)
        heapq.heappush(pq, (-fra_eigval, right_monomial))

    return np.array(fra_eigvals), monomials


def lookup_monomial_idx(monomials, monomial):
    for i, mon in enumerate(monomials):
        if mon == monomial:
            return i
    return None


def get_eigenspectrum_comparison(X, kernel_class):
    
    S_normalized = X.cpu().numpy()/np.sum(X.cpu().numpy()) #normalization condition
    kernel = kernel_class(X, kernel_width=1)
    eigvals = kernel.eigenvals()

    level_coeffs = kernel_class.get_level_coeff_fn(kernel_width=1, data_eigvals=S_normalized)

    top_mode_idx = 500
    fra_eigvals, monomials = generate_fra_monomials(S_normalized, top_mode_idx, level_coeffs)
    degrees = [monomial.degree() for monomial in monomials]