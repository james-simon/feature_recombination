import heapq
import numpy as np
from tqdm import tqdm


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


def compute_hea_eigval(data_eigvals, monomial, eval_level_coeff):
    hea_eigval = eval_level_coeff(monomial.degree())
    for i, exp in monomial.items():
        hea_eigval *= data_eigvals[i].item() ** exp
    return hea_eigval


def generate_hea_monomials(data_covar_eigvals, num_monomials, eval_level_coeff, kmax=10):
    """
    Generates HEA eigenvalues and monomials in canonical learning order.

    Args:
        data_covar_eigvals (iterable): data covariance eigenvalues
        num_monomials (int): Number of monomials to generate.
        eval_level_coeff (function): Function to evaluate kernel level coefficients.
        kmax (int): Search monomials up to order k

    Returns:
        - hea_eigvals (np.ndarray): Array of HEA eigenvalues.
        - monomials (list): List of generated monomials.
    """
    #implemented as num_monomials is occasionally gotten from a numpy array, so they're np.int's
    if not(isinstance(num_monomials, int) and num_monomials > 0):    
        print(f"num_monomials being forced to a positive integer, currently {type(num_monomials)}, {num_monomials}")
        num_monomials = abs(int(num_monomials))
        if type(num_monomials) is not int:
            raise ValueError(f"num_monomials must be a positive integer, got {num_monomials} of type {type(num_monomials)}")
    d = len(data_covar_eigvals)

    monomials = [Monomial({})]
    hea_eigvals = [compute_hea_eigval(data_covar_eigvals, monomials[0], eval_level_coeff)]
    # Each entry in the priority queue is (-hea_eigval, Monomial({idx:exp, ...}))
    pq = [(-compute_hea_eigval(data_covar_eigvals, Monomial({0:1}), eval_level_coeff), Monomial({0:1}))]
    heapq.heapify(pq)
    # only show tqdm bar in console
    for _ in tqdm(range(num_monomials-1), initial=1, desc="Generating monomials", disable=None):
        if not pq:
            return np.array(hea_eigvals), monomials
        neg_hea_eigval, monomial = heapq.heappop(pq)
        hea_eigvals.append(-neg_hea_eigval)
        monomials.append(monomial)

        last_idx = max(monomial.keys())
        if last_idx + 1 < d:
            left_monomial = monomial.copy()
            left_monomial[last_idx] -= 1
            if left_monomial[last_idx] == 0:
                del left_monomial[last_idx]
            left_monomial[last_idx + 1] = left_monomial.get(last_idx + 1, 0) + 1

            hea_eigval = compute_hea_eigval(data_covar_eigvals, left_monomial, eval_level_coeff)
            heapq.heappush(pq, (-hea_eigval, left_monomial))

        if monomial.degree() < kmax:
            right_monomial = monomial.copy()
            right_monomial[last_idx] += 1
            if eval_level_coeff(right_monomial.degree()) < 1e-12:
                right_monomial[last_idx] += 1
            if right_monomial.degree() > kmax:
                continue
            hea_eigval = compute_hea_eigval(data_covar_eigvals, right_monomial, eval_level_coeff)
            heapq.heappush(pq, (-hea_eigval, right_monomial))

    return np.array(hea_eigvals), monomials


generate_fra_monomials = generate_hea_monomials


def fra_terms_from_monomials(monomials, data_eigvals, eval_level_coeff):
    """
    Computes the eigenvalues for a list of monomials.

    Args:
        monomials (list): List of Monomial objects.
        data_eigvals (torch.Tensor): Eigenvalues of the covariance matrix.
        eval_level_coeff (function): Function to evaluate level coefficients.

    Returns:
        np.ndarray: Array of eigenvalues corresponding to the monomials.
    """
    fra_eigvals = np.zeros(len(monomials))
    for i, monomial in enumerate(monomials):
        fra_eigvals[i] = compute_hea_eigval(data_eigvals, monomial, eval_level_coeff)
    return fra_eigvals


def lookup_monomial_idx(monomials, monomial):
    return next((i for i, m in enumerate(monomials) if m == monomial), None)
