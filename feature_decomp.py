import heapq
import numpy as np
from tqdm import tqdm
from utils import ensure_numpy


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
        for idx, exp in sorted(self.items()):
            expstr = f"^{exp}" if exp > 1 else ""
            monostr += f"x_{{{idx}}}{expstr}"
        return f"${monostr}$"


def compute_hea_eigval(data_eigvals, monomial, eval_level_coeff):
    hea_eigval = eval_level_coeff(monomial.degree())
    for i, exp in monomial.items():
        hea_eigval *= data_eigvals[i] ** exp
    return hea_eigval


def generate_hea_monomials(data_eigvals, num_monomials, eval_level_coeff, kmax=10):
    """
    Generates HEA eigenvalues and monomials in canonical learning order.

    Args:
        data_eigvals (iterable): data covariance eigenvalues
        num_monomials (int): Number of monomials to generate.
        eval_level_coeff (function): Function to evaluate kernel level coefficients.
        kmax (int): Search monomials up to degree kmax

    Returns:
        - hea_eigvals (np.ndarray): Array of HEA eigenvalues.
        - monomials (list): List of generated monomials.
    """
    try:
        num_monomials = abs(int(num_monomials))
    except Exception as e:
        raise ValueError(f"type(num_monomials) must be int, not {type(num_monomials)}") from e
    assert num_monomials >= 1
    data_eigvals = ensure_numpy(data_eigvals)
    d = len(data_eigvals)

    # populate priority queue with top monomial at each degree up to kmax
    pq = []
    pq_members = set()
    first_hea_eigval = compute_hea_eigval(data_eigvals, Monomial({}), eval_level_coeff)
    for k in range(1, kmax+1):
        monomial = Monomial({0: k})
        hea_eigval = compute_hea_eigval(data_eigvals, monomial, eval_level_coeff)
        if hea_eigval > first_hea_eigval:
            break
        # Each entry in the priority queue is (-hea_eigval, Monomial({idx:exp, ...}))
        pq.append((-hea_eigval, monomial))
        pq_members.add(repr(monomial))
    heapq.heapify(pq)
    
    monomials = [Monomial({})]
    hea_eigvals = [first_hea_eigval]
    for _ in range(num_monomials-1):
        if not pq:
            print("Warning: priority queue exhausted before reaching num_monomials.")
            return np.array(hea_eigvals), monomials
        neg_hea_eigval, monomial = heapq.heappop(pq)
        pq_members.remove(repr(monomial))
        hea_eigvals.append(-neg_hea_eigval)
        monomials.append(monomial)
        
        # generate successor monomials of same degree
        for idx in list(monomial.keys()):
            if idx + 1 < d:
                next_monomial = monomial.copy()
                next_monomial[idx] -= 1
                if next_monomial[idx] == 0:
                    del next_monomial[idx]
                next_monomial[idx + 1] = next_monomial.get(idx + 1, 0) + 1
                if repr(next_monomial) not in pq_members:
                    hea_eigval = compute_hea_eigval(data_eigvals, next_monomial, eval_level_coeff)
                    heapq.heappush(pq, (-hea_eigval, next_monomial))
                    pq_members.add(repr(next_monomial))

    return np.array(hea_eigvals), monomials


# DEPRECATED
def generate_fra_monomials(data_covar_eigvals, num_monomials, eval_level_coeff, kmax=10):
    """
    Generates FRA eigenvalues and monomials in canonical learning order.

    Args:
        data_covar_eigvals (iterable): data covariance eigenvalues
        num_monomials (int): Number of monomials to generate.
        eval_level_coeff (function): Function to evaluate kernel level coefficients.
        kmax (int): Search monomials up to order k

    Returns:
        - fra_eigvals (np.ndarray): Array of FRA eigenvalues.
        - monomials (list): List of generated monomials.
    """
    try:
        num_monomials = abs(int(num_monomials))
    except Exception as e:
        raise ValueError(f"type(num_monomials) must be int, not {type(num_monomials)}") from e
    assert num_monomials >= 1
    data_covar_eigvals = ensure_numpy(data_covar_eigvals)
    d = len(data_covar_eigvals)

    monomials = [Monomial({})]
    fra_eigvals = [compute_hea_eigval(data_covar_eigvals, monomials[0], eval_level_coeff)]
    # Each entry in the priority queue is (-hea_eigval, Monomial({idx:exp, ...}))
    pq = [(-compute_hea_eigval(data_covar_eigvals, Monomial({0:1}), eval_level_coeff), Monomial({0:1}))]
    heapq.heapify(pq)
    # only show tqdm bar in console
    for _ in tqdm(range(num_monomials-1), initial=1, desc="Generating monomials", disable=None):
        if not pq:
            return np.array(fra_eigvals), monomials
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

            fra_eigval = compute_hea_eigval(data_covar_eigvals, left_monomial, eval_level_coeff)
            heapq.heappush(pq, (-fra_eigval, left_monomial))

        if monomial.degree() < kmax:
            right_monomial = monomial.copy()
            right_monomial[last_idx] += 1
            if eval_level_coeff(right_monomial.degree()) < 1e-12:
                right_monomial[last_idx] += 1
            if right_monomial.degree() > kmax:
                continue
            fra_eigval = compute_hea_eigval(data_covar_eigvals, right_monomial, eval_level_coeff)
            heapq.heappush(pq, (-fra_eigval, right_monomial))

    return np.array(fra_eigvals), monomials


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
