import heapq
import numpy as np
from tqdm import trange


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

    def max_degree(self):
        if len(self) == 0:
            return 0
        return max(self.values())

    def copy(self):
        return Monomial(super().copy())

    #refactored from __repr__ to also get __str__
    def _latex_body(self):
        if self.degree() == 0:
            return "1"
        monostr = ""
        for idx, exp in self.items():
            expstr = f"^{exp}" if exp > 1 else ""
            monostr += f"x_{{{idx}}}{expstr}"
        return f"{monostr}"

    def __str__(self) -> str:
        return f"${self._latex_body()}$"

    def __repr__(self):
        return f"${self._latex_body()}$"

    @classmethod
    def from_repr(cls, s: str) -> "Monomial":
        """
        Parse strings like '$x_{0}^2x_{3}x_{10}^5$' or '$1$' into a Monomial.
        No regex used. Strict about format produced by __repr__/__str__.
        """
        if not isinstance(s, str):
            raise TypeError("from_repr expects a string")

        s = s.strip()
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1]
        s = s.replace(" ", "")

        if s in {"", "1"}:
            return cls()

        i, n = 0, len(s)
        out = {}

        def expect(ch: str):
            nonlocal i
            if i >= n or s[i] != ch:
                raise ValueError(f"Expected '{ch}' at pos {i} in {s!r}")
            i += 1

        def read_digits() -> int:
            nonlocal i
            start = i
            while i < n and s[i].isdigit():
                i += 1
            if start == i:
                raise ValueError(f"Expected digits at pos {start} in {s!r}")
            return int(s[start:i])

        while i < n:
            # x_{idx}
            expect('x')
            expect('_')
            expect('{')
            idx = read_digits()
            expect('}')

            # optional ^exp
            exp = 1
            if i < n and s[i] == '^':
                i += 1
                exp = read_digits()

            out[idx] = out.get(idx, 0) + exp

        return cls(out)
    
    def basis_list(self, include_one: bool = False, canonical: bool = True):
        """
        Return a list of unit-degree Monomials whose product equals this monomial.
        Example: Monomial({0: 2, 3: 1}) -> [Monomial({0:1}), Monomial({0:1}), Monomial({3:1})]
        If degree == 0, returns [] unless include_one=True (then [Monomial({})]).
        If canonical=True, factors are ordered by increasing variable index.
        """
        if self.degree() == 0:
            return [Monomial({})] if include_one else []

        items = sorted(self.items()) if canonical else self.items()
        factors = []
        for idx, exp in items:
            for _ in range(int(exp)):
                factors.append(Monomial({idx: 1}))
        return factors


def get_fra_eigval(data_eigvals, monomial, eval_level_coeff):
    fra_eigval = eval_level_coeff(monomial.degree())
    for i, exp in monomial.items():
        fra_eigval *= data_eigvals[i].item() ** exp
    return fra_eigval


def generate_fra_monomials(data_covar_eigvals, num_monomials, eval_level_coeff, kmax=10):
    """
    Generates monomials through a greedy search.

    Args:
        data_covar_eigvals (torch.Tensor): Eigenvalues of the covariance matrix.
        num_monomials (int): Number of monomials to generate.
        eval_level_coeff (function): Function to evaluate level coefficients.
        kmax (int): Search monomials up to order k

    Returns:
        A tuple containing:
        - fra_eigvals (np.ndarray): Array of fra eigenvalues.
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
    fra_eigvals = [get_fra_eigval(data_covar_eigvals, monomials[0], eval_level_coeff)]
    # Each entry in the priority queue is (-fra_eigval, Monomial({idx:exp, ...}))
    pq = [(-get_fra_eigval(data_covar_eigvals, Monomial({0:1}), eval_level_coeff), Monomial({0:1}))]
    heapq.heapify(pq)
    with trange(num_monomials-1, initial=1, desc="Generating monomials", unit="step", total=num_monomials) as pbar:
        for _ in pbar:
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

                fra_eigval = get_fra_eigval(data_covar_eigvals, left_monomial, eval_level_coeff)
                heapq.heappush(pq, (-fra_eigval, left_monomial))

            if monomial.degree() < kmax:
                right_monomial = monomial.copy()
                right_monomial[last_idx] += 1
                if eval_level_coeff(right_monomial.degree()) < 1e-12:
                    right_monomial[last_idx] += 1
                if right_monomial.degree() > kmax:
                    continue
                fra_eigval = get_fra_eigval(data_covar_eigvals, right_monomial, eval_level_coeff)
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
        fra_eigvals[i] = get_fra_eigval(data_eigvals, monomial, eval_level_coeff)
    return fra_eigvals


def lookup_monomial_idx(monomials, monomial):
    return next((i for i, m in enumerate(monomials) if m == monomial), None)


def group_by_deg_max(monomials):
    groups = {}
    for i, m in enumerate(monomials):
        key = (m.degree(), m.max_degree())
        if key not in groups:
            groups[key] = []
        groups[key].append((i, m))
    return groups

def stratified_sample_monomials(
    monomials,
    n,
    m=0,
    return_indices=False,
    np_rng=None,
    torch_gen=None
):
    """
    Per stratum (degree, max_degree):
      - (0,0): always return its single element (ignores n, m).
      - Otherwise: include first m, and sample (n-m) from the rest without replacement.
    """
    if np_rng is None:
        np_rng = np.random.default_rng()

    use_torch = torch_gen is not None

    groups = group_by_deg_max(monomials)
    out = {}

    for key, items in groups.items():
        # Special-case the constant term stratum
        if key == (0, 0):
            chosen = items[:1]  # exactly one element exists
            out[key] = [i for (i, _) in chosen] if return_indices else [m for (_, m) in chosen]
            continue

        L = len(items)
        if L < n:
            raise ValueError(f"Group {key} has {L} items, needs at least n={n}.")
        m_eff = min(m, n)
        prefix = items[:m_eff]

        k = n - m_eff
        if k > 0:
            rem = items[m_eff:]
            R = len(rem)
            if R < k:
                raise ValueError(f"Group {key}: only {R} available after first m={m_eff}, needs {k} more.")
            if use_torch:
                import torch
                perm = torch.randperm(R, generator=torch_gen).tolist()
                idxs = perm[:k]
            else:
                idxs = np_rng.choice(R, size=k, replace=False).tolist()
            chosen = prefix + [rem[i] for i in idxs]
        else:
            chosen = prefix

        out[key] = [i for (i, _) in chosen] if return_indices else [m for (_, m) in chosen]

    return out
