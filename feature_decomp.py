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


def group_by_deg_max(monomials, stop_at_degree=None, assume_sorted=False):
    groups = {}
    for i, m in enumerate(monomials):
        deg = m.degree()
        if stop_at_degree is not None and deg > stop_at_degree:
            if assume_sorted:
                break
            continue
        key = (deg, m.max_degree())
        if key not in groups:
            groups[key] = []
        groups[key].append((i, m))
    return groups

def _make_weights(R, mode="exp", tau=0.5):
    """
    Produce weights w[0..R-1] where smaller index => larger weight.
    mode:
      - "exp": w_i ∝ exp(-(i)/tau)  (strong early preference as tau ↓)
      - "linear": w_i ∝ (R - i)
    """
    if R <= 0:
        return np.array([], dtype=float)
    if mode == "linear":
        w = (R - np.arange(R)).astype(float)
    else:  # "exp" (default)
        w = np.exp(-np.arange(R, dtype=float) / max(tau, 1e-8))
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        # safety fallback to uniform
        w = np.ones(R, dtype=float) / R
    else:
        w /= s
    return w

def stratified_sample_monomials(monomials, n, m=0, return_indices=False, np_rng=None, stop_at_degree=None,
                                assume_sorted=False, weight_mode="exp", tau=0.5,):
    """
    For each (degree, max_degree) stratum:
      • Include first m (deterministic, capped by target size).
      • Fill the remaining (target - m) by weighted sampling WITHOUT replacement,
        upweighting earlier items in the remainder (lower index = higher prob).
      • If stratum has < n items, return all of them.

    Reproducibility:
      - Pass np_rng = np.random.default_rng(seed) OR
      - Pass torch_gen = torch.Generator().manual_seed(seed) (takes precedence).
    """
    if np_rng is None:
        np_rng = np.random.default_rng()


    groups = group_by_deg_max(monomials, stop_at_degree=stop_at_degree, assume_sorted=assume_sorted)
    out = {}

    for key, items in groups.items():
        L = len(items)
        if L == 0:
            continue
        target = min(n, L)

        # Deterministic prefix (first m)
        m_eff = min(m, target)
        prefix = items[:m_eff]

        k = target - m_eff
        if k > 0:
            rem = items[m_eff:]
            R = len(rem)
            # weights favor earlier items in 'rem'
            w = _make_weights(R, mode=weight_mode, tau=tau)
            idxs = np_rng.choice(R, size=k, replace=False, p=w).tolist()
            chosen = prefix + [rem[i] for i in idxs]
        else:
            chosen = prefix

        out[key] = [i for (i, _) in chosen] if return_indices else [mm for (_, mm) in chosen]

    return out
