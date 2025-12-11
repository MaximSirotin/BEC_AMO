import matplotlib.pyplot as plt
import numpy as np
from qutip import destroy, qeye, basis, tensor, expect, thermal_dm, displace, entropy_vn
from qutip import coherent, ptrace, expect, ket2dm

def bogoliubov_omega(k, c=1.0, xi=1.0):
    """
    Simple dimensionless Bogoliubov dispersion:
        ω(k) = c * k * sqrt(1 + 0.5 * (k*xi)^2)
    """
    k = np.asarray(k, dtype=float)
    return c * k * np.sqrt(1.0 + 0.5 * (k * xi)**2)

def group_velocity(k, c=1.0, xi=1.0, dk=1e-3):
    """
    Numerical group velocity v_g = dω/dk using central finite difference.
    """
    k = np.asarray(k, dtype=float)
    return (bogoliubov_omega(k + dk, c, xi) - bogoliubov_omega(k - dk, c, xi)) / (2.0 * dk)

def mode_centers_from_dispersion(k_vals, t_emit,
                                 v_flow_out=0.0,
                                 v_flow_in=-1.5,
                                 c=1.0, xi=1.0):
    """
    Given a set of mode wavenumbers k_vals (one per Hawking pair),
    return outside and inside centers x_Ej(t), x_Pj(t) at time t_emit,
    including background flow plus Bogoliubov group velocities.
    """
    k_vals = np.asarray(k_vals, dtype=float)
    M = len(k_vals)

    v_g = group_velocity(k_vals, c=c, xi=xi)

    # Outside: flow + group velocity
    x_out_centers = (v_flow_out + v_g) * t_emit

    # Inside: flow + (possibly different) group velocity.
    # For a simple toy, use same |v_g| but negative direction.
    x_in_centers = (v_flow_in - v_g) * t_emit

    return x_out_centers, x_in_centers

def build_mode_functions(xs, x_out_centers, x_in_centers, sigma=0.5):
    """
    Build normalized Gaussian mode functions fE[j,x], fP[j,x]
    centered at x_out_centers[j], x_in_centers[j].
    """
    xs = np.asarray(xs, dtype=float)
    M = len(x_out_centers)
    Nx = len(xs)
    dx = xs[1] - xs[0]

    def gaussian(x, x0, s):
        return np.exp(-(x - x0)**2 / (2.0 * s**2))

    fE = np.zeros((M, Nx), dtype=complex)
    fP = np.zeros((M, Nx), dtype=complex)

    for j in range(M):
        x0_out = x_out_centers[j]
        x0_in  = x_in_centers[j]

        fE[j, :] = gaussian(xs, x0_out, sigma)
        fP[j, :] = gaussian(xs, x0_in,  sigma)

        # L2 normalize on the grid
        normE = np.sqrt(np.sum(np.abs(fE[j, :])**2) * dx)
        normP = np.sqrt(np.sum(np.abs(fP[j, :])**2) * dx)
        if normE > 0:
            fE[j, :] /= normE
        if normP > 0:
            fP[j, :] /= normP

    return fE, fP

def analytic_G2(xs, fE, fP, r, alpha_P, S_E=None, S_P=None): 
    """
    Analytic G^2(x,x') for independent two-mode squeezed + coherent-seeded pairs,
    including optional Bogoliubov density-response factors S_E, S_P.

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid. Only used for shape consistency.
    fE, fP : (M, Nx) complex arrays
        Mode functions f_{E j}(x), f_{P j}(x) evaluated on xs.
        These should already be normalized with sum_x |f|^2 dx = 1.
    r : float or (M,) array
        Two-mode squeezing parameter(s) for each pair.
    alpha_P : (M,) complex array
        Coherent seeds in the P modes: initial |0_E, alpha_P_j_P> for each j.
    S_E, S_P : (M,) arrays or None
        Static structure factors S_E(k_j), S_P(k_j) for density response
        of E and P modes. If None, they are set to 1 (no Bogoliubov dressing).

    Returns
    -------
    G2 : (Nx, Nx) array (real)
        Analytic G^2(x,x') = <δn(x) δn(x')> on the grid xs.
    """

    xs = np.asarray(xs)
    fE = np.asarray(fE)
    fP = np.asarray(fP)
    alpha_P = np.asarray(alpha_P, dtype=complex)

    M, Nx = fE.shape
    assert fP.shape == (M, Nx)

    # Broadcast r to per-mode array
    r = np.asarray(r, dtype=float)
    if r.ndim == 0:
        r = np.full(M, float(r))

    # Bogoliubov static structure factors
    if S_E is None:
        S_E = np.ones(M, dtype=float)
    else:
        S_E = np.asarray(S_E, dtype=float)

    if S_P is None:
        S_P = np.ones(M, dtype=float)
    else:
        S_P = np.asarray(S_P, dtype=float)

    # Sqrt factors for the cross term (δn_E ∝ sqrt(S_E) X_E, δn_P ∝ sqrt(S_P) X_P)
    S_EP = np.sqrt(S_E * S_P)

    c = np.cosh(r)
    s = np.sinh(r)

    abs2 = np.abs(alpha_P)**2
    Re_alpha2 = np.real(alpha_P**2)

    # Number expectations
    nE = s**2 * (abs2 + 1.0)
    nP = c**2 * abs2 + s**2

    # Quadrature second moments X = a + a†
    XE2  = 2.0 * (s**2 * Re_alpha2) + 2.0 * nE + 1.0
    XP2  = 2.0 * (c**2 * Re_alpha2) + 2.0 * nP + 1.0

    # Cross term (negative for TMSV)
    XEXP = -2.0 * c * s * (1.0 + abs2 + Re_alpha2)

    # Build G2(x,x') as sum over pairs, with Bogoliubov factors
    G2 = np.zeros((Nx, Nx), dtype=complex)

    for j in range(M):
        fE_j = fE[j, :]   # (Nx,)
        fP_j = fP[j, :]   # (Nx,)

        EE = np.outer(fE_j, fE_j)
        PP = np.outer(fP_j, fP_j)
        EP = np.outer(fE_j, fP_j)
        PE = np.outer(fP_j, fE_j)

        # EE: outside-outside, weighted by S_E
        G2 += S_E[j]  * XE2[j]  * EE

        # PP: inside-inside, weighted by S_P
        G2 += S_P[j]  * XP2[j]  * PP

        # EP+PE: cross-horizon, weighted by sqrt(S_E S_P)
        G2 += S_EP[j] * XEXP[j] * (EP + PE)

    return G2.real

def leakage_from_G2(
    G2_0,
    G2_1,
    xs,
    horizon_x=0.0,
    reg=1e-6,
):
    """
    Estimate information leakage to the outside from two G^2(x,x') maps
    (e.g. unseeded vs seeded interior), assuming everything is Gaussian.

    Parameters
    ----------
    G2_0, G2_1 : 2D np.ndarray (Nx x Nx)
        G^2(x,x') arrays for two different interior preparations.
        Example: G2_0 = BH with vacuum interior, G2_1 = BH with seeded interior.
    xs : 1D np.ndarray (Nx,)
        Spatial grid corresponding to indices of G2.
    horizon_x : float
        Position of the horizon; points with x > horizon_x are treated as "outside".
    reg : float
        Small diagonal regularization added to covariance matrices for stability.

    Returns
    -------
    info_bits : float
        Symmetric information-like measure (in bits) of how distinguishable
        the two outside G^2 patterns are. Roughly: upper bound on how much
        classical information about A can be extracted from outside density
        measurements.
    D_kl_0_to_1, D_kl_1_to_0 : float
        One-way KL divergences (in nats) between the two Gaussian distributions
        over outside fluctuations:
            D_kl_0_to_1 = KL( P0 || P1 )
            D_kl_1_to_0 = KL( P1 || P0 )
    """

    G2_0 = np.asarray(G2_0, dtype=float)
    G2_1 = np.asarray(G2_1, dtype=float)
    xs = np.asarray(xs, dtype=float)

    Nx = len(xs)
    assert G2_0.shape == (Nx, Nx)
    assert G2_1.shape == (Nx, Nx)

    # 1) Pick OUTSIDE indices (x > horizon_x)
    outside_mask = xs > horizon_x
    idx_out = np.where(outside_mask)[0]
    if len(idx_out) == 0:
        raise ValueError("No outside points selected (check horizon_x vs xs).")

    # Extract outside-outside blocks: these are the covariances
    # of the measured outside density fluctuations δn(x_i), i in outside.
    Sigma0 = G2_0[np.ix_(idx_out, idx_out)].copy()
    Sigma1 = G2_1[np.ix_(idx_out, idx_out)].copy()

    # 2) Regularize for numerical stability
    n_out = Sigma0.shape[0]
    Sigma0 += reg * np.eye(n_out)
    Sigma1 += reg * np.eye(n_out)

    # 3) Compute KL divergences between zero-mean multivariate Gaussians
    #    P0 ~ N(0, Sigma0), P1 ~ N(0, Sigma1)
    #
    #    D_KL(P0 || P1) = 0.5 * [ tr(Sigma1^{-1} Sigma0) - n + ln(det(Sigma1)/det(Sigma0)) ]
    #
    # We use symmetric KL (or JS-like) as a leakage measure.

    # Cholesky + logdet for stability
    def logdet_sym_posdef(S):
        L = np.linalg.cholesky(S)
        return 2.0 * np.sum(np.log(np.diag(L)))

    # Solve for inverse via linear solves rather than explicit inverse
    def solve_for_inv(S, I):
        # Returns S^{-1} I = S^{-1}
        return np.linalg.solve(S, I)

    I = np.eye(n_out)

    # Sigma1^{-1} Sigma0
    Sigma1_inv = solve_for_inv(Sigma1, I)
    Sigma0_inv = solve_for_inv(Sigma0, I)

    tr_10 = np.trace(Sigma1_inv @ Sigma0)
    tr_01 = np.trace(Sigma0_inv @ Sigma1)

    logdet0 = logdet_sym_posdef(Sigma0)
    logdet1 = logdet_sym_posdef(Sigma1)

    D_kl_0_to_1 = 0.5 * (tr_10 - n_out + (logdet1 - logdet0))
    D_kl_1_to_0 = 0.5 * (tr_01 - n_out + (logdet0 - logdet1))

    # 4) Symmetric information-like measure.
    #
    # We can use the symmetrized KL as a proxy, or better:
    # Jensen-Shannon-like divergence between the two Gaussians.
    #
    # For simplicity here, we use the average KL (in nats), and convert to bits.

    D_sym_nats = 0.5 * (D_kl_0_to_1 + D_kl_1_to_0)

    # Interpret D_sym_nats / ln(2) as "info bits" ~ how many bits of
    # classical information about the seed can be obtained from the outside
    # G^2 pattern (in the pixel basis).
    info_bits = D_sym_nats / np.log(2.0)

    return info_bits, D_kl_0_to_1, D_kl_1_to_0

def leakage_from_G2_with_Holevo(
    G2_0,
    G2_1,
    xs,
    horizon_x=0.0,
    reg=1e-6,
):
    """
    Estimate information leakage from two G^2(x,x') maps via:
    - Symmetric KL divergence (classical distinguishability)
    - Gaussian Holevo information (quantum distinguishability)

    Parameters
    ----------
    G2_0, G2_1 : (Nx, Nx) arrays
        Full G^2 maps for two different interior preparations.
    xs : (Nx,) array
        Spatial grid.
    horizon_x : float
        x > horizon_x are treated as outside.
    reg : float
        Small regularization for stability.

    Returns
    -------
    info_bits_KL : float
        Symmetric KL-based leakage (bits).
    D_kl_0_to_1, D_kl_1_to_0 : float
        KL divergences KL(P0||P1), KL(P1||P0) (nats).
    chi_bits : float
        Holevo information χ (bits): quantum info leakage.
    """

    G2_0 = np.asarray(G2_0, float)
    G2_1 = np.asarray(G2_1, float)
    xs = np.asarray(xs, float)

    Nx = len(xs)
    outside = xs > horizon_x
    idx = np.where(outside)[0]

    if len(idx) == 0:
        raise ValueError("No outside points selected.")

    # Extract outside-outside covariances
    Sigma0 = G2_0[np.ix_(idx, idx)].copy()
    Sigma1 = G2_1[np.ix_(idx, idx)].copy()

    # Regularization
    n = Sigma0.shape[0]
    Sigma0 += reg * np.eye(n)
    Sigma1 += reg * np.eye(n)

    # KL divergence computations (classical)
    def logdet_pd(A):
        L = np.linalg.cholesky(A)
        return 2*np.sum(np.log(np.diag(L)))

    inv1 = np.linalg.solve(Sigma1, np.eye(n))
    inv0 = np.linalg.solve(Sigma0, np.eye(n))

    tr_10 = np.trace(inv1 @ Sigma0)
    tr_01 = np.trace(inv0 @ Sigma1)

    logdet0 = logdet_pd(Sigma0)
    logdet1 = logdet_pd(Sigma1)

    D_kl_0_to_1 = 0.5 * (tr_10 - n + (logdet1 - logdet0))
    D_kl_1_to_0 = 0.5 * (tr_01 - n + (logdet0 - logdet1))

    D_sym = 0.5 * (D_kl_0_to_1 + D_kl_1_to_0)
    info_bits_KL = D_sym / np.log(2)

    # --------------------------
    #     HOLEVO INFORMATION
    # --------------------------
    #
    # Treat Sigma_i as classical covariances of outcomes of X=a+a†,
    # so quantum covariance matrices are V_i = Sigma_i / 2.
    #

    V0 = 0.5 * Sigma0
    V1 = 0.5 * Sigma1
    Vbar = 0.5 * (V0 + V1)

    # Von Neumann entropy of an n-mode Gaussian state is:
    # S(V) = sum_k f(ν_k), where ν_k are symplectic eigenvalues of V
    # and
    # f(ν) = (ν+1/2)log(ν+1/2) - (ν-1/2)log(ν-1/2).
    #
    # For classical covariance-only measurement, V_i are diagonal
    # in quadrature basis, so we treat them as 1D independent modes.
    # (Full symplectic diagonalization included for generality.)

    def gaussian_entropy(V):
        # Compute symplectic eigenvalues
        # General formula: eigenvalues of |iΩ V|
        # Construct Ω for n modes:
        n = V.shape[0]
        # If treating each pixel as *one* quadrature: no conjugate P-quadrature.
        # So entropy is simply:
        # S = sum_i f(ν_i) with ν_i = V_ii  (treat as independent 1D modes)
        # This is the consistent way given the measurement model.
        nu = np.sqrt(np.maximum(V.diagonal()**2, 0))  # effectively |X| variances
        # convert quadrature variance into 'ν = sqrt(det V_mode)'.
        # For a single quadrature measurement we treat it as classical mode:
        def f(nu):
            return (nu+0.5)*np.log(nu+0.5) - (nu-0.5)*np.log(nu-0.5)
        return np.sum(f(nu))

    S0 = gaussian_entropy(V0)
    S1 = gaussian_entropy(V1)
    Sbar = gaussian_entropy(Vbar)

    chi_nats = Sbar - 0.5*(S0+S1)
    chi_bits = chi_nats/np.log(2)

    return info_bits_KL, D_kl_0_to_1, D_kl_1_to_0, chi_bits

def holevo_single_pair_fock(r, alpha0, alpha1, N_cut=5):
    """
    True quantum Holevo χ (in bits) for a single Hawking pair (E,P)
    in truncated Fock space, with coherent seeds on P and two-mode
    squeezing S(r).

    Input:
      r      : squeezing parameter (float)
      alpha0 : complex amplitude for interior seed in case 0
      alpha1 : complex amplitude for interior seed in case 1
      N_cut  : Fock cutoff for each mode (dimension N_cut)

    Returns:
      chi_bits : Holevo information (bits) accessible from the outside mode E.
    """

    # 1) Operators
    aE = destroy(N_cut)
    aP = destroy(N_cut)
    I  = qeye(N_cut)

    # Two-mode space
    aE_full = tensor(aE, I)
    aP_full = tensor(I, aP)

    # 2) Two-mode squeezing unitary S(r)
    H_sq = r * (aE_full * aP_full - aE_full.dag() * aP_full.dag())
    S2   = H_sq.expm()   # S(r) = exp[r(aE aP - aE† aP†)]

    # 3) Input states |ψ_i^in> = |0>_E ⊗ |alpha_i>_P
    vac_E = basis(N_cut, 0)
    psiP0 = coherent(N_cut, alpha0)
    psiP1 = coherent(N_cut, alpha1)

    psi_in_0 = tensor(vac_E, psiP0)
    psi_in_1 = tensor(vac_E, psiP1)

    # 4) Output pure states after Hawking map
    psi_out_0 = S2 * psi_in_0
    psi_out_1 = S2 * psi_in_1

    rho_out_0 = psi_out_0 * psi_out_0.dag()
    rho_out_1 = psi_out_1 * psi_out_1.dag()

    # 5) Reduce to outside mode E: ρ^E_i = Tr_P ρ_out_i
    rhoE0 = ptrace(rho_out_0, 0)   # subsystem 0 = E
    rhoE1 = ptrace(rho_out_1, 0)

    # 6) Holevo χ = S((ρE0+ρE1)/2) - (S(ρE0)+S(ρE1))/2  (nats -> bits)
    rho_avg = 0.5 * (rhoE0 + rhoE1)

    S0 = entropy_vn(rhoE0, base=np.e)     # nats
    S1 = entropy_vn(rhoE1, base=np.e)
    S_avg = entropy_vn(rho_avg, base=np.e)

    chi_nats = S_avg - 0.5 * (S0 + S1)
    chi_bits = chi_nats / np.log(2.0)
    return chi_bits

def moments_from_G2(G2, xs, fE, fP):
    """
    Reconstruct effective XE2_j, XP2_j, XEXP_j from G2 via mode projections.
    G2 : (Nx,Nx) real (or complex but physically real)
    xs : (Nx,)
    fE,fP : (M,Nx) complex arrays of mode functions
    """
    G2 = np.asarray(G2)
    xs = np.asarray(xs)
    dx = xs[1] - xs[0]

    M, Nx = fE.shape

    XE2 = np.zeros(M, dtype=float)
    XP2 = np.zeros(M, dtype=float)
    XEXP = np.zeros(M, dtype=float)

    for j in range(M):
        fEj = fE[j, :]
        fPj = fP[j, :]

        # Note the conjugate on the left (vdot does conj on first argument)
        XE2_j  = np.vdot(fEj, G2 @ fEj) * dx * dx
        XP2_j  = np.vdot(fPj, G2 @ fPj) * dx * dx
        XEXP_j = np.vdot(fEj, G2 @ fPj) * dx * dx

        XE2[j]  = XE2_j.real
        XP2[j]  = XP2_j.real
        XEXP[j] = XEXP_j.real

    return XE2, XP2, XEXP

def V_pair_from_moments(XE2, XP2, XEXP):
    """
    Build a 4x4 effective CM for one pair in basis (qE,pE,qP,pP)
    given XE2=<X_E^2>, XP2=<X_P^2>, XEXP=<X_E X_P>.
    """
    V = np.zeros((4,4), dtype=float)

    V[0,0] = 0.5 * XE2   # <qE^2>
    V[1,1] = 0.5 * XE2   # <pE^2>
    V[2,2] = 0.5 * XP2   # <qP^2>
    V[3,3] = 0.5 * XP2   # <pP^2>

    C = 0.5 * XEXP

    V[0,2] = V[2,0] = C  # <qE qP>
    V[1,3] = V[3,1] = -C # <pE pP>

    return V

def V_E_from_pair(V_pair):
    # V_pair in basis (qE,pE,qP,pP)
    return V_pair[:2, :2]

def chi_gaussian_from_CMs(VE0, VE1):
    Vavg = 0.5*(VE0 + VE1)
    S0 = gaussian_entropy_from_CM(VE0)
    S1 = gaussian_entropy_from_CM(VE1)
    Savg = gaussian_entropy_from_CM(Vavg)
    chi_nats = Savg - 0.5*(S0 + S1)
    return chi_nats / np.log(2.0)   # bits

def chi_from_G2_pair(G2_0, G2_1, xs, fE_j, fP_j):
    """
    Approximate per-pair Gaussian Holevo χ (bits)
    reconstructed from G2 (vac vs seed) for a single pair j.
    """

    # Wrap fE,fP as shape (1,Nx) so we can reuse moments_from_G2
    fE = fE_j[None, :]
    fP = fP_j[None, :]

    XE2_0, XP2_0, XEXP_0 = moments_from_G2(G2_0, xs, fE, fP)
    XE2_1, XP2_1, XEXP_1 = moments_from_G2(G2_1, xs, fE, fP)

    V_pair_0 = V_pair_from_moments(XE2_0[0], XP2_0[0], XEXP_0[0])
    V_pair_1 = V_pair_from_moments(XE2_1[0], XP2_1[0], XEXP_1[0])

    VE0 = V_E_from_pair(V_pair_0)
    VE1 = V_E_from_pair(V_pair_1)

    chi_bits = chi_gaussian_from_CMs(VE0, VE1)
    return chi_bits

def mean_and_conn_from_G2(G2, mu, xs, fE, fP):
    """
    Mean-aware extraction of per-mode moments from G2 and mean profile mu(x).

    Parameters
    ----------
    G2 : (Nx,Nx) array
        < δn(x) δn(x') > for a given case (vac or seed).
    mu : (Nx,) array
        < δn(x) > for that case.
    xs : (Nx,) array
        Spatial grid.
    fE, fP : (M,Nx) complex arrays
        Mode functions for E and P modes.

    Returns
    -------
    XE_mean, XP_mean : (M,) arrays
        <X_Ej>, <X_Pj> per mode.
    XE2c, XP2c, XEXPc : (M,) arrays
        Connected second moments for X quadratures:
        XE2c = <X_E^2>_conn, etc.
    """
    G2 = np.asarray(G2)
    mu = np.asarray(mu)
    xs = np.asarray(xs)
    dx = xs[1] - xs[0]

    M, Nx = fE.shape

    # 1) Connected correlator in x-space
    C2 = G2 - np.outer(mu, mu)

    XE_mean = np.zeros(M, dtype=float)
    XP_mean = np.zeros(M, dtype=float)
    XE2c    = np.zeros(M, dtype=float)
    XP2c    = np.zeros(M, dtype=float)
    XEXPc   = np.zeros(M, dtype=float)

    for j in range(M):
        fEj = fE[j, :]
        fPj = fP[j, :]

        # <X_E> = ∫ fE*(x) mu(x) dx
        XE_mean_j = np.vdot(fEj, mu) * dx
        XP_mean_j = np.vdot(fPj, mu) * dx

        # connected second moments from C2
        XE2c_j  = np.vdot(fEj, C2 @ fEj) * dx * dx
        XP2c_j  = np.vdot(fPj, C2 @ fPj) * dx * dx
        XEXPc_j = np.vdot(fEj, C2 @ fPj) * dx * dx

        XE_mean[j] = XE_mean_j.real
        XP_mean[j] = XP_mean_j.real
        XE2c[j]    = XE2c_j.real
        XP2c[j]    = XP2c_j.real
        XEXPc[j]   = XEXPc_j.real

    return XE_mean, XP_mean, XE2c, XP2c, XEXPc

def V_and_d_pair_from_moments(XE_mean, XP_mean, XE2c, XP2c, XEXPc,
                              assume_real_seed=True):
    """
    Build a 4x4 covariance matrix V and 4-dim mean vector d for one pair
    in basis (qE,pE,qP,pP), from mean-aware X-moments.

    XE_mean,XP_mean : <X_E>, <X_P>
    XE2c,XP2c,XEXPc : connected second moments of X quadratures
                      (i.e. Var(X_E), Var(X_P), Cov_conn(XE,XP))

    assume_real_seed : if True, set p-means to 0 and put all displacement in q.
    """

    V = np.zeros((4, 4), dtype=float)
    d = np.zeros(4, dtype=float)

    # Means: X = sqrt(2) q  (if seed real, we take <p>=0)
    if assume_real_seed:
        qE_mean = XE_mean / np.sqrt(2.0)
        qP_mean = XP_mean / np.sqrt(2.0)
        pE_mean = 0.0
        pP_mean = 0.0
    else:
        qE_mean = XE_mean / np.sqrt(2.0)
        qP_mean = XP_mean / np.sqrt(2.0)
        pE_mean = 0.0
        pP_mean = 0.0

    d[0] = qE_mean
    d[1] = pE_mean
    d[2] = qP_mean
    d[3] = pP_mean

    # Connected second moments:
    # Var(X) = XE2c etc.  Since X = sqrt(2) q, Var(q) = Var(X)/2
    V_qE_qE = 0.5 * XE2c
    V_qP_qP = 0.5 * XP2c
    V_qE_qP = 0.5 * XEXPc

    # Assume no single-mode squeezing: Var(p)=Var(q)
    # and Hawking-like cross-correlation structure: Cov(pE,pP) = -Cov(qE,qP)
    V[0,0] = V_qE_qE
    V[1,1] = V_qE_qE
    V[2,2] = V_qP_qP
    V[3,3] = V_qP_qP

    V[0,2] = V[2,0] = V_qE_qP
    V[1,3] = V[3,1] = -V_qE_qP

    return V, d

def symplectic_omega(n_modes):
    I = np.eye(n_modes)
    J = np.array([[0, 1], [-1, 0]])
    return np.kron(I, J)

def chi_mean_aware_gaussian(V0, d0, V1, d1):
    """
    Mean-aware Gaussianized Holevo approximation (bits):

    χ_G ≈ S(V_mix) - 0.5(S(V0) + S(V1)),
    with
      V_mix = 0.5(V0+V1) + 0.25 Δd Δd^T.
    """
    d0 = d0.reshape(-1,1)
    d1 = d1.reshape(-1,1)
    dd   = d1 - d0

    V0_phys = make_physical_CM(V0)
    V1_phys = make_physical_CM(V1)

    Vmix = 0.5*(V0_phys + V1_phys) + 0.25*(dd @ dd.T)

    S0   = gaussian_entropy_from_CM(V0_phys)
    S1   = gaussian_entropy_from_CM(V1_phys)
    Smix = gaussian_entropy_from_CM(Vmix)

    chi_nats = Smix - 0.5*(S0 + S1)
    chi_bits = chi_nats / np.log(2.0)
    return chi_bits

def chi_mean_aware_from_G2_pair(G2_0, mu_0, G2_1, mu_1, xs, fE_j, fP_j):
    """
    Mean-aware Gaussian χ estimate (bits) for one pair j,
    reconstructed from G2 and mu for vac vs seed.
    """

    # shape them as (1,Nx) to reuse mean_and_conn
    fE = fE_j[None, :]
    fP = fP_j[None, :]

    XE_m0, XP_m0, XE2c0, XP2c0, XEXPc0 = mean_and_conn_from_G2(G2_0, mu_0, xs, fE, fP)
    XE_m1, XP_m1, XE2c1, XP2c1, XEXPc1 = mean_and_conn_from_G2(G2_1, mu_1, xs, fE, fP)

    V0, d0 = V_and_d_pair_from_moments(XE_m0[0], XP_m0[0],
                                       XE2c0[0], XP2c0[0], XEXPc0[0])
    V1, d1 = V_and_d_pair_from_moments(XE_m1[0], XP_m1[0],
                                       XE2c1[0], XP2c1[0], XEXPc1[0])

    chi_bits = chi_mean_aware_gaussian(V0, d0, V1, d1)
    return chi_bits

def analytic_mu(xs, fE, fP, r, alpha_P):
    """
    Compute μ(x) = <δn(x)> for the analytic model with
    two-mode squeezing S(r) and coherent seeds alpha_P in P modes.

    Parameters
    ----------
    xs      : (Nx,) grid (not explicitly used except for length)
    fE,fP   : (M,Nx) mode functions for E and P (complex)
    r       : squeezing parameter (float)
    alpha_P : (M,) array of complex seeding amplitudes in P

    Returns
    -------
    mu : (Nx,) array of <δn(x)>.
    """
    xs = np.asarray(xs)
    alpha_P = np.asarray(alpha_P, dtype=complex)
    M, Nx = fE.shape

    c = np.cosh(r)
    s = np.sinh(r)

    # Per-mode expectations after squeezing
    aE_mean = s * np.conjugate(alpha_P)   # <a_Ej>
    aP_mean = c * alpha_P                 # <a_Pj>

    mu = np.zeros(Nx, dtype=float)

    for ix in range(Nx):
        acc = 0+0j
        for j in range(M):
            acc += fE[j, ix]*aE_mean[j] + fP[j, ix]*aP_mean[j]
        mu[ix] = 2.0 * np.real(acc)

    return mu

def symplectic_omega(n_modes):
    I = np.eye(n_modes)
    J = np.array([[0, 1], [-1, 0]])
    return np.kron(I, J)

def symplectic_eigenvalues_raw(V):
    """
    Raw symplectic eigenvalues of V (no clamping).
    """
    n2 = V.shape[0]
    n = n2 // 2
    Omega = symplectic_omega(n)
    eigvals = np.linalg.eigvals(1j * Omega @ V)
    # eigenvalues come in ±ν pairs, take absolute value and sort
    nu = np.sort(np.real(np.abs(eigvals)))[::2]
    return nu

def make_physical_CM(V, eps=1e-6):
    """
    Adjust V minimally (add isotropic noise) so that it becomes a
    valid covariance matrix, i.e. all symplectic eigenvalues >= 1/2.
    """
    V = 0.5 * (V + V.T)   # enforce symmetry
    nu = symplectic_eigenvalues_raw(V)
    min_nu = np.min(nu)

    if min_nu < 0.5 + eps:
        # add isotropic noise: V -> V + delta * I
        delta = (0.5 + eps) - min_nu
        V = V + delta * np.eye(V.shape[0])
    return V

def symplectic_eigenvalues(V):
    """
    Physical symplectic eigenvalues (clamped to >= 1/2).
    """
    nu_raw = symplectic_eigenvalues_raw(V)
    # enforce ν >= 1/2
    nu = np.maximum(nu_raw, 0.5*(1.0 + 1e-9))
    return nu

def gaussian_entropy_from_CM(V):
    """
    Von Neumann entropy of a Gaussian state with CM V, in nats.
    Safely handles small numerical violations of the uncertainty bound.
    """
    V_phys = make_physical_CM(V)
    nu = symplectic_eigenvalues(V_phys)

    def f(nu_k):
        # both args inside log are strictly > 0 now
        return (nu_k + 0.5)*np.log(nu_k + 0.5) - (nu_k - 0.5)*np.log(nu_k - 0.5)

    return np.sum([f(v) for v in nu])
# ============================================================
# 1) True Holevo χ for a single Hawking pair (E,P)
# ============================================================

def holevo_single_pair_fock(r, alpha0, alpha1, N_cut=5):
    """
    True quantum Holevo χ (in bits) for a single Hawking pair (E,P) with:
      input states: |0>_E ⊗ |alpha_i>_P,  i = 0,1
      channel:      two-mode squeezer S(r)
      output:       reduced outside states ρ_E0, ρ_E1.

    Parameters
    ----------
    r      : float
        Squeezing parameter.
    alpha0 : complex
        Coherent amplitude in P for case 0 (usually 0).
    alpha1 : complex
        Coherent amplitude in P for case 1 (seed).
    N_cut  : int
        Fock cutoff per mode.

    Returns
    -------
    chi_bits : float
        Holevo information χ in bits.
    """
    # Local operators
    aE = destroy(N_cut)
    aP = destroy(N_cut)
    I  = qeye(N_cut)

    aE_full = tensor(aE, I)
    aP_full = tensor(I, aP)

    # Two-mode squeezing unitary
    H_sq = r * (aE_full * aP_full - aE_full.dag() * aP_full.dag())
    S2   = H_sq.expm()

    # Input states
    vac_E = basis(N_cut, 0)
    psiP0 = coherent(N_cut, alpha0)
    psiP1 = coherent(N_cut, alpha1)

    psi_in_0 = tensor(vac_E, psiP0)
    psi_in_1 = tensor(vac_E, psiP1)

    # Output pure states
    psi_out_0 = S2 * psi_in_0
    psi_out_1 = S2 * psi_in_1

    rho_out_0 = psi_out_0 * psi_out_0.dag()
    rho_out_1 = psi_out_1 * psi_out_1.dag()

    # Reduce to outside mode E
    rhoE0 = ptrace(rho_out_0, 0)
    rhoE1 = ptrace(rho_out_1, 0)

    # Holevo χ = S((ρE0+ρE1)/2) - (S(ρE0)+S(ρE1))/2
    rho_avg = 0.5 * (rhoE0 + rhoE1)

    S0   = entropy_vn(rhoE0, base=np.e)   # nats
    S1   = entropy_vn(rhoE1, base=np.e)
    Savg = entropy_vn(rho_avg, base=np.e)

    chi_nats = Savg - 0.5 * (S0 + S1)
    chi_bits = chi_nats / np.log(2.0)
    return chi_bits

# ============================================================
# 2) Estimate μ(x) from ΔG² and project onto a single BdG pair
# ============================================================

def estimate_mu_from_deltaG2(delta_G2):
    """
    Estimate μ(x) from ΔG2(x,x') ≈ μ(x) μ(x') by taking
    the leading eigenvector of ΔG2.

    Parameters
    ----------
    delta_G2 : (Nx,Nx) array
        G2_seed - G2_vac

    Returns
    -------
    mu_est : (Nx,) array
        Estimated mean profile μ(x) up to an overall sign.
    """
    # Symmetrize for safety
    A = 0.5 * (delta_G2 + delta_G2.T)
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(A)
    idx_max = np.argmax(eigvals)
    lam_max = eigvals[idx_max]
    v_max   = eigvecs[:, idx_max]

    if lam_max < 0:
        # No coherent contribution resolved; return zeros
        return np.zeros_like(v_max)

    mu_est = np.sqrt(lam_max) * v_max
    return mu_est

# ============================================================
# 3) Full pipeline: from G2 maps to alpha and Holevo χ
# ============================================================

def estimate_alpha_and_holevo_from_G2(
    G2_vac,
    G2_seed,
    xs,
    fE,
    fP,
    r,
    j_seed,
    N_cut=5,
    enforce_real_alpha=True,
):
    """
    Estimate the interior coherent amplitude alpha_P for a given BdG pair j_seed
    from differential G2 maps, and compute the corresponding Holevo χ per pair.

    Assumptions:
    ------------
    - The seed is a coherent state in a single partner mode P_{j_seed}.
    - Mode functions fE,fP are (approximately) orthonormal on the grid xs.
    - Dynamics are a two-mode squeezer with parameter r.
    - G2_vac has zero mean; G2_seed differs by a coherent displacement only.

    Parameters
    ----------
    G2_vac : (Nx,Nx) array
        Simulated G^2(x,x') for vacuum interior.
    G2_seed : (Nx,Nx) array
        Simulated G^2(x,x') for seeded interior.
    xs : (Nx,) array
        Spatial grid (only dx is used here).
    fE, fP : (M,Nx) arrays (complex)
        BdG mode functions for outside (E) and inside (P).
    r : float
        Squeezing parameter.
    j_seed : int
        Index of the BdG pair that is coherently seeded.
    N_cut : int
        Fock cutoff for the Holevo calculation.
    enforce_real_alpha : bool
        If True, take only the real part of the estimated alpha_P
        (common in the purely real Gaussian toy model).

    Returns
    -------
    alpha_est : complex
        Estimated coherent amplitude in P_{j_seed}.
    chi_bits  : float
        Estimated Holevo χ (bits) for the single seeded pair.
    """
    G2_vac = np.asarray(G2_vac, dtype=float)
    G2_seed = np.asarray(G2_seed, dtype=float)
    xs = np.asarray(xs, dtype=float)

    delta_G2 = G2_seed - G2_vac
    Nx = len(xs)
    dx = xs[1] - xs[0]

    # 1) Estimate μ(x) from ΔG²
    mu_est = estimate_mu_from_deltaG2(delta_G2)  # shape (Nx,)

    # 2) Project μ onto the chosen BdG partner mode
    fP_j = fP[j_seed, :]
    # Use vdot: conj(f) * mu * dx
    X_P_mean = np.vdot(fP_j, mu_est) * dx  # ≈ <X_P> = <a+a†>

    # 3) Infer alpha_P from <X_P>
    c = np.cosh(r)
    # For real alpha in the simple model: <X_P> ≈ 2 c alpha
    alpha_est = X_P_mean / (2.0 * c)

    if enforce_real_alpha:
        alpha_est = float(np.real(alpha_est))

    # 4) Compute Holevo χ from the estimated alpha via the full channel
    chi_bits = holevo_single_pair_fock(r, alpha0=0.0, alpha1=alpha_est, N_cut=N_cut)

    return alpha_est, chi_bits

def shift_mode(fn, xs, shift):
    """
    Shift a mode function fn(x) along x by 'shift' using a simple roll.
    'shift' is in the same units as xs; we round to the nearest grid point.
    """
    dx = xs[1] - xs[0]
    n_shift = int(np.round(shift / dx))
    return np.roll(fn, n_shift)

def analytic_G2_emission_smeared(xs, fE0, fP0, r, alpha_P,
                                 vE, vP,
                                 t_obs,
                                 t_min, t_max,
                                 N_emit,
                                 kE=None, kP=None,
                                 rng=None):
    """
    Build a more experiment-like G^2(x,x') by averaging over random emission times.
    
    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid.
    fE0, fP0 : (M, Nx) complex arrays
        'Base' mode functions at some reference time (e.g. emission), normalized.
    r : float or (M,) array
        Two-mode squeezing parameter(s).
    alpha_P : (M,) complex array
        Coherent seeds for P modes.
    vE, vP : float or (M,) array
        Group velocities of E and P modes (same units as xs / time).
    t_obs : float
        Observation time.
    t_min, t_max : float
        Range of possible emission times t_e (e.g. 0 to T).
    N_emit : int
        Number of emission times to sample (Monte Carlo).
    kE, kP : float or (M,) array, optional
        Carrier wavenumbers for E and P modes. If given, we add exp(i k x) phases
        to create fringes along the tongue.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    G2_avg : (Nx, Nx) array (real)
        Emission-smeared G^2(x,x') resembling experiment-like tongues + modulation.
    """
    xs = np.asarray(xs)
    fE0 = np.asarray(fE0)
    fP0 = np.asarray(fP0)
    alpha_P = np.asarray(alpha_P, dtype=complex)
    M, Nx = fE0.shape

    # Broadcast velocities
    vE = np.broadcast_to(np.asarray(vE, float), (M,))
    vP = np.broadcast_to(np.asarray(vP, float), (M,))

    # Optional carrier ks for modulation
    if kE is not None:
        kE = np.broadcast_to(np.asarray(kE, float), (M,))
    if kP is not None:
        kP = np.broadcast_to(np.asarray(kP, float), (M,))

    if rng is None:
        rng = np.random.default_rng()

    # Container for accumulated G2
    G2_accum = np.zeros((Nx, Nx), dtype=float)

    # Sample emission times
    t_es = rng.uniform(t_min, t_max, size=N_emit)

    for t_e in t_es:
        dt = t_obs - t_e

        # Build time-shifted mode functions for this emission time
        fE_shifted = np.empty_like(fE0, dtype=complex)
        fP_shifted = np.empty_like(fP0, dtype=complex)

        for j in range(M):
            # shift envelopes according to group velocities
            shiftE = vE[j] * dt
            shiftP = vP[j] * dt

            fE_env = shift_mode(fE0[j], xs, shiftE)
            fP_env = shift_mode(fP0[j], xs, shiftP)

            # add optional plane-wave modulation to produce fringes
            if kE is not None:
                fE_env = fE_env * np.exp(1j * kE[j] * xs)
            if kP is not None:
                fP_env = fP_env * np.exp(1j * kP[j] * xs)

            fE_shifted[j] = fE_env
            fP_shifted[j] = fP_env

        # original analytic_G2 on these time-shifted modes
        G2_snapshot = analytic_G2(xs, fE_shifted, fP_shifted, r, alpha_P)
        G2_accum += G2_snapshot

    # Average over emission times
    G2_avg = G2_accum / float(N_emit)
    return G2_avg

def shift_envelope(fx, xs, shift):
    """
    Non-periodic shift: f_shifted(x) = f(x - shift), zero outside the domain.
    """
    xs = np.asarray(xs)
    fx = np.asarray(fx)
    return np.interp(xs - shift, xs, fx, left=0.0, right=0.0)

def analytic_G2_envelope_emission_smeared(xs, fE0, fP0, r, alpha_P,
                                          vE, vP,
                                          t_obs,
                                          t_min, t_max,
                                          N_emit,
                                          rng=None):
    """
    Build an experiment-like G^2(x,x') by averaging over random emission times,
    propagating only the envelopes (no phase factors).

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid.
    fE0, fP0 : (M, Nx) arrays
        'Base' mode envelopes at some reference time (e.g. emission).
        These can be real or complex; only the envelope is used.
    r : float or (M,) array
        Two-mode squeezing parameters.
    alpha_P : (M,) complex array
        Coherent seeds in the P modes.
    vE, vP : float or (M,) array
        Group velocities of E and P modes (same units as xs / time).
    t_obs : float
        Observation time.
    t_min, t_max : float
        Range of possible emission times t_e.
    N_emit : int
        Number of emission times to sample (Monte Carlo).
    rng : np.random.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    G2_avg : (Nx, Nx) array (real)
        Emission-smeared, envelope-only G^2(x,x') showing a ridge/tongue
        aligned with the effective group-velocity trajectories.
    """
    xs = np.asarray(xs)
    fE0 = np.asarray(fE0)
    fP0 = np.asarray(fP0)
    alpha_P = np.asarray(alpha_P, dtype=complex)

    M, Nx = fE0.shape
    assert fP0.shape == (M, Nx)

    # Broadcast velocities to per-mode arrays
    vE = np.broadcast_to(np.asarray(vE, float), (M,))
    vP = np.broadcast_to(np.asarray(vP, float), (M,))

    if rng is None:
        rng = np.random.default_rng()

    G2_accum = np.zeros((Nx, Nx), dtype=float)

    # Sample emission times uniformly in [t_min, t_max]
    t_es = rng.uniform(t_min, t_max, size=N_emit)

    for t_e in t_es:
        dt = t_obs - t_e

        # Time-shifted envelopes for all modes at this emission time
        fE_shifted = np.empty_like(fE0, dtype=complex)
        fP_shifted = np.empty_like(fP0, dtype=complex)

        for j in range(M):
            shiftE = vE[j] * dt
            shiftP = vP[j] * dt

            fE_shifted[j] = shift_envelope(fE0[j], xs, shiftE)
            fP_shifted[j] = shift_envelope(fP0[j], xs, shiftP)

        # Use   original analytic_G2 on these time-shifted modes
        G2_snapshot = analytic_G2(xs, fE_shifted, fP_shifted, r, alpha_P)
        G2_accum += G2_snapshot

    G2_avg = G2_accum / float(N_emit)
    return G2_avg

def v_E_of_omega(omega, v0_E=0.6, alpha=0.2, omega0=1.0):
    """
    Toy group velocity for the exterior (Hawking) mode as a function of omega.
    v_E(omega) varies smoothly around v0_E.
    """
    # Smooth variation (e.g. gentle dispersion):
    return v0_E + alpha * (omega - omega0)

def v_P_of_omega(omega, v0_P=-0.3, beta=0.1, omega0=1.0):
    """
    Toy group velocity for the partner mode as a function of omega.
    Negative sign to send it 'inside'.
    """
    return v0_P + beta * (omega - omega0)

def build_frequency_band_modes(xs, 
                               omega0=1.0, domega=0.2, 
                               M_omega=8,
                               xE0=-10.0, xP0=+10.0,
                               sigma=3.0,
                               r0=0.5,
                               alpha_P0=1.0+0j):
    """
    Construct a small band of modes indexed by frequency.

    Returns
    -------
    omegas : (M,) array
        Sampled frequencies.
    fE0, fP0 : (M, Nx) complex arrays
        Initial envelopes at emission for each frequency mode.
    vE, vP : (M,) arrays
        Group velocities v_E(omega), v_P(omega).
    r : (M,) array
        Squeezing parameters per mode.
    alpha_P : (M,) complex array
        Coherent seeds per mode.
    """
    xs = np.asarray(xs)
    Nx = xs.size

    # Sample omegas in a narrow band
    omegas = np.linspace(omega0 - domega, omega0 + domega, M_omega)

    # Allocate arrays
    fE0 = np.zeros((M_omega, Nx), dtype=complex)
    fP0 = np.zeros((M_omega, Nx), dtype=complex)
    vE  = np.zeros(M_omega, dtype=float)
    vP  = np.zeros(M_omega, dtype=float)
    r   = np.zeros(M_omega, dtype=float)
    alpha_P = np.zeros(M_omega, dtype=complex)

    dx = xs[1] - xs[0]

    for j, w in enumerate(omegas):
        # Gaussian envelopes at emission, same shape, different velocities later
        envE = np.exp(- (xs - xE0)**2 / (2 * sigma**2))
        envP = np.exp(- (xs - xP0)**2 / (2 * sigma**2))

        # Normalize each envelope so that sum |f|^2 dx = 1
        envE /= np.sqrt(np.sum(np.abs(envE)**2) * dx)
        envP /= np.sqrt(np.sum(np.abs(envP)**2) * dx)

        fE0[j] = envE
        fP0[j] = envP

        # Group velocities from toy dispersion
        vE[j] = v_E_of_omega(w)
        vP[j] = v_P_of_omega(w)

        # Squeezing and seed: can be omega-dependent if desired
        r[j] = r0
        alpha_P[j] = alpha_P0

    return omegas, fE0, fP0, vE, vP, r, alpha_P

def simulate_G2_frequency_band(xs,
                               omega0=1.0, domega=0.2, M_omega=8,
                               xE0=-10.0, xP0=+10.0,
                               sigma=3.0,
                               r0=0.5,
                               alpha_P0=1.0+0j,
                               t_obs=10.0,
                               t_min=0.0, t_max=10.0,
                               N_emit=200,
                               rng=None):
    """
    Simulate a G^2(x,x') map coming from a narrow band of Hawking frequencies,
    each with its own group velocity v_E(omega), v_P(omega), and averaged over
    random emission times.

    This should produce a single (slightly thickened and curved) cross-horizon ridge,
    qualitatively similar to experimental plots.

    Returns
    -------
    G2_ridge : (Nx, Nx) array (real)
        Simulated G^2(x,x') with one dominant ridge from the ω-band.
    aux : dict
        Auxiliary info: omegas, vE, vP, etc.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) Build band of frequency-labelled modes at emission
    (omegas, fE0, fP0, 
     vE, vP, r, alpha_P) = build_frequency_band_modes(xs,
                                                      omega0=omega0,
                                                      domega=domega,
                                                      M_omega=M_omega,
                                                      xE0=xE0, xP0=xP0,
                                                      sigma=sigma,
                                                      r0=r0,
                                                      alpha_P0=alpha_P0)

    # 2) Use the emission-smeared envelope-only G2 builder
    G2_ridge = analytic_G2_envelope_emission_smeared(xs, fE0, fP0, r, alpha_P,
                                                     vE, vP,
                                                     t_obs,
                                                     t_min, t_max,
                                                     N_emit,
                                                     rng=rng)
    aux = {
        "omegas": omegas,
        "vE": vE,
        "vP": vP,
        "r": r,
        "alpha_P": alpha_P,
    }
    return G2_ridge, aux

def analytic_G2_envelope_emission_smeared_from_modes(xs, fE0, fP0, r, alpha_P,
                                                     vE, vP,
                                                     t_obs,
                                                     t_min, t_max,
                                                     N_emit,
                                                     rng=None):
    """
    Emission-time-averaged, envelope-only G^2(x,x') using local per-mode velocities
    vE[j] ~ c - v, vP[j] ~ v - c at the mode centers.
    """
    xs = np.asarray(xs)
    fE0 = np.asarray(fE0)
    fP0 = np.asarray(fP0)
    alpha_P = np.asarray(alpha_P, dtype=complex)

    M, Nx = fE0.shape
    assert fP0.shape == (M, Nx)

    vE = np.broadcast_to(np.asarray(vE, float), (M,))
    vP = np.broadcast_to(np.asarray(vP, float), (M,))

    if rng is None:
        rng = np.random.default_rng()

    G2_accum = np.zeros((Nx, Nx), dtype=float)

    # Sample emission times uniformly in [t_min, t_max]
    t_es = rng.uniform(t_min, t_max, size=N_emit)

    for t_e in t_es:
        dt = t_obs - t_e

        fE_shifted = np.empty_like(fE0, dtype=complex)
        fP_shifted = np.empty_like(fP0, dtype=complex)

        for j in range(M):
            shiftE = vE[j] * dt
            shiftP = vP[j] * dt
            fE_shifted[j] = shift_envelope(fE0[j], xs, shiftE)
            fP_shifted[j] = shift_envelope(fP0[j], xs, shiftP)

        # IMPORTANT: pass the *shifted* fP into analytic_G2
        G2_snapshot = analytic_G2(xs, fE_shifted, fP_shifted, r, alpha_P)
        G2_accum += G2_snapshot

    return G2_accum / float(N_emit)

def hydrodynamic_speeds_from_centers(x_out_centers, alpha_hydro, curv,
                                     s_out0=1.0):
    """
    Compute local hydrodynamic correlation speeds v_E, v_P
    from the phenomenological mapping x_in(x_out).

    v_E[j] ~ c(x_out_j) - v(x_out_j)
    v_P[j] ~ v(x_in_j) - c(x_in_j)
    with v_P[j]/v_E[j] ≈ d x_in/d x_out at x_out_j.
    """
    x_out_centers = np.asarray(x_out_centers, dtype=float)

    # Local slope of   fitted ridge
    slope = -alpha_hydro + 2.0 * curv * x_out_centers  # d x_in / d x_out

    vE = np.full_like(x_out_centers, s_out0, dtype=float)
    vP = slope * vE

    return vE, vP

def shift_envelope(fx, xs, shift):
    """
    Non-periodic shift: f_shifted(x) = f(x - shift), zero outside the domain.
    """
    xs = np.asarray(xs)
    fx = np.asarray(fx)
    return np.interp(xs - shift, xs, fx, left=0.0, right=0.0)


def analytic_G2_envelope_emission_smeared_curved(xs, fE0, fP0, r, alpha_P,
                                                 x_out_centers, x_in_centers,
                                                 alpha_hydro, curv,
                                                 vE,
                                                 t_obs, t_min, t_max,
                                                 N_emit,
                                                 rng=None):
    """
    Emission-time-averaged, envelope-only G^2(x,x') where the exterior
    packet center moves linearly with speed vE, and the interior packet
    center follows the *curved* hydrodynamic mapping

        x_in(t) = -alpha_hydro * x_out(t) + curv * x_out(t)^2

    so that the cross-horizon ridge matches the phenomenological
    curvature used in the static model.
    """
    xs = np.asarray(xs)
    fE0 = np.asarray(fE0)
    fP0 = np.asarray(fP0)
    alpha_P = np.asarray(alpha_P, dtype=complex)
    x_out_centers = np.asarray(x_out_centers, dtype=float)
    x_in_centers  = np.asarray(x_in_centers,  dtype=float)

    M, Nx = fE0.shape
    assert fP0.shape == (M, Nx)

    if rng is None:
        rng = np.random.default_rng()

    # ensure scalar vE
    vE = float(vE)

    G2_accum = np.zeros((Nx, Nx), dtype=float)

    # Sample emission times uniformly in [t_min, t_max]
    if t_max == t_min:
        t_es = np.array([t_obs])  # trivial (no propagation) case
    else:
        t_es = rng.uniform(t_min, t_max, size=N_emit)

    for t_e in t_es:
        dt = t_obs - t_e

        fE_shifted = np.empty_like(fE0, dtype=complex)
        fP_shifted = np.empty_like(fP0, dtype=complex)

        for j in range(M):
            xE0 = x_out_centers[j]
            xP0 = x_in_centers[j]

            # new exterior position at this dt
            xE_t = xE0 + vE * dt

            # curved hydrodynamic mapping for interior at this dt
            xP_t = -alpha_hydro * xE_t + curv * xE_t**2

            # shifts relative to original centers
            shiftE = xE_t - xE0
            shiftP = xP_t - xP0

            fE_shifted[j] = shift_envelope(fE0[j], xs, shiftE)
            fP_shifted[j] = shift_envelope(fP0[j], xs, shiftP)

        G2_snapshot = analytic_G2(xs, fE_shifted, fP_shifted, r, alpha_P)
        G2_accum += G2_snapshot

    G2_avg = G2_accum / float(max(len(t_es), 1))
    return G2_avg

def bandlimit_modes(xs, f_modes, k_max):
    """
    Band-limit a set of mode functions along x to |k| <= k_max.

    Parameters
    ----------
    xs : (Nx,) array
    f_modes : (M, Nx) complex array
    k_max : float
        Maximum |k| to keep (in 1/xi units if xi=1).
    """
    xs = np.asarray(xs)
    f_modes = np.asarray(f_modes)
    M, Nx = f_modes.shape
    dx = xs[1] - xs[0]

    # FFT frequencies (2πn/L convention)
    ks = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    mask = np.abs(ks) <= k_max

    f_bl = np.zeros_like(f_modes, dtype=complex)
    for j in range(M):
        Fk = np.fft.fft(f_modes[j])
        Fk[~mask] = 0.0
        fx_bl = np.fft.ifft(Fk)

        # renormalize so sum |f|^2 dx = 1
        norm = np.sqrt(np.sum(np.abs(fx_bl)**2) * dx)
        if norm > 0:
            fx_bl /= norm
        f_bl[j] = fx_bl

    return f_bl

def hawking_r_of_k(k_vals, T_H, c_s=1.0):
    """
    Given k-values (1/xi units) and a Hawking temperature T_H (in units of c_s/xi),
    return mode-dependent squeezing parameters r_j such that
        sinh^2(r_j) = n_H(omega_j),
    with n_H thermal at T_H and omega_j = c_s * |k_j| (hydrodynamic dispersion).
    """
    k_vals = np.asarray(k_vals, dtype=float)
    omega = c_s * np.abs(k_vals)

    # Thermal occupation n_H(omega) = 1/(exp(omega/T_H) - 1)
    # Avoid divide-by-zero if omega=0 by adding tiny epsilon
    eps = 1e-12
    n_H = 1.0 / (np.exp(omega / (T_H + eps)) - 1.0 + eps)

    # r such that sinh^2 r = n_H
    r_vals = np.arcsinh(np.sqrt(n_H))
    return r_vals

def soft_bandlimit_modes(xs, f_modes, k_max):
    xs = np.asarray(xs)
    f_modes = np.asarray(f_modes)
    M, Nx = f_modes.shape
    dx = xs[1] - xs[0]

    ks = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    # Smooth Gaussian window in k, centered at 0, width ~ k_max
    window = np.exp(-(ks / k_max)**4)   # super-Gaussian, smoother than hard cutoff

    f_bl = np.zeros_like(f_modes, dtype=complex)
    for j in range(M):
        Fk = np.fft.fft(f_modes[j])
        Fk_filtered = Fk * window
        fx_bl = np.fft.ifft(Fk_filtered)

        # renormalize
        norm = np.sqrt(np.sum(np.abs(fx_bl)**2) * dx)
        if norm > 0:
            fx_bl /= norm
        f_bl[j] = fx_bl

    return f_bl

def static_structure_factor(k_vals, c_s=1.0, m=1.0):
    """
    Zero-temperature Bogoliubov static structure factor S(k) = ε_k / E_k
    with ε_k = k^2 / (2m), E_k = sqrt[ ε_k ( ε_k + 2 m c_s^2 ) ].
    """
    k_vals = np.asarray(k_vals, dtype=float)
    eps = k_vals**2 / (2.0 * m)
    Ek  = np.sqrt(eps * (eps + 2.0 * m * c_s**2))
    S_k = eps / Ek
    return S_k

def thermal_r_from_k(k_vals, T_H, c_s=1.0):
    """
    Given k-values and Hawking temperature T_H (ħ = k_B = 1),
    return squeezing parameters r_j such that n̄_k = sinh^2 r_k
    follows a Bose-Einstein distribution in ω = c_s |k|.
    """
    k_vals = np.asarray(k_vals, dtype=float)
    omega = c_s * np.abs(k_vals)
    beta  = 1.0 / T_H
    # avoid overflow at very small omega if needed:
    # omega = np.clip(omega, 1e-6, None)
    nbar  = 1.0 / (np.exp(beta * omega) - 1.0)
    r_vals = np.arcsinh(np.sqrt(nbar))
    return r_vals

def spectrum_from_G2(xs, G2, region_mask=None):
    """
    Estimate an effective mode-density spectrum S_eff(k) from G^2(x,x').

    Steps:
      1. Restrict to a region (e.g. outside) via region_mask, if given.
      2. Build the averaged correlator C(Δx) by averaging G2(x, x+Δx)
         over the center coordinate within that region.
      3. Fourier-transform C(Δx) over Δx to get S_eff(k).

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid (assumed uniform).
    G2 : (Nx, Nx) array
        Equal-time density covariance G^2(x,x').
    region_mask : (Nx,) bool array or None
        If given, only indices where region_mask[i] is True are used
        (e.g. xs > 0 for the exterior). If None, use all points.

    Returns
    -------
    k_vals : (Nlags,) array
        Wavenumbers corresponding to S_eff(k) (in rad/length).
    S_eff : (Nlags,) array
        Effective spectrum S_eff(k); real-valued.
    C_dx  : (Nlags,) array
        Real-space correlator C(Δx) that was Fourier-transformed.
    dx    : float
        Grid spacing (for reference).
    """

    xs = np.asarray(xs)
    G2 = np.asarray(G2)
    Nx = xs.size
    assert G2.shape == (Nx, Nx)

    dx = xs[1] - xs[0]

    # Default: whole system
    if region_mask is None:
        region_mask = np.ones(Nx, dtype=bool)
    else:
        region_mask = np.asarray(region_mask, dtype=bool)
        if region_mask.shape != (Nx,):
            raise ValueError("region_mask must have shape (Nx,)")

    max_lag = Nx - 1
    Nlags   = 2 * max_lag + 1
    lags    = np.arange(-max_lag, max_lag + 1)

    C_dx = np.zeros(Nlags, dtype=float)
    center = max_lag  # index where lag = 0 lives in C_dx

    # Compute C(Δx) for non-negative lags and mirror to negative
    for lag in range(0, max_lag + 1):
        if lag == 0:
            i0 = np.arange(Nx)
            j0 = i0
        else:
            i0 = np.arange(0, Nx - lag)
            j0 = i0 + lag

        # Both points must lie in the chosen region
        mask = region_mask[i0] & region_mask[j0]
        if np.any(mask):
            C_lag = G2[i0[mask], j0[mask]].mean()
        else:
            C_lag = 0.0

        # Positive lag
        C_dx[center + lag] = C_lag
        # Negative lag: use symmetry C(-Δx) = C(+Δx)
        C_dx[center - lag] = C_lag

    # FFT: C(Δx) -> S(k), with k=0 centered
    C_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(C_dx)))
    # k-grid in rad/length
    k_vals = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nlags, d=dx))

    # Real spectrum (imag part ~ numerical noise)
    S_eff = C_fft.real * dx

    return k_vals, S_eff, C_dx, dx

# -------------------------------------------------------------------
# Helper: compute G2 and extracted S_eff(k) for a given T_H
# -------------------------------------------------------------------
def G2_and_spectrum_for_TH(xs, fE, fP, alpha_P,
                           k_vals, T_H,
                           c_s=1.0, m=1.0,
                           region_mask=None):
    """
    For a given Hawking temperature T_H, compute:
      - mode-dependent squeezing r_k(T_H),
      - Bogoliubov S(k),
      - analytic G^2(x,x'),
      - extracted spectrum S_eff(k) from G^2.
    """

    # thermal squeezing
    r_k = thermal_r_from_k(k_vals, T_H=T_H, c_s=c_s)

    # Bogoliubov structure factor (same for E and P here)
    S_k = static_structure_factor(k_vals, c_s=c_s, m=m)

    # build G^2 with Bogoliubov factors
    G2_bogo = analytic_G2(xs, fE, fP,
                          r=r_k,
                          alpha_P=alpha_P,
                          S_E=S_k,
                          S_P=S_k)

    # extract effective spectrum from G^2
    k_spec, S_eff, C_dx, dx = spectrum_from_G2(xs, G2_bogo,
                                               region_mask=region_mask)
    return k_spec, S_eff, C_dx, dx, G2_bogo

# -------------------------------------------------------------------
# Main diagnostic: compare spectra for several T_H
# -------------------------------------------------------------------
def compare_spectra_vs_TH(xs, fE, fP, alpha_P,
                          k_vals,
                          TH_list,
                          c_s=1.0,
                          m=1.0,
                          region_mask=None,
                          kmax_plot=None):
    """
    For a list of Hawking temperatures TH_list, compute and plot
    the effective spectrum S_eff(k) extracted from G^2(x,x').

    Plots:
      1) Raw S_eff(k) (amplitude + shape)
      2) S_eff(k) normalized by its max (shape only)

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid.
    fE, fP : (M, Nx) arrays
        Mode functions for exterior and partner modes.
    alpha_P : (M,) array
        Coherent seeds in partner modes.
    k_vals : (M,) array
        Mode wavenumbers associated with pairs (for r_k, S_k).
        Note: this is *not* the same as the FFT k-grid of S_eff(k).
    TH_list : list of floats
        Hawking temperatures to sweep.
    c_s, m : floats
        Sound speed and mass in static_structure_factor.
    region_mask : (Nx,) bool or None
        Spatial mask for spectrum extraction (e.g. xs > 0 for outside).
    kmax_plot : float or None
        If given, restrict plots to |k| <= kmax_plot (for readability).
    """

    spectra_raw   = []
    spectra_norm  = []
    k_fft_grid    = None

    for T_H in TH_list:
        k_spec, S_eff, C_dx, dx, G2_bogo = G2_and_spectrum_for_TH(
            xs, fE, fP, alpha_P, k_vals, T_H,
            c_s=c_s, m=m, region_mask=region_mask
        )

        if k_fft_grid is None:
            k_fft_grid = k_spec
        else:
            # sanity: all runs should give the same FFT k-grid
            assert np.allclose(k_fft_grid, k_spec)

        spectra_raw.append(S_eff)

        # normalize spectrum by its max absolute value to compare shapes
        max_val = np.max(np.abs(S_eff))
        if max_val > 0:
            spectra_norm.append(S_eff / max_val)
        else:
            spectra_norm.append(S_eff)

    spectra_raw  = np.array(spectra_raw)   # shape (nT, Nk)
    spectra_norm = np.array(spectra_norm)  # shape (nT, Nk)

    # Optionally restrict to |k| <= kmax_plot for visual clarity
    if kmax_plot is not None:
        mask_k = np.abs(k_fft_grid) <= kmax_plot
        k_plot = k_fft_grid[mask_k]
        spectra_raw_plot  = spectra_raw[:, mask_k]
        spectra_norm_plot = spectra_norm[:, mask_k]
    else:
        k_plot = k_fft_grid
        spectra_raw_plot  = spectra_raw
        spectra_norm_plot = spectra_norm

    # ----------------------------------------------------------------
    # Plot: raw spectra
    # ----------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for S, T_H in zip(spectra_raw_plot, TH_list):
        plt.plot(k_plot, S, label=f"T_H = {T_H}")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$S_{\rm eff}(k)$ (raw)")
    plt.title("Effective spectrum from $G^{(2)}(x,x')$ vs Hawking $T_H$")
    plt.legend()
    plt.tight_layout()

    # ----------------------------------------------------------------
    # Plot: normalized spectra (compare shapes)
    # ----------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for S, T_H in zip(spectra_norm_plot, TH_list):
        plt.plot(k_plot, S, label=f"T_H = {T_H}")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$S_{\rm eff}(k)/\max|S_{\rm eff}|$")
    plt.title("Normalized spectrum shapes vs Hawking $T_H$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return k_fft_grid, spectra_raw, spectra_norm

def build_mode_functions_with_k(xs,
                                x_out_centers,
                                x_in_centers,
                                k_vals,
                                sigma,
                                counterprop=True):
    """
    Build 1D mode functions fE, fP with explicit k-dependent phases.

    Each mode j is a Gaussian envelope centered at x_out_centers[j] or
    x_in_centers[j], multiplied by a plane-wave phase factor.

      fE_j(x) ∝ exp(-(x - x_out_j)^2 / (4σ^2)) * exp(+i k_j (x - x_out_j))
      fP_j(x) ∝ exp(-(x - x_in_j)^2  / (4σ^2)) * exp(±i k_j (x - x_in_j))

    The sign for the partner phase is controlled by `counterprop`:
      - counterprop=True  → fE has +k_j, fP has -k_j (counter-propagating)
      - counterprop=False → both use +k_j

    All modes are normalized so that
        sum_x |f|^2 dx = 1
    on the discrete grid.

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid (assumed uniform).
    x_out_centers : (M,) array
        Centers of exterior (Hawking) packets.
    x_in_centers : (M,) array
        Centers of partner packets.
    k_vals : (M,) array
        Wavenumbers associated with each mode j.
    sigma : float
        Gaussian width parameter.
    counterprop : bool
        If True, partner gets opposite k sign.

    Returns
    -------
    fE, fP : (M, Nx) complex arrays
        Mode functions f_{E j}(x), f_{P j}(x) with proper normalization.
    """

    xs = np.asarray(xs, dtype=float)
    x_out_centers = np.asarray(x_out_centers, dtype=float)
    x_in_centers  = np.asarray(x_in_centers,  dtype=float)
    k_vals        = np.asarray(k_vals,        dtype=float)

    M  = x_out_centers.size
    Nx = xs.size
    assert x_in_centers.shape == (M,)
    assert k_vals.shape == (M,)

    dx = xs[1] - xs[0]

    fE = np.zeros((M, Nx), dtype=complex)
    fP = np.zeros((M, Nx), dtype=complex)

    for j in range(M):
        x0_out = x_out_centers[j]
        x0_in  = x_in_centers[j]
        k      = k_vals[j]

        # Gaussian envelopes
        env_E = np.exp(-(xs - x0_out)**2 / (4.0 * sigma**2))
        env_P = np.exp(-(xs - x0_in)**2  / (4.0 * sigma**2))

        # Phase factors (measured from the packet center, to keep
        # the overall phase from drifting wildly with x0)
        phase_E = np.exp(1j * k * (xs - x0_out))

        if counterprop:
            phase_P = np.exp(-1j * k * (xs - x0_in))  # opposite k
        else:
            phase_P = np.exp(1j * k * (xs - x0_in))   # same k

        fE_j = env_E * phase_E
        fP_j = env_P * phase_P

        # Normalize: sum_x |f|^2 dx = 1
        norm_E = np.sqrt(np.sum(np.abs(fE_j)**2) * dx)
        norm_P = np.sqrt(np.sum(np.abs(fP_j)**2) * dx)

        if norm_E > 0:
            fE_j /= norm_E
        if norm_P > 0:
            fP_j /= norm_P

        fE[j, :] = fE_j
        fP[j, :] = fP_j

    return fE, fP

def analytic_G2_spectral(xs, k_vals, T_H, c_s=1.0, m=1.0):
    """
    Homogeneous toy G^2(x,x') with correct k-structure:
        G^2(x,x') = sum_k W(k; T_H) cos(k (x - x'))

    Where W(k;T_H) ~ S(k) (2 n_k(T_H) + 1) and
    n_k(T_H) = 1 / (exp(omega_k / T_H) - 1),
    omega_k = c_s |k|.

    This is NOT using   wavepacket geometry, just a
    testbed to see how temperature changes the spectral shape.
    """

    xs = np.asarray(xs)
    k_vals = np.asarray(k_vals)
    Nx = xs.size
    dx = xs[1] - xs[0]

    # Bogoliubov static structure factor S(k)
    def S_k(k):
        eps = k**2 / (2.0 * m)
        Ek  = np.sqrt(eps * (eps + 2.0 * m * c_s**2))
        return eps / Ek

    # Hawking occupation n_k(T_H)
    omega = c_s * np.abs(k_vals)
    nbar  = 1.0 / (np.exp(omega / T_H) - 1.0)

    # weights ~ S(k) (2 n + 1)
    Svals = S_k(k_vals)
    Wk    = Svals * (2.0 * nbar + 1.0)

    # Build G^2(x,x') = sum_k Wk cos(k (x - x'))
    G2 = np.zeros((Nx, Nx), dtype=float)
    for k, w in zip(k_vals, Wk):
        phase = k * (xs[:, None] - xs[None, :])  # (Nx,Nx)
        G2 += w * np.cos(phase)

    return G2, Wk

def analytic_G2_packet_spectral(xs, fE, k_vals, T_H, c_s=1.0, m=1.0):
    """
    Build an out-out G^2(x,x') that keeps   packet envelopes but
    adds explicit cos(k_j (x - x')) fringes per mode:

        G^2(x,x') = sum_j W_j(T_H) |fE_j(x)| |fE_j(x')| cos(k_j (x - x'))

    where W_j(T_H) ~ S(k_j) [2 n̄_j(T_H) + 1],
    n̄_j(T_H) = 1 / (exp(c_s |k_j| / T_H) - 1),
    and S(k) = ε_k / E_k is the Bogoliubov static structure factor.

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid (uniform).
    fE : (M, Nx) complex array
        Exterior mode functions (from   packet builder).
    k_vals : (M,) array
        Wavenumbers associated with each mode j.
    T_H : float
        Hawking temperature (units with ħ = k_B = 1).
    c_s : float
        Sound speed.
    m : float
        Atomic mass (sets S(k) via ε_k = k^2/(2m)).

    Returns
    -------
    G2 : (Nx, Nx) float array
        Out-out G^2(x,x') with k-sensitive structure.
    Wk : (M,) float array
        Mode weights W_j(T_H) used in the sum.
    """

    xs     = np.asarray(xs, dtype=float)
    fE     = np.asarray(fE, dtype=complex)
    k_vals = np.asarray(k_vals, dtype=float)

    M, Nx = fE.shape
    assert xs.shape == (Nx,)
    assert k_vals.shape == (M,)

    dx = xs[1] - xs[0]

    # --- Bogoliubov static structure factor S(k) ---
    eps = k_vals**2 / (2.0 * m)
    Ek  = np.sqrt(eps * (eps + 2.0 * m * c_s**2))
    S_k = eps / Ek               # S(k) = ε_k / E_k

    # --- Hawking occupation n̄(k; T_H) ---
    omega = c_s * np.abs(k_vals)
    nbar  = 1.0 / (np.exp(omega / T_H) - 1.0)

    # --- Mode weights W_j(T_H) ~ S(k) (2 n̄ + 1) ---
    # Wk = S_k * (2.0 * nbar + 1.0)
    Wk = hawking_weight_W_exc(k_vals, T_H, c_s=c_s, m=m)

    # --- Build G^2(x,x') ---
    G2 = np.zeros((Nx, Nx), dtype=float)

    # Precompute Δx matrix once
    DeltaX = xs[:, None] - xs[None, :]   # (Nx, Nx)

    # Use envelope = |fE_j(x)| (drop phase)
    env = np.abs(fE)   # (M, Nx)

    for j in range(M):
        env_j = env[j, :].copy()

        # normalize envelope s.t. sum_x |env_j|^2 dx = 1 (optional but nice)
        norm = np.sqrt(np.sum(env_j**2) * dx)
        if norm > 0:
            env_j /= norm

        EE_env = np.outer(env_j, env_j)        # (Nx, Nx)
        phase  = k_vals[j] * DeltaX           # k_j (x - x')
        G2 += Wk[j] * EE_env * np.cos(phase)

    return G2, Wk
def static_structure_factor(k_vals, c_s=1.0, m=1.0):
    k_vals = np.asarray(k_vals, dtype=float)
    eps = k_vals**2 / (2.0 * m)
    Ek  = np.sqrt(eps * (eps + 2.0 * m * c_s**2))
    return eps / Ek

def hawking_weight_W(k_vals, T_H, c_s=1.0, m=1.0):
    k_vals = np.asarray(k_vals, dtype=float)
    S_k = static_structure_factor(k_vals, c_s=c_s, m=m)
    omega = c_s * np.abs(k_vals)
    nbar  = 1.0 / (np.exp(omega / T_H) - 1.0)
    Wk    = S_k * (2.0 * nbar + 1.0)
    return Wk

def hawking_weight_W_exc(k_vals, T_H, c_s=1.0, m=1.0):
    """
    Thermal *excess* weight:
        W_exc(k;T_H) = 2 S(k) n̄_k(T_H),
    i.e. zero-temperature baseline subtracted.
    """
    k_vals = np.asarray(k_vals, dtype=float)
    S_k = static_structure_factor(k_vals, c_s=c_s, m=m)
    omega = c_s * np.abs(k_vals)
    nbar  = 1.0 / (np.exp(omega / T_H) - 1.0)
    Wk_exc = S_k * (2.0 * nbar)
    return Wk_exc

def gaussian_packet(xs, x0, sigma):
    """
    Real, normalized Gaussian envelope centered at x0
    with width sigma:
        phi(x) ∝ exp(-(x - x0)^2 / (4 sigma^2))
    Normalized so sum_x |phi|^2 dx = 1.
    """
    xs = np.asarray(xs, dtype=float)
    dx = xs[1] - xs[0]
    env = np.exp(-(xs - x0)**2 / (4.0 * sigma**2))
    norm = np.sqrt(np.sum(env**2) * dx)
    if norm > 0:
        env /= norm
    return env

def analytic_G2_with_seeds(xs,
                           fE, fP,
                           r, alpha_P,
                           seed_events,
                           seed_sigma_out,
                           seed_sigma_in,
                           A_auto=1.0,
                           B_cross=1.0,
                           C_cross_out=1.0):
    """
    Extend analytic_G2 by adding effective contributions from
    several seeded wavepackets at different (effective) times
    (i.e. at different out/in positions at t_obs).

    Background:
        G2_bg(x,x') = analytic_G2(...)  # Hawking + any stationary seeds

    Seeds:
      For each event a in seed_events:
        - place an out-packet phi_E,a(x) around x_out,a
        - place an in-packet  phi_P,a(x) around x_in,a

      Then add phenomenological contributions

        G2_seed(x,x') =
            sum_a  A_auto * phi_E,a(x) phi_E,a(x')         [extra out-out auto blobs]
          + sum_a  B_cross * [phi_E,a(x) phi_P,a(x') +
                              phi_P,a(x) phi_E,a(x')]      [extra cross-horizon blobs]
          + sum_{a,b} C_cross_out * phi_E,a(x) phi_E,b(x') [off-diagonal out-out blobs]

      where a != b terms in the last sum create off-diagonal out-out peaks.

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid.
    fE, fP, r, alpha_P :
        Same as in   analytic_G2.
    seed_events : list of dicts
        Each dict must have keys:
            'x_out' : float, out-position of seed at t_obs
            'x_in'  : float, in-position of seed at t_obs
    seed_sigma_out : float
        Width of out seed packets.
    seed_sigma_in : float
        Width of in seed packets.
    A_auto, B_cross, C_cross_out : floats
        Phenomenological amplitudes for the three seed contributions.

    Returns
    -------
    G2_total : (Nx, Nx) float array
        Background + seed contributions.
    G2_bg    : (Nx, Nx) float array
        Background G^2 from analytic_G2.
    G2_seed  : (Nx, Nx) float array
        Seed-only phenomenological contribution.
    """

    xs = np.asarray(xs)
    Nx = xs.size

    # --- 1) background from   analytic_G2 ---
    G2_bg = analytic_G2(xs, fE, fP, r, alpha_P)  #   existing function

    # --- 2) build seed wavepackets ---
    # one phi_E,a and phi_P,a per event
    phis_E = []
    phis_P = []
    for ev in seed_events:
        x_out_a = ev['x_out']
        x_in_a  = ev['x_in']

        phi_E_a = gaussian_packet(xs, x_out_a, seed_sigma_out)
        phi_P_a = gaussian_packet(xs, x_in_a,  seed_sigma_in)

        phis_E.append(phi_E_a)
        phis_P.append(phi_P_a)

    phis_E = np.array(phis_E)  # (Nseeds, Nx)
    phis_P = np.array(phis_P)  # (Nseeds, Nx)

    Nseeds = len(seed_events)

    # --- 3) construct seed G2 contribution ---
    G2_seed = np.zeros((Nx, Nx), dtype=float)

    # (a) extra out-out auto blobs
    if A_auto != 0.0:
        for a in range(Nseeds):
            phiE = phis_E[a]
            G2_seed += A_auto * np.outer(phiE, phiE)

    # (b) cross-horizon blobs
    if B_cross != 0.0:
        for a in range(Nseeds):
            phiE = phis_E[a]
            phiP = phis_P[a]
            G2_seed += B_cross * (np.outer(phiE, phiP) + np.outer(phiP, phiE))

    # (c) off-diagonal out-out blobs (a != b)
    if C_cross_out != 0.0:
        for a in range(Nseeds):
            for b in range(Nseeds):
                if a == b:
                    continue
                phiEa = phis_E[a]
                phiEb = phis_E[b]
                G2_seed += C_cross_out * np.outer(phiEa, phiEb)

    # total
    G2_total = G2_bg + G2_seed

    return G2_total, G2_bg, G2_seed

def add_seed_blobs_to_G2(xs,
                         G2_bg,
                         seed_events,
                         seed_sigma_out,
                         seed_sigma_in,
                         A_auto=1.0,
                         B_cross=1.0,
                         C_cross_out=1.0):
    """
    Add effective seeded-wavepacket contributions to an existing G^2(x,x').

    G2_total(x,x') = G2_bg(x,x') + G2_seed(x,x')

    Seeds:
      For each event a in seed_events:
        - place an out-packet phi_E,a(x) around x_out,a
        - place an in-packet  phi_P,a(x) around x_in,a

      Then add phenomenological contributions

        G2_seed(x,x') =
            sum_a  A_auto * phi_E,a(x) phi_E,a(x')         [extra out-out auto blobs]
          + sum_a  B_cross * [phi_E,a(x) phi_P,a(x') +
                              phi_P,a(x) phi_E,a(x')]      [extra cross-horizon blobs]
          + sum_{a,b} C_cross_out * phi_E,a(x) phi_E,b(x') [off-diagonal out-out blobs]

      where a != b terms in the last sum create off-diagonal out-out peaks.

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid.
    G2_bg : (Nx, Nx) array
        Background G^2(x,x') (e.g. from analytic_G2 or simulation).
    seed_events : list of dicts
        Each dict must have keys:
            'x_out' : float, out-position of seed at t_obs
            'x_in'  : float, in-position of seed at t_obs
    seed_sigma_out : float
        Width of out seed packets.
    seed_sigma_in : float
        Width of in seed packets.
    A_auto, B_cross, C_cross_out : floats
        Amplitudes for the three seed contributions.

    Returns
    -------
    G2_total : (Nx, Nx) float array
        Background + seed contributions.
    G2_seed  : (Nx, Nx) float array
        Seed-only phenomenological contribution.
    """

    xs   = np.asarray(xs)
    G2_bg = np.asarray(G2_bg)
    Nx   = xs.size
    assert G2_bg.shape == (Nx, Nx)

    # --- build seed wavepackets ---
    phis_E = []
    phis_P = []
    for ev in seed_events:
        x_out_a = ev['x_out']
        x_in_a  = ev['x_in']

        phi_E_a = gaussian_packet(xs, x_out_a, seed_sigma_out)
        phi_P_a = gaussian_packet(xs, x_in_a,  seed_sigma_in)

        phis_E.append(phi_E_a)
        phis_P.append(phi_P_a)

    phis_E = np.array(phis_E)  # (Nseeds, Nx)
    phis_P = np.array(phis_P)  # (Nseeds, Nx)
    Nseeds = len(seed_events)

    # --- seed-only G2 ---
    G2_seed = np.zeros((Nx, Nx), dtype=float)

    # (a) extra out-out auto blobs (diagonal ridge)
    if A_auto != 0.0:
        for a in range(Nseeds):
            phiE = phis_E[a]
            G2_seed += A_auto * np.outer(phiE, phiE)

    # (b) cross-horizon blobs
    if B_cross != 0.0:
        for a in range(Nseeds):
            phiE = phis_E[a]
            phiP = phis_P[a]
            G2_seed += B_cross * (np.outer(phiE, phiP) + np.outer(phiP, phiE))

    # (c) off-diagonal out-out blobs (a != b)
    if C_cross_out != 0.0:
        for a in range(Nseeds):
            for b in range(Nseeds):
                if a == b:
                    continue
                phiEa = phis_E[a]
                phiEb = phis_E[b]
                G2_seed += C_cross_out * np.outer(phiEa, phiEb)

    G2_total = G2_bg + G2_seed
    return G2_total, G2_seed

def analytic_G2_with_two_mode_cross(xs,
                                    fE, fP,
                                    r, alpha_P,
                                    j1, j2,
                                    XE12,
                                    S_E=None,
                                    S_P=None):
    """
    Extend analytic_G2 by adding an out-out cross-correlation
    between two specific exterior modes j1 and j2:

        <X_{E,j1} X_{E,j2}>_c = XE12  (real)

    This adds to G^2(x,x') the term

        G2_cross(x,x') =
            XE12 * sqrt(S_E[j1] S_E[j2]) *
            [ fE_{j1}(x) fE_{j2}(x') + fE_{j2}(x) fE_{j1}(x') ],

    which produces off-diagonal out-out blobs at the overlap
    of modes j1 and j2.

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid.
    fE, fP : (M, Nx) arrays
        Exterior and partner mode functions.
    r : float or (M,) array
        Squeezing parameters (as in analytic_G2).
    alpha_P : (M,) array
        Coherent seeds in P modes.
    j1, j2 : int
        Indices of the two exterior modes to correlate.
    XE12 : float
        Cross-covariance <X_{E,j1} X_{E,j2}>_c.
        Controls strength of off-diagonal out-out blob.
    S_E, S_P : (M,) arrays or None
        Static structure factors for E and P; if None, taken as 1.

    Returns
    -------
    G2_total : (Nx, Nx) array
        Full G^2 with cross term.
    G2_base  : (Nx, Nx) array
        Original analytic_G2 (no cross).
    G2_cross : (Nx, Nx) array
        Cross term only.
    """

    xs = np.asarray(xs)
    fE = np.asarray(fE)
    fP = np.asarray(fP)
    alpha_P = np.asarray(alpha_P, dtype=complex)

    M, Nx = fE.shape
    assert fP.shape == (M, Nx)
    assert 0 <= j1 < M and 0 <= j2 < M and j1 != j2

    # 1) base G^2 from   existing Gaussian model
    if S_E is None and S_P is None:
        G2_base = analytic_G2(xs, fE, fP, r, alpha_P)
        S_E_arr = np.ones(M, dtype=float)
    else:
        S_E_arr = np.ones(M, dtype=float) if S_E is None else np.asarray(S_E, dtype=float)
        S_P_arr = np.ones(M, dtype=float) if S_P is None else np.asarray(S_P, dtype=float)
        G2_base = analytic_G2(xs, fE, fP, r, alpha_P, S_E=S_E_arr, S_P=S_P_arr)

    # 2) build cross term between j1 and j2 in the exterior
    fE1 = fE[j1, :]
    fE2 = fE[j2, :]

    # include Bogoliubov weighting if present
    amp = XE12 * np.sqrt(S_E_arr[j1] * S_E_arr[j2])

    G2_cross = amp * (np.outer(fE1, fE2) + np.outer(fE2, fE1))

    G2_total = G2_base + G2_cross

    return G2_total, G2_base, G2_cross

def G2_two_seed_block(xs,
                      fE, fP,
                      r, alpha_P,
                      j1, j2,
                      C_block,
                      S_E=None,
                      S_P=None):
    """
    Replace the contributions of modes j1 and j2 in analytic_G2 by a
    general 4-mode Gaussian block over (E1,P1,E2,P2) with covariance C_block.

    Indices in C_block correspond to:
        0 -> X_{E1}, 1 -> X_{P1}, 2 -> X_{E2}, 3 -> X_{P2}.

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid.
    fE, fP : (M, Nx) arrays
        Exterior and partner mode functions.
    r : float or (M,) array
        Squeezing parameters for all modes (used for the base G2).
    alpha_P : (M,) array
        Coherent seeds in P-modes for base G2.
    j1, j2 : int
        Mode indices for the two seed modes (0 <= j < M, j1 != j2).
    C_block : (4, 4) array
        Quadrature covariance matrix for (X_{E1}, X_{P1}, X_{E2}, X_{P2}).
    S_E, S_P : (M,) arrays or None
        Static structure factors for each mode. If None, taken as 1.

    Returns
    -------
    G2_total : (Nx, Nx) array
        Full G^2 including the two-mode block.
    G2_base  : (Nx, Nx) array
        Original analytic_G2 (all modes independent).
    G2_block : (Nx, Nx) array
        Contribution from just the (E1,P1,E2,P2) block.
    """

    xs = np.asarray(xs)
    fE = np.asarray(fE)
    fP = np.asarray(fP)
    C_block = np.asarray(C_block, dtype=float)

    M, Nx = fE.shape
    assert fP.shape == (M, Nx)
    assert C_block.shape == (4, 4)
    assert 0 <= j1 < M and 0 <= j2 < M and j1 != j2

    # Structure factors
    if S_E is None:
        S_E_arr = np.ones(M, dtype=float)
    else:
        S_E_arr = np.asarray(S_E, dtype=float)

    if S_P is None:
        S_P_arr = np.ones(M, dtype=float)
    else:
        S_P_arr = np.asarray(S_P, dtype=float)

    # 1) full base G2, using   existing Gaussian model
    G2_base = analytic_G2(xs, fE, fP, r, alpha_P, S_E=S_E_arr, S_P=S_P_arr)

    # 2) subtract the "old" contributions of j1, j2 (diagonal-only model)
    #    We reconstruct just their analytic_G2 pieces and subtract them.

    # Mask arrays so only j1,j2 are kept
    mask = np.zeros(M, dtype=bool)
    mask[j1] = True
    mask[j2] = True

    # zero out all other modes by passing alpha_P=0 and r=0 for them, etc.
    fE_2 = np.zeros_like(fE)
    fP_2 = np.zeros_like(fP)
    fE_2[mask, :] = fE[mask, :]
    fP_2[mask, :] = fP[mask, :]

    r_2 = np.asarray(r, dtype=float)
    if r_2.ndim == 0:
        r_2 = np.full(M, float(r_2))
    alpha_2 = np.zeros_like(alpha_P)  

    G2_j1j2_diag = analytic_G2(xs, fE_2, fP_2, r_2, alpha_2, S_E=S_E_arr, S_P=S_P_arr)

    # 3) build the new 4-mode block contribution from C_block

    G2_block = np.zeros((Nx, Nx), dtype=float)

    # Effective g_a(x) = sqrt(S)*f(x) for a = E1,P1,E2,P2
    gE1 = np.sqrt(S_E_arr[j1]) * fE[j1, :]
    gP1 = np.sqrt(S_P_arr[j1]) * fP[j1, :]
    gE2 = np.sqrt(S_E_arr[j2]) * fE[j2, :]
    gP2 = np.sqrt(S_P_arr[j2]) * fP[j2, :]

    gs = [gE1, gP1, gE2, gP2]

    for a in range(4):
        for b in range(4):
            G2_block += C_block[a, b] * np.outer(gs[a], gs[b])

    # 4) total = base - old (diag block) + new (full block)
    G2_total = G2_base - G2_j1j2_diag + G2_block

    return G2_total, G2_base, G2_block

def single_pair_quadrature_moments(r_j, alpha_j):
    """
    Compute XE2, XP2, XEP for a single Hawking pair j with
    squeezing r_j and coherent seed alpha_j in P.

    Returns
    -------
    XE2_j, XP2_j, XEP_j : floats
        Second moments <X_E^2>, <X_P^2>, <X_E X_P>.
    """
    r_j = float(r_j)
    alpha_j = complex(alpha_j)

    c = np.cosh(r_j)
    s = np.sinh(r_j)

    abs2 = np.abs(alpha_j)**2
    Re_alpha2 = np.real(alpha_j**2)

    # Number expectations
    nE = s**2 * (abs2 + 1.0)
    nP = c**2 * abs2 + s**2

    # Quadrature second moments X = a + a†
    XE2  = 2.0 * (s**2 * Re_alpha2) + 2.0 * nE + 1.0
    XP2  = 2.0 * (c**2 * Re_alpha2) + 2.0 * nP + 1.0

    # Cross term (negative for TMSV)
    XEP = -2.0 * c * s * (1.0 + abs2 + Re_alpha2)

    return XE2, XP2, XEP

def build_C_block_for_two_seeds(r, alpha_P,
                                j1, j2,
                                eta_EE=0.5,
                                eta_PP=0.0,
                                eta_EP=0.0,
                                eta_PE=0.0):
    """
    Build a 4x4 covariance block C_block for (X_E1, X_P1, X_E2, X_P2),
    starting from the standard Hawking moments for each pair and
    adding tunable quantum cross-correlations between the two seed modes.

    Parameters
    ----------
    r : float or (M,) array
        Squeezing parameters for all modes.
    alpha_P : (M,) array
        Coherent seeds in P modes for all modes.
    j1, j2 : int
        Indices of the two seed modes (0 <= j < M, j1 != j2).
    eta_EE, eta_PP, eta_EP, eta_PE : floats
        Dimensionless strengths of cross-correlations, in [-1,1] ideally.
        They are used as fractions of geometric means of local variances, e.g.
        <X_E1 X_E2> = eta_EE * sqrt(XE1_sq * XE2_sq).

    Returns
    -------
    C_block : (4,4) float array
        Covariance matrix over (X_E1, X_P1, X_E2, X_P2).
    """

    r = np.asarray(r, dtype=float)
    if r.ndim == 0:
        # broadcast scalar r to all modes if needed; we only use j1,j2
        # but this guarantees r[j1], r[j2] exist
        r = np.full_like(alpha_P, float(r), dtype=float)

    alpha_P = np.asarray(alpha_P, dtype=complex)

    # Local moments for pair 1
    XE1_sq, XP1_sq, XEP1 = single_pair_quadrature_moments(r[j1], alpha_P[j1])
    # Local moments for pair 2
    XE2_sq, XP2_sq, XEP2 = single_pair_quadrature_moments(r[j2], alpha_P[j2])

    # Cross-covariances: scale by geometric mean of variances
    XE1XE2 = eta_EE * np.sqrt(XE1_sq * XE2_sq)
    XP1XP2 = eta_PP * np.sqrt(XP1_sq * XP2_sq)
    XE1XP2 = eta_EP * np.sqrt(XE1_sq * XP2_sq)
    XE2XP1 = eta_PE * np.sqrt(XE2_sq * XP1_sq)

    C_block = np.array([
        [XE1_sq,   XEP1,    XE1XE2,  XE1XP2],
        [XEP1,     XP1_sq,  XE2XP1,  XP1XP2],
        [XE1XE2,   XE2XP1,  XE2_sq,  XEP2   ],
        [XE1XP2,   XP1XP2,  XEP2,    XP2_sq ]
    ], dtype=float)

    return C_block

def G2_with_two_seed_block(xs,
                           fE, fP,
                           r, alpha_P,
                           j1, j2,
                           C_block,
                           S_E=None,
                           S_P=None):
    """
    Replace the contributions of modes j1 and j2 in analytic_G2 by a
    general 4-mode Gaussian block over (E1,P1,E2,P2) with covariance C_block.

    Indices in C_block correspond to:
        0 -> X_{E1}, 1 -> X_{P1}, 2 -> X_{E2}, 3 -> X_{P2}.

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid.
    fE, fP : (M, Nx) arrays
        Exterior and partner mode functions.
    r : float or (M,) array
        Squeezing parameters for all modes (used for base G2).
    alpha_P : (M,) array
        Coherent seeds in P-modes for base G2.
    j1, j2 : int
        Mode indices for the two seed modes (0 <= j < M, j1 != j2).
    C_block : (4, 4) array
        Quadrature covariance matrix for (X_{E1}, X_{P1}, X_{E2}, X_{P2}).
    S_E, S_P : (M,) arrays or None
        Static structure factors for each mode. If None, taken as 1.

    Returns
    -------
    G2_total : (Nx, Nx) array
        Full G^2 including the two-mode block.
    G2_base  : (Nx, Nx) array
        Original analytic_G2 (all modes independent).
    G2_block : (Nx, Nx) array
        Contribution from just the (E1,P1,E2,P2) block.
    """

    xs = np.asarray(xs)
    fE = np.asarray(fE)
    fP = np.asarray(fP)
    C_block = np.asarray(C_block, dtype=float)

    M, Nx = fE.shape
    assert fP.shape == (M, Nx)
    assert C_block.shape == (4, 4)
    assert 0 <= j1 < M and 0 <= j2 < M and j1 != j2

    # Structure factors
    if S_E is None:
        S_E_arr = np.ones(M, dtype=float)
    else:
        S_E_arr = np.asarray(S_E, dtype=float)

    if S_P is None:
        S_P_arr = np.ones(M, dtype=float)
    else:
        S_P_arr = np.asarray(S_P, dtype=float)

    # 1) full base G2 (  original Gaussian model)
    r_arr = np.asarray(r, dtype=float)
    if r_arr.ndim == 0:
        r_arr = np.full(M, float(r_arr))
    G2_base = analytic_G2(xs, fE, fP, r_arr, alpha_P, S_E=S_E_arr, S_P=S_P_arr)

    # 2) subtract old diagonal-only contribution from modes j1,j2

    mask = np.zeros(M, dtype=bool)
    mask[j1] = True
    mask[j2] = True

    fE_2 = np.zeros_like(fE)
    fP_2 = np.zeros_like(fP)
    fE_2[mask, :] = fE[mask, :]
    fP_2[mask, :] = fP[mask, :]

    alpha_2 = np.zeros_like(alpha_P)
    alpha_2[mask] = alpha_P[mask]

    G2_old_block = analytic_G2(xs, fE_2, fP_2, r_arr, alpha_2, S_E=S_E_arr, S_P=S_P_arr)

    # 3) build new 4-mode block G2_block from C_block

    # Effective spatial profiles g_a(x) = sqrt(S)*f(x)
    gE1 = np.sqrt(S_E_arr[j1]) * fE[j1, :]
    gP1 = np.sqrt(S_P_arr[j1]) * fP[j1, :]
    gE2 = np.sqrt(S_E_arr[j2]) * fE[j2, :]
    gP2 = np.sqrt(S_P_arr[j2]) * fP[j2, :]

    gs = [gE1, gP1, gE2, gP2]

    G2_block = np.zeros((Nx, Nx), dtype=float)
    for a in range(4):
        for b in range(4):
            G2_block += C_block[a, b] * np.outer(gs[a], gs[b])

    # 4) total = base - old_diag + new_block
    G2_total = G2_base - G2_old_block + G2_block

    return G2_total, G2_base, G2_block

def plot_G2_three_panel(xs, G2_vac, G2_seeded, cmap='bwr', vmin=None, vmax=None):
    """
    Plot three imshow panels with a common colorbar:
        1) G²_vac (background)
        2) G²_seeded (background + seeds)
        3) ΔG² = G²_seeded - G²_vac

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid for both axes.
    G2_vac : (Nx, Nx) array
        Background G² (e.g. vacuum/Hawking-only).
    G2_seeded : (Nx, Nx) array
        Seeded G² with classical or quantum seed contributions.
    cmap : str
        Matplotlib colormap (default: 'turbo').
    vmin, vmax : float or None
        Color scale limits for all three plots. If None, automatic.
    """

    xs = np.asarray(xs)
    G2_vac = np.asarray(G2_vac)
    G2_seeded = np.asarray(G2_seeded)

    assert G2_vac.shape == G2_seeded.shape
    Nx = xs.size

    # differential map
    G2_diff = G2_seeded - G2_vac

    # choose vmin/vmax automatically if not provided
    if vmin is None or vmax is None:
        all_vals = np.concatenate([
            G2_vac.ravel(),
            G2_seeded.ravel(),
            G2_diff.ravel()
        ])
        vmin = np.percentile(all_vals, 1) if vmin is None else vmin
        vmax = np.percentile(all_vals, 99) if vmax is None else vmax

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, dpi=150)

    # 1) Vacuum map
    im0 = axes[0].imshow(
        G2_vac,
        origin='lower',
        extent=[xs[0], xs[-1], xs[0], xs[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )
    axes[0].set_title(r"$G^{(2)}_{\rm vac}$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("x'")

    # 2) Seeded map
    im1 = axes[1].imshow(
        G2_seeded,
        origin='lower',
        extent=[xs[0], xs[-1], xs[0], xs[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )
    axes[1].set_title(r"$G^{(2)}_{\rm seeded}$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("x'")

    # 3) Differential
    im2 = axes[2].imshow(
        G2_diff*5,
        origin='lower',
        extent=[xs[0], xs[-1], xs[0], xs[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )
    axes[2].set_title(r"$\Delta G^{(2)} = G^{(2)}_{\rm seeded} - G^{(2)}_{\rm vac}$")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("x'")
    for i in range(3):
        axes[i].axhline(0, color='k', linestyle='--')
        axes[i].axvline(0, color='k', linestyle='--')
        axes[i].set_xlim(-15, 15)
        axes[i].set_ylim(-15, 15)

    # Shared colorbar
    # We attach it to the last axes; all 3 images share same vmin/vmax so it's OK.
    cbar = fig.colorbar(im2, ax=axes, location='right', fraction=0.015, pad=0.02)
    cbar.set_label("$G^{(2)}$, a.u.")
    # plt.savefig('threeplot.png', bbox_inches='tight', dpi=300)
    # plt.savefig('threeplot.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    return fig, axes

def hawking_holevo_dual_rail(alpha_P, r, N_cut=10):
    """
    Compute the Holevo information χ for a binary coherent seed {0, α_P}
    through a single Hawking pair, as seen on the outside mode E.

    Model:
        - Hawking pair (E,P) is a two-mode squeezer with parameter r.
        - Interior code: α_P^(0) = 0, α_P^(1) = α_P (coherent in P).
        - Outside ensemble on E:
              ρ_E^(0) = ρ_th(n̄),
              ρ_E^(1) = D(β_E) ρ_th(n̄) D(β_E)†,
          where
              n̄    = sinh(r)^2,
              β_E   = sinh(r) * α_P*.
        - Holevo information (bits):
              χ = S( (ρ0 + ρ1)/2 ) - S(ρ0),
          where S is the von Neumann entropy (base 2).

    Parameters
    ----------
    alpha_P : complex
        Coherent seed amplitude in the partner mode P.
    r : float
        Two-mode squeezing parameter (Hawking strength).
    N_cut : int, optional
        Fock cutoff dimension (Hilbert space size).
        Must be large enough that truncation errors are negligible.
        A rough rule: N_cut ≳ n̄ + |β_E|^2 + 5.

    Returns
    -------
    chi_bits : float
        Holevo information in bits for the outside mode E.
    """

    alpha_P = complex(alpha_P)
    r = float(r)

    # Hawking parameters
    s = np.sinh(r)
    nbar = s**2
    beta_E = s * np.conjugate(alpha_P)

    # Single-mode operators
    a = destroy(N_cut)

    # Thermal state on E (for vacuum input)
    rho_th = thermal_dm(N_cut, nbar)

    # Displacement operator for E
    D_E = displace(N_cut, beta_E)

    # Codeword states on E
    rho0 = rho_th                    # |0>_P input
    rho1 = D_E * rho_th * D_E.dag()  # |α_P> input -> displaced thermal

    # Average state
    rho_avg = 0.5 * (rho0 + rho1)

    # Entropies (base 2)
    S_rho0  = entropy_vn(rho0, base=2)
    S_rho1  = entropy_vn(rho1, base=2)  # should equal S_rho0 numerically
    S_avg   = entropy_vn(rho_avg, base=2)

    # Holevo χ = S(ρ̄) - (1/2 S(ρ0) + 1/2 S(ρ1))
    chi_bits = S_avg - 0.5 * (S_rho0 + S_rho1)

    return chi_bits

def add_dual_mode_code_G2_full(xs,
                               G2_bg,
                               fE, fP,
                               j1, j2,
                               alpha,
                               p=0.5):
    """
    Add the full four-quadrant G^2 contribution of a dual-mode binary code
        |0_L> = |0,0>
        |1_L> = |alpha, alpha>
    on two global modes j1, j2, averaged over codewords with
        P(bit=1) = p, P(bit=0) = 1-p.

    Each code mode j has a global spatial intensity profile
        I_j(x) = |fE_j(x)|^2 + |fP_j(x)|^2,
    so the resulting ΔG^2 has features in all quadrants:
        - out–out (E–E) where both x,x'>0,
        - in–in  (P–P) where both x,x'<0,
        - out–in (E–P) and in–out (P–E) where signs differ.

    Classical model:
        N1 = B * N_on, N2 = B * N_on, with B in {0,1},
        N_on = |alpha|^2,
        P(B=1) = p.

    Then:
        E[N1] = E[N2] = p N_on,
        Var(N1) = Var(N2) = p(1-p) N_on^2,
        Cov(N1,N2)       = p(1-p) N_on^2.

    G^2 contribution:
        ΔG2(x,x') =
            Var(N1) I1(x) I1(x')
          + Var(N2) I2(x) I2(x')
          + Cov(N1,N2) [ I1(x) I2(x') + I2(x) I1(x') ].

    Parameters
    ----------
    xs : (Nx,) array
        Spatial grid (for sanity; not used explicitly).
    G2_bg : (Nx, Nx) array
        Background G^2 (e.g. from analytic_G2).
    fE, fP : (M, Nx) arrays
        Mode functions for exterior (E) and partner (P) modes.
    j1, j2 : int
        Indices of the two modes that define the dual-mode code.
    alpha : complex
        Coherent amplitude used in the |alpha, alpha> codeword.
    p : float
        Probability of sending the bright codeword |alpha, alpha>.
        Default is 0.5.

    Returns
    -------
    G2_total : (Nx, Nx) array
        G^2 including the dual-mode code contribution.
    G2_code  : (Nx, Nx) array
        Code-only contribution ΔG2.
    """

    xs = np.asarray(xs)
    G2_bg = np.asarray(G2_bg)
    fE = np.asarray(fE)
    fP = np.asarray(fP)

    M, Nx = fE.shape
    assert fP.shape == (M, Nx)
    assert G2_bg.shape == (Nx, Nx)
    assert 0 <= j1 < M and 0 <= j2 < M and j1 != j2

    alpha = complex(alpha)
    p = float(p)

    # "On" intensity per mode when the bright codeword is sent
    N_on = abs(alpha)**2

    # Classical statistics for N1,N2 (0 or N_on, always equal in this simple code)
    N1_mean = p * N_on
    N2_mean = p * N_on

    Var1 = p * (1.0 - p) * N_on**2  # Var(N1)
    Var2 = Var1                     # Var(N2)
    Cov12 = Var1                    # Cov(N1,N2) = Var(N1) in this model

    # Global spatial intensities for code modes j1 and j2: E+P
    I1 = np.abs(fE[j1, :])**2 + np.abs(fP[j1, :])**2  # (Nx,)
    I2 = np.abs(fE[j2, :])**2 + np.abs(fP[j2, :])**2  # (Nx,)

    # Build ΔG2
    G2_code = np.zeros_like(G2_bg, dtype=float)

    # Auto blobs: one per mode (diagonal in "mode index")
    G2_code += Var1 * np.outer(I1, I1)
    G2_code += Var2 * np.outer(I2, I2)

    # Cross blobs: off-diagonal in mode index
    G2_code += Cov12 * (np.outer(I1, I2) + np.outer(I2, I1))

    G2_total = G2_bg + G2_code
    return G2_total, G2_code

def simulate_hawking_G2(
    M=2,
    N_cut=6,
    T_H=0.1,
    c_s=1.0,
    m=1.0,
    L=20.0,
    Nx=80,
    k_min=0.2,
    k_max=1.2,
    alpha_P=0.5,
    phi_sq=0.0,
    sigma=0.5,
    x_out_min=2.0,
    x_out_max=7.0,
    alpha_geom=0.7,
    curv=0.0,
    use_phases=False,
    return_state=False,
):
    """
    Full Qutip simulation of G^2(x,x') for M Hawking pairs with Bogoliubov dispersion
    and coherent seeding in the partner modes.

    This version uses *envelope-only* mode functions by default:
        fE_j(x) = Gaussian envelope centered outside horizon
        fP_j(x) = Gaussian envelope centered inside horizon

    so that:
      - The sign of the cross-horizon ridge comes purely from the Bogoliubov minus sign
        in the two-mode squeezing (always negative).
      - It does not oscillate or flip with k.

    Optionally, set use_phases=True to include plane-wave phases with *correct*
    Hawking directions:
        fE_j(x) ~ env_out(x) * exp(+i k_j x)
        fP_j(x) ~ env_in (x) * exp(-i k_j x).

    Model:
      - M Hawking pairs (E_j, P_j), j=0..M-1.
      - Each pair is two-mode squeezed with k-dependent parameter r_k determined by
        a Bogoliubov dispersion and Hawking temperature T_H:
            ω_k = sqrt((c_s k)^2 + (k^2 / (2m))^2)
            n̄_k = 1 / (exp(ω_k / T_H) - 1)
            sinh^2 r_k = n̄_k
      - Initial state: E_j in vacuum, P_j in coherent state |alpha_P[j]>.
      - Global state: product coherent seed, then product of two-mode squeezers S_j(r_k[j]).
      - Field operator:
            δn(x) = Σ_j [ fE_j(x)(aE_j+aE_j†) + fP_j(x)(aP_j+aP_j†) ].
      - G^2(x,x') = <ψ| δn(x) δn(x') |ψ>, where ψ is the final state.

    Parameters
    ----------
    M : int
        Number of Hawking pairs.
    N_cut : int
        Local Fock cutoff for each mode.
    T_H : float
        Hawking temperature (in same energy units as ω_k).
    c_s : float
        Sound speed.
    m : float
        Mass (in units with ħ=1).
    L : float
        Spatial window length, xs ∈ [-L/2, L/2].
    Nx : int
        Number of spatial grid points.
    k_min, k_max : float
        Min and max wavenumbers for the M Hawking modes.
    alpha_P : complex or array_like
        Coherent seed amplitude(s) in the partner modes P_j.
        If scalar, same for all j; if array-like, length must be M.
    phi_sq : float
        Squeezing phase (radians). 0.0 → standard real two-mode squeezing.
    sigma : float
        Width of Gaussian envelopes for fE and fP.
    x_out_min, x_out_max : float
        Range of outside centers for E_j modes.
    alpha_geom : float
        Hydrodynamic tilt parameter mapping outside to inside centers:
            x_in_j = -alpha_geom * x_out_j + curv * x_out_j^2
    curv : float
        Curvature in the inside mapping.
    use_phases : bool
        If False (default): fE,fP are real Gaussian envelopes (no exp(ikx)).
        If True: fE ~ env * exp(+ikx), fP ~ env * exp(-ikx).
    return_state : bool
        If True, also return the final state psi.

    Returns
    -------
    xs : (Nx,) ndarray
        Spatial grid.
    G2 : (Nx, Nx) ndarray
        G^2(x,x') = <δn(x) δn(x')>.
    psi : Qobj (optional)
        Final global state |ψ>, returned if return_state=True.
    """

    # --------------------------------------------------------
    # 0. Helper functions: dispersion and Hawking r_k
    # --------------------------------------------------------
    def omega_k(k, c_s=1.0, m=1.0):
        """Bogoliubov-like dispersion: ω_k = sqrt((c_s k)^2 + (k^2 / (2m))**2)."""
        return np.sqrt((c_s * k)**2 + (k**2 / (2.0 * m))**2)

    def hawking_occupation(k, T_H, c_s=1.0, m=1.0):
        """Thermal occupation number: n̄_k = 1 / (exp(ω_k / T_H) - 1)."""
        w = omega_k(k, c_s=c_s, m=m)
        if T_H <= 0:
            return np.zeros_like(w)
        return 1.0 / (np.exp(w / T_H) - 1.0)

    def squeezing_r_from_nbar(nbar):
        """For two-mode squeezed vacuum, sinh^2 r = n̄."""
        return np.arcsinh(np.sqrt(nbar))

    # --------------------------------------------------------
    # 1. Spatial and k grids, squeezing profile
    # --------------------------------------------------------
    xs = np.linspace(-L/2, L/2, Nx)
    dx = xs[1] - xs[0]

    k_vals = np.linspace(k_min, k_max, M)
    nbar_k = hawking_occupation(k_vals, T_H=T_H, c_s=c_s, m=m)
    r_k = squeezing_r_from_nbar(nbar_k)  # (M,) array

    # --------------------------------------------------------
    # 2. Mode operators
    # --------------------------------------------------------
    def mode_destroy(m_idx):
        return tensor([
            destroy(N_cut) if k == m_idx else qeye(N_cut)
            for k in range(2 * M)
        ])

    a_list = [mode_destroy(m_idx) for m_idx in range(2 * M)]

    def aE(j): return a_list[2*j]
    def aP(j): return a_list[2*j+1]

    # --------------------------------------------------------
    # 3. Initial state: vacuum in E, coherent |alpha_P[j]> in P
    # --------------------------------------------------------
    vac_single = basis(N_cut, 0)

    # alpha_P can be scalar (same for all) or length M
    if np.isscalar(alpha_P):
        alpha_P_arr = np.full(M, complex(alpha_P), dtype=complex)
    else:
        alpha_P_arr = np.asarray(alpha_P, dtype=complex)
        if alpha_P_arr.shape[0] != M:
            raise ValueError("alpha_P must be scalar or length M")

    state_list = []
    for j in range(M):
        state_list.append(vac_single)                      # E_j
        state_list.append(coherent(N_cut, alpha_P_arr[j])) # P_j

    psi_init = tensor(state_list)

    # --------------------------------------------------------
    # 4. Global two-mode squeezing with k-dependent r_k
    # --------------------------------------------------------
    S_total = None
    phase = np.exp(1j * phi_sq)

    for j in range(M):
        aE_j = aE(j)
        aP_j = aP(j)

        rj = float(r_k[j])
        # H_sq_j = rj ( e^{iφ} a_E a_P - e^{-iφ} a_E† a_P† )
        H_sq_j = rj * (phase * aE_j * aP_j - np.conjugate(phase) * aE_j.dag() * aP_j.dag())
        S2_j = H_sq_j.expm()

        S_total = S2_j if S_total is None else S2_j * S_total

    psi = S_total * psi_init

    # --------------------------------------------------------
    # 5. Mode functions fE[j,x], fP[j,x]
    #    Default: real Gaussian envelopes, no phases.
    #    If use_phases=True: add exp(+ikx) on E, exp(-ikx) on P.
    # --------------------------------------------------------
    x_out_centers = np.linspace(x_out_min, x_out_max, M)
    x_in_centers  = -alpha_geom * x_out_centers + curv * (x_out_centers**2)

    def gaussian(x, x0, s):
        return np.exp(-(x - x0)**2 / (2.0 * s**2))

    fE = np.zeros((M, Nx), dtype=complex)
    fP = np.zeros((M, Nx), dtype=complex)

    for j in range(M):
        x0_out = x_out_centers[j]
        x0_in  = x_in_centers[j]
        k_j    = k_vals[j]

        env_out = gaussian(xs, x0_out, sigma)
        env_in  = gaussian(xs, x0_in,  sigma)

        if use_phases:
            # Hawking-like directions: E ~ e^{+ikx}, P ~ e^{-ikx}
            fE[j, :] = env_out * np.exp(1j * k_j * xs)
            fP[j, :] = env_in  * np.exp(-1j * k_j * xs)
        else:
            # Envelope-only: real Gaussians, no oscillations
            fE[j, :] = env_out
            fP[j, :] = env_in

        # normalize ∑ |f|^2 dx = 1
        normE = np.sqrt(np.sum(np.abs(fE[j, :])**2) * dx)
        normP = np.sqrt(np.sum(np.abs(fP[j, :])**2) * dx)
        if normE > 0:
            fE[j, :] /= normE
        if normP > 0:
            fP[j, :] /= normP

    # --------------------------------------------------------
    # 6. δn(x) = Σ_j [ fE_j(x)(aE_j+aE_j†) + fP_j(x)(aP_j+aP_j†) ]
    # --------------------------------------------------------
    def delta_n_op_ix(ix):
        op = 0
        for j in range(M):
            fE_val = fE[j, ix]
            fP_val = fP[j, ix]
            op += fE_val * aE(j) + fP_val * aP(j)
            op += np.conjugate(fE_val) * aE(j).dag() + np.conjugate(fP_val) * aP(j).dag()
        return op

    # 6. δn(x) operators (still called dn_x) 
    delta_n_ops = [delta_n_op_ix(ix) for ix in range(Nx)]

    # 7. Compute means <n(x)> to form *connected* G^2
    n_means = np.array([expect(op, psi).real for op in delta_n_ops])

    G2 = np.zeros((Nx, Nx), dtype=float)

    for ix in range(Nx):
        dn_x = delta_n_ops[ix]
        for jx in range(Nx):
            dn_xp = delta_n_ops[jx]
            # full <n(x) n(x')>
            val = expect(dn_x * dn_xp, psi).real
            # subtract mean-mean to get connected G^2
            G2[ix, jx] = val - n_means[ix] * n_means[jx]

    if return_state:
        return xs, G2, psi
    else:
        return xs, G2

def simulate_hawking_G2_dual_mode_code(
    M=2,
    N_cut=6,
    T_H=0.1,
    c_s=1.0,
    m=1.0,
    L=20.0,
    Nx=80,
    k_min=0.2,
    k_max=1.2,
    # background seeds (same in both codewords)
    alpha_P_bg=0.0,
    # dual-mode code parameters (on/off in two P modes)
    j1=0,
    j2=1,
    alpha_code=1.0,
    p_code=0.5,
    # squeezing phase
    phi_sq=0.0,
    # mode geometry
    sigma=0.5,
    x_out_min=2.0,
    x_out_max=7.0,
    alpha_geom=0.7,
    curv=0.0,
    # mode phases or envelopes only
    use_phases=False,
    return_states=False,
):
    """
    Hawking G^2 with a dual-mode classical on/off code on two partner modes P_j1, P_j2.

    Codewords:
        |psi_0> : both seed modes "off" (P_j1, P_j2 have alpha = 0)
        |psi_1> : both seed modes "on"  (P_j1, P_j2 have alpha = alpha_code)

    Mixed state:
        rho = (1 - p_code) |psi_0><psi_0| + p_code |psi_1><psi_1|

    This classical mixture produces *connected* cross-mode blobs in G^2_c
    for E1-P2, E2-P1, E1-E2, etc.

    Parameters
    ----------
    M : int
        Number of Hawking pairs (E_j, P_j).
    N_cut : int
        Local Fock cutoff per mode.
    T_H : float
        Hawking temperature (same units as omega_k).
    c_s, m : float
        Bogoliubov dispersion parameters (c_s: sound speed, m: mass).
    L : float
        Spatial window length; xs ∈ [-L/2, L/2].
    Nx : int
        Number of spatial grid points.
    k_min, k_max : float
        Range of k for the M modes (linearly spaced).
    alpha_P_bg : complex or array-like
        Background coherent seed in P modes, present in BOTH codewords.
        If scalar, used for all P_j; if array-like, length must be M.
    j1, j2 : int
        Indices of the two partner modes that carry the dual-mode code.
    alpha_code : complex
        Seed amplitude for the "on" codeword in those two modes.
    p_code : float
        Probability of the bright codeword (|alpha,alpha>).
    phi_sq : float
        Squeezing phase (radians).
    sigma : float
        Width of Gaussian envelopes.
    x_out_min, x_out_max : float
        Range for outside centers of E_j modes.
    alpha_geom, curv : float
        Geometric mapping from outside to inside centers:
          x_in_j = -alpha_geom * x_out_j + curv * x_out_j^2.
    use_phases : bool
        If False: use real Gaussian envelopes only (no exp(ikx)).
        If True : fE ~ env * exp(+ikx), fP ~ env * exp(-ikx).
    return_states : bool
        If True, also return |psi_0>, |psi_1|, rho for debugging.

    Returns
    -------
    xs : (Nx,) ndarray
        Spatial grid.
    G2_conn : (Nx, Nx) ndarray
        Connected G^2_c(x,x') for the dual-mode code mixture.
    rho : Qobj (optional)
        Density matrix rho, if return_states=True.
    psi0, psi1 : Qobj (optional)
        Pure codeword states, if return_states=True.
    """

    # --------------------------------------------------------
    # Helper functions: dispersion and Hawking r_k
    # --------------------------------------------------------
    def omega_k(k, c_s=1.0, m=1.0):
        """Bogoliubov-like dispersion: ω_k = sqrt((c_s k)^2 + (k^2 / (2m))**2)."""
        return np.sqrt((c_s * k)**2 + (k**2 / (2.0 * m))**2)

    def hawking_occupation(k, T_H, c_s=1.0, m=1.0):
        """Thermal occupation: n̄_k = 1 / (exp(ω_k / T_H) - 1)."""
        w = omega_k(k, c_s=c_s, m=m)
        if T_H <= 0:
            return np.zeros_like(w)
        return 1.0 / (np.exp(w / T_H) - 1.0)

    def squeezing_r_from_nbar(nbar):
        """Two-mode squeezed vacuum: sinh^2(r) = n̄."""
        return np.arcsinh(np.sqrt(nbar))

    # --------------------------------------------------------
    # Grids and squeezing profile
    # --------------------------------------------------------
    xs = np.linspace(-L/2, L/2, Nx)
    dx = xs[1] - xs[0]

    k_vals = np.linspace(k_min, k_max, M)
    nbar_k = hawking_occupation(k_vals, T_H=T_H, c_s=c_s, m=m)
    r_k = 0.35*np.ones(M)

    # --------------------------------------------------------
    # Mode operators
    # --------------------------------------------------------
    def mode_destroy(m_idx):
        return tensor([
            destroy(N_cut) if k == m_idx else qeye(N_cut)
            for k in range(2 * M)
        ])

    a_list = [mode_destroy(m_idx) for m_idx in range(2 * M)]

    def aE(j): return a_list[2*j]
    def aP(j): return a_list[2*j+1]

    # --------------------------------------------------------
    # Background seeds alpha_P_bg
    # --------------------------------------------------------
    if np.isscalar(alpha_P_bg):
        alpha_bg_arr = np.full(M, complex(alpha_P_bg), dtype=complex)
    else:
        alpha_bg_arr = np.asarray(alpha_P_bg, dtype=complex)
        if alpha_bg_arr.shape[0] != M:
            raise ValueError("alpha_P_bg must be scalar or length M")

    # Build seeds for the two codewords:
    #   alpha_P0: background only (code modes off)
    #   alpha_P1: background + alpha_code on j1,j2 (code modes on)
    alpha_P0 = alpha_bg_arr.copy()
    alpha_P1 = alpha_bg_arr.copy()
    alpha_P1[j1] += alpha_code
    alpha_P1[j2] += alpha_code

    # --------------------------------------------------------
    # Initial product coherent states for each codeword
    # --------------------------------------------------------
    vac_single = basis(N_cut, 0)

    def build_psi_init(alpha_P_arr):
        state_list = []
        for j in range(M):
            state_list.append(vac_single)                    # E_j
            state_list.append(coherent(N_cut, alpha_P_arr[j]))  # P_j
        return tensor(state_list)

    psi_init0 = build_psi_init(alpha_P0)
    psi_init1 = build_psi_init(alpha_P1)

    # --------------------------------------------------------
    # Global squeezing unitary (same for both codewords)
    # --------------------------------------------------------
    S_total = None
    phase = np.exp(1j * phi_sq)

    for j in range(M):
        aE_j = aE(j)
        aP_j = aP(j)
        rj   = float(r_k[j])
        H_sq_j = rj * (phase * aE_j * aP_j
                       - np.conjugate(phase) * aE_j.dag() * aP_j.dag())
        S2_j = H_sq_j.expm()
        S_total = S2_j if S_total is None else S2_j * S_total

    psi0 = S_total * psi_init0
    psi1 = S_total * psi_init1

    # Mixed state: rho = (1-p) |psi0><psi0| + p |psi1><psi1|
    rho0 = ket2dm(psi0)
    rho1 = ket2dm(psi1)
    rho  = (1.0 - p_code) * rho0 + p_code * rho1

    # --------------------------------------------------------
    # Mode functions fE[j,x], fP[j,x]
    # --------------------------------------------------------
    x_out_centers = np.linspace(x_out_min, x_out_max, M)
    x_in_centers  = -alpha_geom * x_out_centers + curv * (x_out_centers**2)

    def gaussian(x, x0, s):
        return np.exp(-(x - x0)**2 / (2.0 * s**2))

    fE = np.zeros((M, Nx), dtype=complex)
    fP = np.zeros((M, Nx), dtype=complex)

    for j in range(M):
        x0_out = x_out_centers[j]
        x0_in  = x_in_centers[j]
        k_j    = k_vals[j]

        env_out = gaussian(xs, x0_out, sigma)
        env_in  = gaussian(xs, x0_in,  sigma)

        if use_phases:
            # Hawking-like directions: E ~ e^{+ikx}, P ~ e^{-ikx}
            fE[j, :] = env_out * np.exp(1j * k_j * xs)
            fP[j, :] = env_in  * np.exp(-1j * k_j * xs)
        else:
            # Envelope-only: real Gaussians
            fE[j, :] = env_out
            fP[j, :] = env_in

        # normalize ∑ |f|^2 dx = 1
        normE = np.sqrt(np.sum(np.abs(fE[j, :])**2) * dx)
        normP = np.sqrt(np.sum(np.abs(fP[j, :])**2) * dx)
        if normE > 0:
            fE[j, :] /= normE
        if normP > 0:
            fP[j, :] /= normP

    # --------------------------------------------------------
    # δn(x) = Σ_j [ fE_j(x)(aE_j+aE_j†) + fP_j(x)(aP_j+aP_j†) ]
    # --------------------------------------------------------
    def n_op_ix(ix):
        op = 0
        for j in range(M):
            fE_val = fE[j, ix]
            fP_val = fP[j, ix]
            op += fE_val * aE(j) + fP_val * aP(j)
            op += np.conjugate(fE_val) * aE(j).dag() + np.conjugate(fP_val) * aP(j).dag()
        return op

    n_ops = [n_op_ix(ix) for ix in range(Nx)]

    # --------------------------------------------------------
    # Connected G^2: G2_c(x,x') = <n(x)n(x')> - <n(x)><n(x')>
    # --------------------------------------------------------
    n_means = np.array([expect(op, rho).real for op in n_ops])

    G2_conn = np.zeros((Nx, Nx), dtype=float)

    for ix in range(Nx):
        nx_op = n_ops[ix]
        for jx in range(Nx):
            nxp_op = n_ops[jx]
            val = expect(nx_op * nxp_op, rho).real
            G2_conn[ix, jx] = val - n_means[ix] * n_means[jx]

    if return_states:
        return xs, G2_conn, rho, psi0, psi1
    else:
        return xs, G2_conn

def holevo_dual_mode_hawking(
    M=2,
    N_cut=6,
    T_H=0.1,
    c_s=1.0,
    m=1.0,
    k_min=0.2,
    k_max=1.2,
    # background seeds on partners (present in both codewords)
    alpha_P_bg=0.0,
    # dual-mode code parameters (two partner modes P_j1, P_j2)
    j1=0,
    j2=1,
    alpha_code=1.0,
    p_code=0.5,
    # squeezing phase
    phi_sq=0.0,
    # which exterior modes to keep for the channel output
    ext_js=None,
    # if True, return ρ0_E, ρ1_E, ρ_avg for inspection
    return_states=False,
    r_fixed=True,
):
    """
    Compute the Holevo information χ for a dual-mode classical on/off code
    {|0,0>, |α,α>} on two partner modes P_j1, P_j2, passed through a Hawking
    two-mode-squeezer channel, as seen on selected exterior modes E_j.

    Model:
      - There are M Hawking pairs (E_j, P_j), j = 0..M-1.
      - Each pair has k_j in [k_min, k_max] and squeezing parameter r_k[j]
        determined by a Bogoliubov dispersion and Hawking temperature T_H:
            ω_k = sqrt((c_s k)^2 + (k^2/(2m))^2)
            n̄_k = 1 / (exp(ω_k/T_H) - 1)
            sinh^2 r_k = n̄_k.
      - Initial state is a product of coherent states on the partners:
            |ψ_init(α_P)> = ⊗_j |0>_E_j ⊗ |α_P_j>_P_j.
      - Two codewords:
            code 0 (off): α_P0[j] = alpha_P_bg for all j
            code 1 (on):  α_P1[j] = alpha_P_bg for all j, but
                           α_P1[j1] += alpha_code
                           α_P1[j2] += alpha_code
      - Hawking unitary:
            S_total = ∏_j exp[r_k[j](e^{iφ} a_E_j a_P_j - e^{-iφ} a_E_j† a_P_j†)].
        We obtain |ψ_0> = S_total |ψ_init(α_P0)>, |ψ_1> similarly.
      - We look only at the EXTERIOR subsystem E_j (by default the two code modes’ E_j's),
        trace out all P_j (and any other E_j if desired), and define:
            ρ0_E = Tr_{rest}(|ψ_0><ψ_0|),
            ρ1_E = Tr_{rest}(|ψ_1><ψ_1|),
            ρ_avg = (1-p_code) ρ0_E + p_code ρ1_E.
      - Holevo information (bits) for this binary ensemble is:
            χ = S(ρ_avg) - [(1-p) S(ρ0_E) + p S(ρ1_E)],
        with von Neumann entropy S in base 2.

    Parameters
    ----------
    M : int
        Number of Hawking pairs (E_j, P_j).
    N_cut : int
        Local Fock cutoff per mode.
    T_H : float
        Hawking temperature.
    c_s, m : float
        Bogoliubov dispersion parameters.
    k_min, k_max : float
        Range of k for the M modes (linearly spaced).
    alpha_P_bg : complex or array-like
        Background coherent seed on the partners (present in both codewords).
        If scalar, used for all P_j; if array-like, must have length M.
    j1, j2 : int
        Indices of the two partner modes that carry the dual-mode code
        (must satisfy 0 <= j1,j2 < M and j1 != j2).
    alpha_code : complex
        Coherent amplitude used in the "on" codeword (added on both P_j1, P_j2).
    p_code : float
        Probability of the bright codeword |α,α>, with 0 <= p_code <= 1.
    phi_sq : float
        Squeezing phase φ (radians).
    ext_js : list[int] or None
        List of j indices of exterior modes E_j to keep as the channel output.
        If None, defaults to [j1, j2] (i.e. only E_j1 and E_j2).
        All other modes (all P_j and any non-selected E_j) are traced out.
    return_states : bool
        If True, also return (ρ0_E, ρ1_E, ρ_avg) for inspection.

    Returns
    -------
    chi_bits : float
        Holevo information χ in bits for the dual-mode code as seen on the
        chosen exterior subsystem.
    (ρ0_E, ρ1_E, ρ_avg) : Qobj triple (optional)
        Reduced density matrices for the two codewords and their average,
        if return_states=True.
    """

    # -----------------------------
    # 0. Helper functions
    # -----------------------------
    def omega_k(k, c_s=1.0, m=1.0):
        return np.sqrt((c_s * k)**2 + (k**2 / (2.0 * m))**2)

    def hawking_occupation(k, T_H, c_s=1.0, m=1.0):
        w = omega_k(k, c_s=c_s, m=m)
        if T_H <= 0:
            return np.zeros_like(w)
        return 1.0 / (np.exp(w / T_H) - 1.0)

    def squeezing_r_from_nbar(nbar):
        return np.arcsinh(np.sqrt(nbar))

    # -----------------------------
    # 1. k grid and squeezing r_k
    # -----------------------------
    k_vals = np.linspace(k_min, k_max, M)
    nbar_k = hawking_occupation(k_vals, T_H=T_H, c_s=c_s, m=m)
    r_k = squeezing_r_from_nbar(nbar_k)  # shape (M,)
    if r_fixed:
        r_k = np.ones(M)*0.35  # shape (M,)

    # -----------------------------
    # 2. Mode operators
    # -----------------------------
    def mode_destroy(m_idx):
        return tensor([
            destroy(N_cut) if k == m_idx else qeye(N_cut)
            for k in range(2 * M)
        ])

    a_list = [mode_destroy(m_idx) for m_idx in range(2 * M)]

    def aE(j): return a_list[2*j]
    def aP(j): return a_list[2*j+1]

    # -----------------------------
    # 3. Background seeds α_P_bg
    # -----------------------------
    if np.isscalar(alpha_P_bg):
        alpha_bg_arr = np.full(M, complex(alpha_P_bg), dtype=complex)
    else:
        alpha_bg_arr = np.asarray(alpha_P_bg, dtype=complex)
        if alpha_bg_arr.shape[0] != M:
            raise ValueError("alpha_P_bg must be scalar or length M.")

    if not (0 <= j1 < M and 0 <= j2 < M and j1 != j2):
        raise ValueError("j1 and j2 must be distinct integers in [0, M).")

    # codeword 0: "off"
    alpha_P0 = alpha_bg_arr.copy()
    # codeword 1: "on"
    alpha_P1 = alpha_bg_arr.copy()
    alpha_P1[j1] += alpha_code
    alpha_P1[j2] += alpha_code

    # -----------------------------
    # 4. Build initial product coherent states
    # -----------------------------
    vac_single = basis(N_cut, 0)

    def build_psi_init(alpha_P_arr):
        state_list = []
        for j in range(M):
            state_list.append(vac_single)                         # E_j
            state_list.append(coherent(N_cut, alpha_P_arr[j]))    # P_j
        return tensor(state_list)

    psi_init0 = build_psi_init(alpha_P0)
    psi_init1 = build_psi_init(alpha_P1)

    # -----------------------------
    # 5. Global Hawking squeezing unitary S_total
    # -----------------------------
    S_total = None
    phase = np.exp(1j * phi_sq)

    for j in range(M):
        aE_j = aE(j)
        aP_j = aP(j)
        rj   = float(r_k[j])
        H_sq_j = rj * (phase * aE_j * aP_j
                       - np.conjugate(phase) * aE_j.dag() * aP_j.dag())
        S2_j = H_sq_j.expm()
        S_total = S2_j if S_total is None else S2_j * S_total

    psi0 = S_total * psi_init0
    psi1 = S_total * psi_init1

    rho0_full = ket2dm(psi0)
    rho1_full = ket2dm(psi1)

    # -----------------------------
    # 6. Choose exterior modes to keep
    # -----------------------------
    if ext_js is None:
        ext_js = [j1, j2]
    else:
        ext_js = list(ext_js)

    for j in ext_js:
        if not (0 <= j < M):
            raise ValueError("All ext_js indices must be in [0, M).")

    # subsystems in the full tensor product:
    # index 0: E_0, 1: P_0, 2: E_1, 3: P_1, ...
    keep_subsystems = sorted([2*j for j in ext_js])  # keep only those E_j

    # -----------------------------
    # 7. Reduced states on exterior subsystem
    # -----------------------------
    rho0_E = rho0_full.ptrace(keep_subsystems)
    rho1_E = rho1_full.ptrace(keep_subsystems)

    # -----------------------------
    # 8. Average state and Holevo χ
    # -----------------------------
    p = float(p_code)

    rho_avg = (1.0 - p) * rho0_E + p * rho1_E

    # von Neumann entropies in bits
    S0 = entropy_vn(rho0_E, base=2)
    S1 = entropy_vn(rho1_E, base=2)
    S_avg = entropy_vn(rho_avg, base=2)

    chi_bits = S_avg - ((1.0 - p) * S0 + p * S1)

    if return_states:
        return chi_bits, (rho0_E, rho1_E, rho_avg)
    else:
        return chi_bits
