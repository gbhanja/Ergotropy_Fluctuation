import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from math import comb
from tqdm import tqdm
from joblib import Parallel, delayed
import os

print("Current working directory:", os.getcwd())

############
# parameters
############

N_arr = np.arange(2, 102, 2)
# nc = N_arr                           # charger size equal to battery size
ω = 1.0
ω0 = 1.0
g = 1.0

# Create data folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Filename with parameters
def make_filename():
    return f"data_N{N_arr[0]}-{N_arr[-1]}_step{N_arr[1]-N_arr[0]}_w{ω}_w0{ω0}_g{g}.npz"

# Full path
filename = os.path.join(data_folder, "CS_Kac_OP_full.npz")

print("Data file:", filename)

#############################
# Central-Spin Hamiltonian
#############################

def central_spin_chain(N, nc, ω, ω0, g):

    N = int(N)
    nc = int(nc)
   
# ======== Collective Battery operators ============

    JpB = qt.jmat(N / 2, '+')
    JmB = qt.jmat(N / 2, '-')
    JzB = qt.jmat(N / 2, 'z')

# ======== Collective Charger operators ============

    JpC = qt.jmat(nc / 2, '+')
    JmC = qt.jmat(nc / 2, '-')
    JzC = qt.jmat(nc / 2, 'z')
    
# ======== Identities =================

    IC = qt.qeye(nc+1)
    IB = qt.qeye(N+1)
    I_full = qt.tensor(IB, IC)

# ======== Embedded operators into full space =============

# ====== Battery operators ==========

    JpB_full = qt.tensor(JpB, IC)
    JmB_full = qt.tensor(JmB, IC)
    JzB_full = qt.tensor(JzB, IC)

# ===== Charger operators ===========

    JpC_full = qt.tensor(IB, JpC)
    JmC_full = qt.tensor(IB, JmC)
    JzC_full = qt.tensor(IB, JzC)

# ======== Battery Hamiltonian =============

    HB = ω0 * (JzB_full + 0.5 * N * I_full)
    
# ======== Charger Hamiltonian =============

    HC = ω * (JzC_full + 0.5 * nc * I_full)

# ======== Interaction Hamiltonian ===========

    Hint = (g / N) * (JpB_full * JmC_full + JmB_full * JpC_full)
    
# ======== Total Hamiltonian =====================

    H = HB + HC + Hint

# ======= Battery-only Hamiltonian ==============


    HB_local = ω0 * (JzB + 0.5 * N * IB)

    return H, HB, HB_local

###################
# Initial state
###################

def initial_state(N, nc):

    N = int(N)
    nc = int(nc)

# ========= Battery state: all spins DOWN (ground) ================
    
    psiB = qt.basis(N+1, N)

# ========= Charger state: all spins UP (excited) ==================
    
    psiC = qt.basis(nc+1, 0)

    return qt.tensor(psiB, psiC)


###########################
# Passive state moments
###########################

def passive_moments(r_vals, ω0):

    r = np.sort(np.maximum(r_vals,0))[::-1]
    r /= r.sum()

    N = len(r)-1

    E_pass = 0.0
    E2_pass = 0.0

    i = 0

    for k in range(N+1):

        E = k*ω0

        for _ in range(min(comb(N,k), len(r)-i)):
            E_pass += r[i]*E
            E2_pass += r[i]*E**2
            i += 1

        if i == len(r):
            break

    return E_pass, E2_pass


#########################################
# optimal charging time τ (maximum power)
#########################################

def compute_tau(N):

    nc = N

    H, HB, _ = central_spin_chain(N, nc, ω, ω0, g)
    
    psi0 = initial_state(N, nc)

    t_max =  10 / g

    tlist_local = np.linspace(0.01, t_max, 1000)

    opts = {
        "atol":1e-16, 
        "rtol":1e-14,
        "nsteps":100000}            ## ODE solver options

    res = qt.sesolve(H, psi0, tlist_local, e_ops=HB, options=opts)

    EB = np.array(res.expect[0])

    power = EB / tlist_local

    τ = tlist_local[np.argmax(power)]

    return τ

τ_list = Parallel(n_jobs=64)(delayed(compute_tau)(N) for N in tqdm(N_arr, desc="Running simulation 1"))


###########################################
# Calculation of ergotropy and fluctuations
###########################################

def compute_ergotropy(i, N):

    nc = N

    τ = τ_list[i]
    
    H, HB, HB_local = central_spin_chain(N, nc, ω, ω0, g)
    
    psi0 = initial_state(N, nc)

    opts = {
        "atol":1e-16, 
        "rtol":1e-14,
        "nsteps":100000}            ## ODE solver options
    
    res = qt.sesolve(H, psi0, [0, τ], options=opts)
    
    rho_b = res.states[-1].ptrace(0)

    E_B = qt.expect(HB_local, rho_b)

    # Charger spin diagnostics

    # charger reduced density matrix
    rho_c = res.states[-1].ptrace(1)

# charger spin number probabilities
    photon_pop = np.real(np.diag(rho_c.full()))

# occupation of highest level
    edge_population = photon_pop[0]

    JzC = qt.jmat(nc / 2, 'z')

    n_C = (JzC + 0.5 * nc * qt.qeye(nc + 1))


# average charger spin number
    n_mean = qt.expect(n_C, rho_c)

# highest occupied level above tolerance
    tol = 1e-10
    occupied = np.where(photon_pop > tol)[0]

    if len(occupied) == 0:
        highest_occupied = 0
    else:
        highest_occupied = nc - occupied[0]


# Charger Single-spin Bloch vector
    JpB = qt.jmat(N / 2, '+')
    JmB = qt.jmat(N / 2, '-')
    JzB = qt.jmat(N / 2, 'z')

    Sx = (JpB + JmB) / 2
    Sy = (JpB - JmB) / (2j)
    Sz = JzB

    rx = 2 * qt.expect(Sx, rho_b) / N
    ry = 2 * qt.expect(Sy, rho_b) / N
    rz = 2 * qt.expect(Sz, rho_b) / N

    # rho1 = qt.Qobj([[(1 + rz)/2, (rx - 1j*ry)/2],[(rx + 1j*ry)/2, (1 - rz)/2]])

    # spin_purity = rho1.purity() 

    spin_purity = (1 + rx**2 + ry**2 + rz**2)/2

    # battery_purity = rho_b.purity()
    
    r_vals, r_vecs = rho_b.eigenstates()

    # Clip numerical noise to prevent negative probabilities
    r_vals = np.maximum(r_vals, 0) 
    # Renormalize 
    r_vals = r_vals / np.sum(r_vals)
        
    idx = np.argsort(r_vals)[::-1]
    r_vals = r_vals[idx]
    r_vecs = [r_vecs[i] for i in idx]

    # ==========================================================
# Numerical excitation numbers of the eigenvectors of rho_b
# ==========================================================

# m_hat = excitation-number operator in the symmetric battery space
    m_hat = HB_local / ω0

# Excitation number of every eigenvector |z_i'>
    m_i = np.array([np.real_if_close(qt.expect(m_hat, vec)) for vec in r_vecs], dtype=float)

# ----------------------------------------------------------
# Ground-state excitation:
# excitation number of the dominant eigenvector |z_0'>
# ----------------------------------------------------------
    m_0 = m_i[0]

# ----------------------------------------------------------
# Residual excitation:
# sum of excitation numbers of all non-dominant eigenvectors
# ----------------------------------------------------------
    m_tilde = np.sum(m_i[1:])
    
    # Passive-state moments in the full 2^N Hilbert space
    E_pass, E2_pass = passive_moments(r_vals, ω0)

    E_erg = E_B - E_pass

    # Cross term
    cross = 0.0
    i = 0
    for k in range(N + 1):
        E = k * ω0
        for _ in range(min(comb(N, k), len(r_vals) - i)):
            cross += E * r_vals[i] * qt.expect(HB_local, r_vecs[i])
            i += 1
            if i == len(r_vals):
                break
        if i == len(r_vals):
            break

    W_2 = qt.expect(HB_local**2, rho_b) + E2_pass - 2 * cross

    ΔE2 = np.real_if_close(W_2 - E_erg**2)
    ΔE = np.sqrt(max(ΔE2, 0.0))

    Ratio = E_erg / E_B
    
    
    return N, τ, E_B, E_erg, Ratio, ΔE2, W_2, spin_purity, n_mean, edge_population, highest_occupied, m_0, m_tilde
    
results = Parallel(n_jobs=-1)(delayed(compute_ergotropy)(i, N) for i, N in enumerate(tqdm(N_arr, desc="Running simulation 2")))

############################
# LOAD or RUN
############################
if os.path.exists(filename):
    print("Loading data...")
    data = np.load(filename)

    N_arr = data["N"]
    τ_list = data["tau"]
    E_B_arr = data["Eb"]
    E_ergo = data["Eerg"]
    E_ratio = data["ratio"]
    E_var = data["variance"]
    E_W2 = data["W2"]
    Spin_Purity = data["spin_purity"]
    PhotonMean = data["photon_mean"]
    EdgePopulation = data["edge_population"]
    HighestOccupied = data["highest_occupied"]
    Gnd_excitation = data["m_0"]
    Res_excitation = data["m_tilde"]

else:
    print("Running simulation...")

    τ_list = Parallel(n_jobs=-1)(
        delayed(compute_tau)(N)
        for N in tqdm(N_arr, desc="τ computation")
    )

    results = Parallel(n_jobs=-1)(
        delayed(compute_ergotropy)(i, N)
        for i, N in enumerate(tqdm(N_arr, desc="Ergotropy"))
    )

    N_out, tau_out, Eb_out, Eerg_out, ratio_out, var_out, W2_out, spin_purity_out, n_mean_out, edge_population_out, highest_out, m_0_out, m_tilde_out = zip(*results)

    N_arr = np.array(N_out)
    τ_list = np.array(tau_out)
    E_B_arr = np.array(Eb_out)
    E_ergo = np.array(Eerg_out)
    E_ratio = np.array(ratio_out)
    E_var = np.array(var_out)
    E_W2 = np.array(W2_out)
    Spin_Purity = np.array(spin_purity_out)
    PhotonMean = np.array(n_mean_out)
    EdgePopulation = np.array(edge_population_out)
    HighestOccupied = np.array(highest_out)
    Gnd_excitation = np.array(m_0_out)
    Res_excitation = np.array(m_tilde_out)

    
    np.savez_compressed(
        filename,
        N=N_arr,
        tau=τ_list,
        Eb=E_B_arr,
        Eerg=E_ergo,
        ratio=E_ratio,
        variance=E_var,
        W2=E_W2,
        spin_purity=Spin_Purity,
        photon_mean=PhotonMean,
        edge_population=EdgePopulation,
        highest_occupied=HighestOccupied,
        gnd_excitation=Gnd_excitation,
        res_excitation=Res_excitation
    )

    print(f"Saved results to {filename}")
    print("Simulation completed successfully.")
