# -*- coding: utf-8 -*-
"""
ENCODING: utf-8
FILE: Schrodinger_Glaser_v3.py
PROJECT: Splitting-Step Quantum
AUTHOR: Léonard HUANG Hui-Dong
VERSION: 0.0
CREATED: 2025-07-23
LAST MODIFIED: 2026-06-06

DESCRIPTION:
This script implements a split-step Fourier pseudo-spectral solver of 
Schrödinger equation for timely evolving a non-relativistic scalar 
wavepacket under a given Glaser-type magnetic lens field.
"""

#%% --- Imports ---
import os
import numpy as np
from mkl_fft import fftn, ifftn
from scipy.ndimage import map_coordinates
from scipy.integrate import trapezoid
from scipy.special import factorial, eval_genlaguerre
from scipy.constants import e, m_e, hbar, pi
I3 = np.eye(3)                    	# 3x3 identity matrix
from time import time
from tqdm import tqdm

#%% --- Model settings ---
print("\nConfiguring the simulation...")
# --- physical constants ---
q = -e                              # electron charge (C)
m = m_e                             # electron mass (kg)
# --- external fields ---
B0 = 0.02                           # peak field strength (T)
wm = (2*hbar/abs(q*B0))**0.5        # magnetic width (m)
tc = 2*pi*m/abs(q*B0)               # cyclotron period (s)
d0 = 160*wm#3.1623e-6               # Lorentzian radius (m), Glaser field turns to uniform if d0 = np.inf
zc = 0.0e-6                         # focal point (m)
print("External fields:")
print(f'''  Glaser field Bz(z) = B0/(1+((z-zc)/d0)**2):
    intensity maximum: B0 = {B0} T,
    magnetic distance: d0 = {d0*1e6} µm,
    magnetic centroid: zc = {zc*1e6} µm,''')
print(f'''  Corresponding to |B0| = {B0} T,
    magnetic width: {wm*1e9:g} nm,
    cyclotron period: {tc*1e12:g} ps.''')
# --- twisted electron (Landau state) ---
n, ell = 0, +3                      # radial and azimuthal quanta
wr = 1.0*wm                         # radical wave width (m)
px, py, pz = 0, 0, m*(160*wm)/(6*tc)# initial momentum (kg.m/s)
xp, yp, zp = 0, +6.0*wm, -80.0*wm   # initial position (m)
wz = 4.0*wm                         # longitudinal width (m)
print(f'''Twisted electron:
    quantum numbers: n = {n}, l = {ell},
    radial packet-width: {wr*1e9:g} nm, RMSr = {wr*(2*n+abs(ell)+1)**0.5/(2**0.5)*1e9:g} nm,
    axial packet-length: {wz*1e9:g} nm, RMSz = {wz/(2**0.5)*1e9:g} nm,
    initial position: ({xp*1e9:g}, {yp*1e9:g}, {zp*1e9:g}) nm,
    initial velocity: ({px/(m):g}, {py/(m):g}, {pz/(m):g}) m/s,''')
p0 = (px**2 + py**2 + pz**2)**0.5
λdB = 2*pi*hbar/p0 if p0 > 0 else np.inf
print(f"    initial de Broglie wavelength λ = {λdB*1e9:g} nm.")
# --- numerical settings ---
Nx, Ny, Nz = 240, 480, 1200         # grid resolution
Lx, Ly, Lz = 30*wm, 60*wm, 300*wm   # domain size (m)
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz    # grid spacing (m)
Nt = int(100)                       # number of time steps
dt = (6*tc)/Nt                      # time step (s)
Lt = Nt * dt                        # total simulation time (s)
print(f'''Space discretization:
    space size: {Lx*1e6:g} x {Ly*1e6:g} x {Lz*1e6:g} (µm^3),
    space resolution: {Nx} x {Ny} x {Nz},
    voxel size: {dx*1e9:g} x {dy*1e9:g} x {dz*1e9:g} (nm^3),
Time discretization:
    time-step: {dt*1e12:g} ps, stable condition: < {m/(2*pi*hbar)/((Nx/Lx)**2+(Ny/Ly)**2+(Nz/Lz)**2)*1e12:g} ps,
    duration: {Lt*1e9:g} ns, total steps: {Nt}.''')
# --- output settings ---
outdir = "output-schrodinger-Glaser"# output directory
save_interval = 1                   # save data every 'save_interval' steps
print(f'''Output settings:
    save interval: {save_interval} steps,
    output directory: "{str(os.getcwd()).replace('\\','/')}/{outdir}/".''')

#%% --- Algorithm kernels ---
print(f"\nPreparing algorithm kernels...")
start_load_time = time()
#% --- Discrete spaces ---
# --- real space ---
x0, y0, z0 = -Lx/2, -Ly/2, -Lz/2                    # beginnings of the domain
x = np.linspace(x0, x0+Lx, Nx,
                endpoint=False, dtype=np.float32)   # x-axis
y = np.linspace(y0, y0+Ly, Ny,
                endpoint=False, dtype=np.float32)   # y-axis
z = np.linspace(z0, z0+Lz, Nz,
                endpoint=False, dtype=np.float32)   # z-axis
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')       # positions grid
# --- reciprocal space ---
kx = 2*pi*np.fft.fftfreq(Nx,d=dx).astype(np.float32)# kx-axis
ky = 2*pi*np.fft.fftfreq(Ny,d=dy).astype(np.float32)# ky-axis
kz = 2*pi*np.fft.fftfreq(Nz,d=dz).astype(np.float32)# kz-axis
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij') # wavenumbers grid
K_ = np.stack((KX, KY, KZ), axis=-1)                # wavenumber vector field
K2 = np.einsum('...i,...i->...', K_, K_)            # squared wavenumbers
del KX, KY, KZ
#% --- Algorithm functions ---
# --- math tools ---
def reproj(Xp, f):
    # physical coordinates which need interpolation
    Xi, Yi, Zi = Xp[...,0], Xp[...,1], Xp[...,2]
    # convert to grid coordinates
    idx_coords = [(Xi-x0)/dx, (Yi-y0)/dy, (Zi-z0)/dz]
    # displacement interpolation
    fp = map_coordinates(input=f, coordinates=idx_coords, order=5, mode='wrap', cval=0.0)
    # the 'order' of spline is 1, which is linear interpolation.
    # the 'wrap' mode is suitable for periodic boundary condition.
    return fp
def translate(Dx, Dy, Dz):
    def decorator(func):
        def wrapper(x, y, z):
            x_, y_, z_ = x - Dx, y - Dy, z - Dz
            return func(x_, y_, z_)
        return wrapper
    return decorator
def trapz(f):
    integral = trapezoid(
        trapezoid(
            trapezoid(f, x=z, axis=2), 
            x=y, axis=1), 
        x=x, axis=0)
    return integral
def grad(f):
    grad = ifftn(
        fftn(f, axes=(0, 1, 2))[...,None] * (1j*K_),
        axes=(0, 1, 2))
    return grad
def divg(f):
    divg = ifftn(np.einsum('...i,...i->...',
                  fftn(f, axes=(0, 1, 2)),
                  (1j*K_)), axes=(0, 1, 2))
    return divg
def curl(f):
    fft_f = fftn(f, axes=(0, 1, 2))
    curl = ifftn(np.cross((1j*K_), fft_f), axes=(0, 1, 2))
    return curl
# --- EM functions ---
def Glaser_fields(B0, d0, zc):
    _B = lambda z: B0 / (1 + ((z-zc)/d0)**2)
    _C = lambda z: B0 * (z-zc) / (d0 * (1 + ((z-zc)/d0)**2))**2
    As = lambda x,y,z:np.stack(
        (-y/2 * _B(z), 
         +x/2 * _B(z), 
         0*(z)
        ), axis=-1)
    Bs = lambda x,y,z:np.stack(
        (x * _C(z), 
         y * _C(z), 
         _B(z)
        ), axis=-1)
    Gamma = lambda x, y, z: _B(z) * (x*yp-xp*y) / 2
    return As, Bs, Gamma, _B
As, Bs, Gamma, _B = Glaser_fields(B0, d0, zc)
# --- QM functions ---
def generate_LGGpacket(n, l, wr, wz, px, py, pz, xp, yp, zp, A, Gamma):
    # in cylindrical coordinates
    r = lambda x, y: (x**2 + y**2)**0.5
    theta = lambda x, y: np.arctan2(y, x)
    er = lambda theta: np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
    etheta = lambda theta: np.stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=-1)
    ez = np.array([0, 0, 1])
    # the free electron wavefunction reads
    Cnl = ((factorial(n))/(np.pi*factorial(n+abs(l))))**0.5
    psi = lambda r, theta, z: \
        (Cnl/wr * (r/wr)**abs(l) * eval_genlaguerre(n, abs(l), (r/wr)**2) * np.exp(-0.5*(r/wr)**2) * np.exp(1j*l*theta)) * \
        ((np.pi**(-1/4))/(wz**0.5) * np.exp(-0.5*(z/wz)**2))
    dens = lambda r, z: \
        (Cnl**2 * r**(2*abs(l)) / wr**(2*abs(l)+2) * (eval_genlaguerre(n, abs(l), (r/wr)**2))**2 * np.exp(-(r/wr)**2)) * \
        ((np.pi**(-1/2))/wz * np.exp(-(z/wz)**2))
    # u = lambda r, theta: (((l*hbar/m) / np.where(r==0, np.inf, r**2) - (q*Bz)/(2*m)) * r)[...,None] * etheta(theta) + pz/m * ez
    u0 = lambda r, theta: ((l*hbar/m) / np.where(r==0, np.inf, r))[...,None] * etheta(theta)
    flux = lambda r, theta, z: dens(r, z)[...,None] * u0(r, theta)
    # projected in Cartesian coordinates, modified the center-of-mass, the momentum and the gauge phase-shift
    # Psi = lambda x, y, z: psi(r(x,y), theta(x,y), z)
    @translate(Dx=xp,Dy=yp,Dz=zp)
    def Psi_ungauged(x, y, z):
        return psi(r(x,y), theta(x,y), z) * np.exp(1j*(px*x+py*y+pz*z)/hbar)
    Psi = lambda x, y, z: Psi_ungauged(x,y,z) * np.exp(-1j*q*Gamma(x,y,z)/hbar)
    # Dens = lambda x, y, z: dens(r(x,y), z)
    @translate(Dx=xp,Dy=yp,Dz=zp)
    def Dens(x, y, z):
        return dens(r(x,y), z)
    # Flux = lambda x, y, z: flux(r(x,y), theta(x,y), z)
    @translate(Dx=xp,Dy=yp,Dz=zp)
    def Flux_rest(x, y, z):
        return flux(r(x,y), theta(x,y), z)
    u1 = lambda x, y, z: -(q/m) * A(x,y,z) + np.stack([px/m, py/m, pz/m],axis=-1)
    Flux = lambda x, y, z: Flux_rest(x,y,z) + Dens(x,y,z)[...,None] * u1(x, y, z)
    return Psi, Dens, Flux
def precession(Psi):
    '''
    solution of ODEs' IVP
    dX/dz = +B(z) * Y
    dY/dz = -B(z) * X
    '''
    theta = 0.5*q/m*dt * _B(z)
    Cz = np.cos(theta)
    Sz = np.sin(theta)
    # Xi = X * np.cos(theta) - Y * np.sin(theta)
    Xi = np.einsum('ijk,k->ijk', X, Cz) - np.einsum('ijk,k->ijk', Y, Sz)
    # Yi = X * np.sin(theta) + Y * np.cos(theta)
    Yi = np.einsum('ijk,k->ijk', X, Sz) + np.einsum('ijk,k->ijk', Y, Cz)
    idx_coords = [(Xi-x0)/dx, (Yi-y0)/dy, (Z-z0)/dz]
    Psi = map_coordinates(input=Psi, coordinates=idx_coords, 
                          order=3, mode='wrap', cval=0.0)
    return Psi
expK = np.exp(-0.25j*dt*hbar/m * K2).astype(np.complex64)
expU = np.exp(-0.5j*(dt/hbar)*(q**2/(2*m)*np.einsum(
    '...i,...i->...', As(X,Y,Z), As(X,Y,Z)) + q*0)).astype(np.complex64)
def strang(Psi):
    # if (F is not None) or (A_ is not None):
    #     A2 = np.einsum('...i,...i->...', A_, A_)
    #     expU = np.exp(-0.5j*(dt/hbar)*(q**2/(2*m)*A2 + q*F))
    # <1> 1st propagation
    Psi = ifftn(expK * fftn(Psi))
    # <2> 1st scattering
    Psi = expU * Psi
    # <3> advection
    # Psi = advection(Psi)
    Psi = precession(Psi)
    # <4> 2nd scattering
    Psi = expU * Psi
    # <5> 2nd propagation
    Psi = ifftn(expK * fftn(Psi))
    return Psi
def update_wavefunc(Psi):
    return strang(Psi)
def update_densities(Psi, A_):
    Dens = np.abs(Psi)**2
    Psi_conj = np.conj(Psi)
    grad_Psi = grad(Psi)
    Jcan = hbar/m * np.imag(Psi_conj[...,None] * grad_Psi)
    Jkin = Jcan - q/m * A_ * Dens[...,None]
    return Dens, Jkin, Jcan
def expected_position(Dens):
    x = trapz(Dens * X)
    y = trapz(Dens * Y)
    z = trapz(Dens * Z)
    return np.array([x, y, z], dtype=np.float32)
def expected_momentum(Flux):
    πx = trapz(Flux[...,0]) * m
    πy = trapz(Flux[...,1]) * m
    πz = trapz(Flux[...,2]) * m
    return np.array([πx, πy, πz], dtype=np.float32)
def expected_OAM(Flux):
    mFlux = m * Flux
    lx = trapz(Y*mFlux[...,2] - Z*mFlux[...,1])
    ly = trapz(Z*mFlux[...,0] - X*mFlux[...,2])
    lz = trapz(X*mFlux[...,1] - Y*mFlux[...,0])
    return np.array([lx, ly, lz], dtype=np.float32)
def expected_position_rms(Dens, pos):
    rx = (trapz(X**2 * Dens) - pos[0]**2)**0.5
    ry = (trapz(Y**2 * Dens) - pos[1]**2)**0.5
    rz = (trapz(Z**2 * Dens) - pos[2]**2)**0.5
    return np.array([rx, ry, rz], dtype=np.float32)
def evaluate_AM(pos, mom):
    lx = (pos[1]*mom[2] - pos[2]*mom[1])
    ly = (pos[2]*mom[0] - pos[0]*mom[2])
    lz = (pos[0]*mom[1] - pos[1]*mom[0])
    return np.array([lx, ly, lz], dtype=np.float32)
def evaluate_helicity(oam, mom):
    v = mom / m
    v_norm = (v @ v)**0.5
    if v_norm < 0.01:
        h = 0.0
    else:
        h = (oam/hbar)@(v/v_norm)
    return h
# --- iteration functions ---
def step_advance(Psi, A_, t):
    Psi = update_wavefunc(Psi)
    Dens, Jkin, Jcan = update_densities(Psi, A_)
    norm = trapz(Dens)
    Psi /= norm**0.5                                # normalized wavefunction
    Dens /= norm                                    # normalized densities
    Jkin /= norm                                    # normalized kinetical current densities
    Jcan /= norm                                    # normalized canonical current densities
    pos = expected_position(Dens)                   # position
    rms = expected_position_rms(Dens, pos)          # root-mean-square derivation of position
    Pkin = expected_momentum(Jkin)                  # kinetical momentum
    Pcan = expected_momentum(Jcan)                  # canonical momentum
    OAMk = expected_OAM(Jkin)                       # kinetical OAM (with repect to the origin)
    OAMc = expected_OAM(Jcan)                       # canonical OAM (with repect to the origin)
    Lkin = OAMk - evaluate_AM(pos, Pkin)            # intrinsic kinetical OAM
    Lcan = OAMc - evaluate_AM(pos, Pcan)            # intrinsic canonical OAM
    h = evaluate_helicity(oam=Lcan, mom=Pkin)       # helicity: intrinsic canonical OAM projected along kinetical momentum
    expectations = {
        't': t, 'pos': pos, 'rms': rms, 
        'Pkin': Pkin, 'Pcan': Pcan, 
        'OAMk': OAMk, 'OAMc': OAMc,
        'Lkin': Lkin, 'Lcan': Lcan, 'h': h,
    }
    return Psi, expectations
def save(step, Psi, expectations, A_=None, B_=None):
    if A_ is None and B_ is None:
        np.savez(
            os.path.join(outdir, f"step_{step:06d}.npz"),
            Psi=Psi,
            **expectations
        )
    elif A_ is None and B_ is not None:
        np.savez(
            os.path.join(outdir, f"step_{step:06d}.npz"),
            Psi=Psi, B_=B_,
            **expectations
        )
    elif A_ is not None and B_ is None:
        np.savez(
            os.path.join(outdir, f"step_{step:06d}.npz"),
            Psi=Psi, A_=A_,
            **expectations
        )
    else:
        np.savez(
            os.path.join(outdir, f"step_{step:06d}.npz"),
            Psi=Psi, A_=A_, B_=B_,
            **expectations
        )
history = {
    "xp": [], "yp": [], "zp": [], "t": [],
    "rx": [], "ry": [], "rz": [],
    "px": [], "py": [], "pz": [],
    "kx": [], "ky": [], "kz": [],
    "mx0":[], "my0":[], "mz0":[],
    "lx0":[], "ly0":[], "lz0":[],
    "mx": [], "my": [], "mz": [],
    "lx": [], "ly": [], "lz": [], "h": [],
    }
def record_history(t, pos, rms, Pkin, Pcan, OAMk, OAMc, Lkin, Lcan, h):
    history["t"].append(t)
    history["xp"].append(pos[0])
    history["yp"].append(pos[1])
    history["zp"].append(pos[2])
    history["rx"].append(rms[0])
    history["ry"].append(rms[1])
    history["rz"].append(rms[2])
    history["px"].append(Pkin[0])
    history["py"].append(Pkin[1])
    history["pz"].append(Pkin[2])
    history["kx"].append(Pcan[0])
    history["ky"].append(Pcan[1])
    history["kz"].append(Pcan[2])
    history["mx0"].append(OAMk[0])
    history["my0"].append(OAMk[1])
    history["mz0"].append(OAMk[2])
    history["lx0"].append(OAMc[0])
    history["ly0"].append(OAMc[1])
    history["lz0"].append(OAMc[2])
    history["mx"].append(Lkin[0])
    history["my"].append(Lkin[1])
    history["mz"].append(Lkin[2])
    history["lx"].append(Lcan[0])
    history["ly"].append(Lcan[1])
    history["lz"].append(Lcan[2])
    history["h"].append(h)
def save_history():
    np.savez(
        os.path.join(outdir, "history.npz"),
        t = np.array(history["t"]),
        xp = np.array(history["xp"]),
        yp = np.array(history["yp"]),
        zp = np.array(history["zp"]),
        rx = np.array(history["rx"]),
        ry = np.array(history["ry"]),
        rz = np.array(history["rz"]),
        px = np.array(history["px"]),
        py = np.array(history["py"]),
        pz = np.array(history["pz"]),
        kx = np.array(history["kx"]),
        ky = np.array(history["ky"]),
        kz = np.array(history["kz"]),
        mx0 = np.array(history["mx0"]),
        my0 = np.array(history["my0"]),
        mz0 = np.array(history["mz0"]),
        lx0 = np.array(history["lx0"]),
        ly0 = np.array(history["ly0"]),
        lz0 = np.array(history["lz0"]),
        mx = np.array(history["mx"]),
        my = np.array(history["my"]),
        mz = np.array(history["mz"]),
        lx = np.array(history["lx"]),
        ly = np.array(history["ly"]),
        lz = np.array(history["lz"]),
        h = np.array(history["h"]),
    )
# --- END OF KERNEL PREPARATION ---
end_load_time = time()
print(f"Kernel preparation completed in {end_load_time - start_load_time:.2f} seconds.")

#%% --- Initialization ---
print("\nInitializing simulation...")
start_init_time = time()
# --- prepare output directory ---
if not os.path.exists(outdir):
    os.makedirs(outdir)
# --- prepare diagnostic histories ---
# initial state
A_ = As(X,Y,Z).astype(np.float32)
Psi_func, _, _ = generate_LGGpacket(n, ell, wr, wz, px, py, pz, xp, yp, zp, As, Gamma)
Psi = Psi_func(X, Y, Z).astype(np.complex64)
Dens, Jkin, Jcan = update_densities(Psi, A_)
norm = trapz(Dens)
print(f"    initial norm: {norm:.6f}.")
Psi /= norm**0.5                                # normalized wavefunction
Dens /= norm                                    # normalized densities
Jkin /= norm                                    # normalized kinetical current densities
Jcan /= norm                                    # normalized canonical current densities
# initial expectations
t = 0.0                                         # time
pos = expected_position(Dens)                   # position
rms = expected_position_rms(Dens, pos)          # root-mean-square derivation of position
Pkin = expected_momentum(Jkin)                  # kinetical momentum
Pcan = expected_momentum(Jcan)                  # canonical momentum
OAMk = expected_OAM(Jkin)                       # kinetical OAM (with repect to the origin)
OAMc = expected_OAM(Jcan)                       # canonical OAM (with repect to the origin)
Lkin = OAMk - evaluate_AM(pos, Pkin)            # center-of-mass kinetical OAM
Lcan = OAMc - evaluate_AM(pos, Pcan)            # center-of-mass canonical OAM
h = evaluate_helicity(oam=Lcan, mom=Pkin)       # helicity: canonical OAM projected along kinetical momentum
# save initial state
expectations = {
    't': t, 'pos': pos, 'rms': rms, 
    'Pkin': Pkin, 'Pcan': Pcan, 
    'OAMk': OAMk, 'OAMc': OAMc,
    'Lkin': Lkin, 'Lcan': Lcan, 'h': h,
}
save(0, Psi, expectations, A_=A_, B_=Bs(X,Y,Z).astype(np.float32))
# output initial expectations
print(f'''Initial expectations:
    position: ({pos[0]*1e9:.1f}, {pos[1]*1e9:.1f}, {pos[2]*1e9:.1f}) nm,
    RMS: ({rms[0]*1e9:.1f}, {rms[1]*1e9:.1f}, {rms[2]*1e9:.1f}) nm,
    velocity: ({Pkin[0]/(m):.3g}, {Pkin[1]/(m):.3g}, {Pkin[2]/(m):.3g}) m/s,
    kOAM: ({Lkin[0]/hbar:.2f}, {Lkin[1]/hbar:.2f}, {Lkin[2]/hbar:.2f}) ħ,
    cOAM: ({Lcan[0]/hbar:.2f}, {Lcan[1]/hbar:.2f}, {Lcan[2]/hbar:.2f}) ħ,
    helicity: {h:0.2f}.''')
# record initial history
record_history(t, pos, rms, Pkin, Pcan, OAMk, OAMc, Lkin, Lcan, h)
# end initialization
end_init_time = time()
print(f"Initialization completed in {end_init_time - start_init_time:.2f} seconds.")

#%%  --- Time-evolution ---
print(f"\nRunning the Schrödinger solver with grid {Nx}x{Ny}x{Nz} on CPU...")
start_simu_time = time()
for step in tqdm(range(1, Nt+1), desc="Simulation progress", unit="step"):
    t = step * dt
    Psi, expectations = step_advance(Psi, A_, t)
    if step % save_interval == 0 or step == Nt:
        save(step, Psi, expectations)
    record_history(
        expectations['t'], expectations['pos'], expectations['rms'],
        expectations['Pkin'], expectations['Pcan'],
        expectations['OAMk'], expectations['OAMc'],
        expectations['Lkin'], expectations['Lcan'], expectations['h']
    )
save_history()
end_simu_time = time()
print(f"Simulation completed in {end_simu_time - start_simu_time:.2f} seconds.")