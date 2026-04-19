#!/usr/bin/env python3
"""
gui_unificada.py — FK + 5 métodos IK para LA_PATA_SOLA
=======================================================
Pestañas:
  1. Cinemática Directa  — sliders qc/qp/qr → posición + 4 matrices HTM
  2. Algebraico          — IK algebraica: 2 soluciones (rodilla+/-) con derivación paso a paso
  3. MTH                 — IK por inversión de matrices DH
  4. Geométrico          — IK geométrica analítica: 2 soluciones sin iteraciones
  5. Gradiente           — IK por descenso J^T con paso BB1 o fijo
  6. Newton              — IK por Newton-Raphson (pseudo-inversa)
  7. Comparación         — resuelve los 5 métodos simultáneamente

Restricción común: todos los ángulos se limitan a [−90°, +90°] = URDF limits.
FK compartida: cadena URDF directa (metros) → fk_chain / fk_pos.
Unidades FK: metros.  Unidades IK entrada: metros.

Uso:
    ros2 launch parcial2 launch_unificada.py
"""

import math, signal, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5 import QtWidgets, QtCore, QtGui

# =================================================================
#  CONSTANTES COMUNES
# =================================================================
JOINTS     = ["joint_c", "joint_p", "joint_r"]
LIMITS_DEG = {j: (-90.0, 90.0) for j in JOINTS}
QLIM_RAD   = math.pi / 2.0
Q_MIN      = -QLIM_RAD
Q_MAX      =  QLIM_RAD
PUBLISH_HZ = 50.0

# Parámetros geométricos del URDF LA_PATA_SOLA
z0   = 0.1439
dz_p = 0.075
a    = 0.0403
L1   = 0.3505
L2   = math.hypot(0.2905, -0.0010066)
k1   = -0.00365
k2   = 0.093563
phi  = 0.026978 + math.atan2(-0.0010066, 0.2905)
h    = z0 + dz_p
d    = a + k1 + k2

def clamp(v, lo, hi): return max(lo, min(hi, v))

# =================================================================
#  FK  —  cadena URDF exacta (metros)
# =================================================================
def _Rx(t):
    c,s=math.cos(t),math.sin(t)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]],np.float64)
def _Ry(t):
    c,s=math.cos(t),math.sin(t)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]],np.float64)
def _Rz(t):
    c,s=math.cos(t),math.sin(t)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],np.float64)
def _Tr(x,y,z):
    M=np.identity(4,np.float64); M[0,3]=x; M[1,3]=y; M[2,3]=z; return M
def _RPY(r,p,y): return _Rz(y)@_Ry(p)@_Rx(r)

_T_OC = _Tr(0,0,0.1439)
_T_OP = _Tr(-0.0403,0,0.075)@_RPY(math.pi/2,0,-math.pi/2)
_T_OR = _Tr(0.3505,-2.9381e-5,-0.00365)@_RPY(0,0,0.026978)
_T_OE = _Tr(0.2905,-0.0010066,0.093563)@_RPY(-0.18364,1.5696,1.3872)

def fk_chain(qc_deg, qp_deg, qr_deg):
    """FK exacta URDF. Retorna (frames, T_efector) en metros."""
    T01 = _T_OC@_Rz(-math.radians(qc_deg))
    T02 = T01@_T_OP@_Rz(-math.radians(qp_deg))
    T03 = T02@_T_OR@_Rz(-math.radians(qr_deg))
    T04 = T03@_T_OE
    return [("base → link_c   (T₀₁)",T01),
            ("base → link_p   (T₀₂)",T02),
            ("base → link_r   (T₀₃)",T03),
            ("base → efector  (T₀₄)",T04)], T04

def fk_pos(qc_deg, qp_deg, qr_deg):
    _,T=fk_chain(qc_deg,qp_deg,qr_deg); return T[0,3],T[1,3],T[2,3]

def fk_geom_rad(qc,qp,qr):
    """FK geométrica (rad) usada por gradient/newton."""
    return np.array([
        -d*math.cos(qc) - (L1*math.cos(qp)+L2*math.cos(qp+qr-phi))*math.sin(qc),
         d*math.sin(qc) - (L1*math.cos(qp)+L2*math.cos(qp+qr-phi))*math.cos(qc),
        h - L1*math.sin(qp) - L2*math.sin(qp+qr-phi)
    ], np.float64)

# =================================================================
#  IK  1 — ALGEBRAICO (analítico, 2 soluciones con derivación)
# =================================================================
def ik_algebraic_all(x, y, z):
    """
    IK algebraica. Devuelve lista de dicts con 2 soluciones (ramas de rodilla).
    Cada dict: {rama_rodilla, qc_deg, qp_deg, qr_deg, err_mm, dentro_limites, pasos}
    """
    rho_sq = x**2 + y**2
    rho    = math.sqrt(rho_sq)

    if rho_sq < d**2 - 1e-9:
        raise ValueError(
            f"Punto no alcanzable.\n"
            f"  ρ = √(x²+y²) = {rho:.5f} m < d = {d:.5f} m")

    s_sq  = max(rho_sq - d**2, 0.0)
    s_abs = math.sqrt(s_sq)

    # Solo rama principal de cadera (yc = -s)
    s   = s_abs
    yc  = -s
    qc  = math.atan2(yc, -d) - math.atan2(y, x)
    qc  = math.atan2(math.sin(qc), math.cos(qc))   # normalizar (-π,π]

    det_c    = d**2 + s**2
    cqc_show = (-d*x + yc*y) / det_c if det_c > 1e-12 else math.cos(qc)
    sqc_show = (-yc*x - d*y) / det_c if det_c > 1e-12 else math.sin(qc)

    u      = s_abs
    w      = h - z
    c2_num = u**2 + w**2 - L1**2 - L2**2
    c2_den = 2*L1*L2
    c2     = clamp(c2_num / c2_den, -1.0, 1.0)
    s2_abs = math.sqrt(max(0.0, 1.0 - c2**2))

    results = []
    for rama_rodilla, s2 in [("positiva", +s2_abs), ("negativa", -s2_abs)]:
        qr_eff = math.atan2(s2, c2)
        qp     = math.atan2(w, u) - math.atan2(L2*s2, L1 + L2*c2)
        qr     = qr_eff + phi

        A11 = L1 + L2*c2;  A12 = -L2*s2
        A21 = L2*s2;       A22 =  L1 + L2*c2
        det_p = A11*A22 - A12*A21
        cqp = (A22*u - A12*w) / det_p if abs(det_p) > 1e-12 else math.cos(qp)
        sqp = (A11*w - A21*u) / det_p if abs(det_p) > 1e-12 else math.sin(qp)

        qc_d = math.degrees(qc);  qp_d = math.degrees(qp);  qr_d = math.degrees(qr)
        within = abs(qc_d) <= 90.1 and abs(qp_d) <= 90.1 and abs(qr_d) <= 90.1

        xr, yr, zr = fk_geom_rad(qc, qp, qr)
        err = math.sqrt((x-xr)**2+(y-yr)**2+(z-zr)**2)*1e3

        pasos = (
            f"Solución: Rodilla = '{rama_rodilla}'\n"
            f"{'═'*62}\n\n"
            f"PASO 1 — Elevar al cuadrado (identidad trigonométrica):\n"
            f"  x²+y² = d²+s²\n"
            f"  {rho_sq:.6f} = {d**2:.6f} + s²\n"
            f"  s² = {s_sq:.6f}   →   s = {s_abs:.6f} m\n"
            f"       [yc = −s = {yc:+.6f} m]\n\n"
            f"PASO 2 — qc por sistema matricial 2×2:\n"
            f"  De la FK:  x = −d·cos(qc) − s·sin(qc)\n"
            f"              y = −s·cos(qc) + d·sin(qc)\n"
            f"  ┌ −d   −s ┐ ┌ cos qc ┐   ┌ x ┐\n"
            f"  └ −s    d ─┘ └ sin qc ┘ = └ y ┘\n"
            f"  det = d²+s² = {det_c:.6f}\n"
            f"  cos qc = (−d·x + yc·y) / det = {cqc_show:.6f}\n"
            f"  sin qc = (−yc·x − d·y) / det = {sqc_show:.6f}\n"
            f"  qc = atan2({sqc_show:.5f}, {cqc_show:.5f}) = {qc_d:+.4f}°\n\n"
            f"PASO 3 — Ley de cosenos (c₂, s₂):\n"
            f"  u = s = {u:.5f} m,  w = h−z = {w:.5f} m\n"
            f"  c₂ = (u²+w²−L1²−L2²) / (2·L1·L2)\n"
            f"     = ({c2_num:.5f}) / {c2_den:.5f} = {c2:+.5f}\n"
            f"  s₂ = ±√(1−c₂²) = ±{s2_abs:.5f}\n"
            f"  Rama rodilla = '{rama_rodilla}'  →  s₂ = {s2:+.5f}\n"
            f"  qr_eff = atan2({s2:.5f},{c2:.5f}) = {math.degrees(qr_eff):+.4f}°\n\n"
            f"PASO 4 — qp por sistema matricial 2×2:\n"
            f"  u = (L1+L2c₂)·cos(qp) − L2s₂·sin(qp)\n"
            f"  w =   L2s₂  ·cos(qp) + (L1+L2c₂)·sin(qp)\n"
            f"  [  {A11:.4f}    {A12:.4f} ]\n"
            f"  [  {A21:.4f}    {A22:.4f} ]\n"
            f"  det = L1²+L2²+2L1L2c₂ = {det_p:.5f}\n"
            f"  cos qp = (A₂₂·u − A₁₂·w) / det = {cqp:.6f}\n"
            f"  sin qp = (A₁₁·w − A₂₁·u) / det = {sqp:.6f}\n"
            f"  qp = atan2({sqp:.5f}, {cqp:.5f}) = {qp_d:+.4f}°\n\n"
            f"PASO 5 — qr:\n"
            f"  qr = qr_eff + φ = {math.degrees(qr_eff):+.4f}° + {math.degrees(phi):+.4f}°\n"
            f"     = {qr_d:+.4f}°\n\n"
            f"VERIFICACIÓN FK(IK):\n"
            f"  p_rec = ({xr:.5f}, {yr:.5f}, {zr:.5f}) m\n"
            f"  ‖error‖ = {err:.4f} mm"
        )

        results.append({
            'rama_rodilla':   rama_rodilla,
            'qc_deg':         qc_d,
            'qp_deg':         qp_d,
            'qr_deg':         qr_d,
            'err_mm':         err,
            'dentro_limites': within,
            'pasos':          pasos,
        })

    return results


def ik_algebraic_best(x, y, z):
    """Wrapper para comparación: retorna (qc°, qp°, qr°) de la mejor solución."""
    sols = ik_algebraic_all(x, y, z)
    # Elegir primera dentro de límites, si no la de menor error
    valid = [s for s in sols if s['dentro_limites']]
    best  = min(valid or sols, key=lambda s: s['err_mm'])
    if not best['dentro_limites']:
        raise ValueError(
            f"Algebraico: sin solución en ±90° "
            f"(err={best['err_mm']:.2f} mm, "
            f"qc={best['qc_deg']:.1f}° qp={best['qp_deg']:.1f}° qr={best['qr_deg']:.1f}°)")
    return best['qc_deg'], best['qp_deg'], best['qr_deg']


# =================================================================
#  IK  2 — MTH (inversión DH)
# =================================================================
_T_BASE_0 = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,143.9],[0,0,0,1]],np.float64)
_T_3_EF   = np.array([[-1e-5,-0.999635,-0.027018,290.44],
                       [-0.001185,0.027018,-0.999634,-6.83],
                       [0.999999,2.2e-5,-0.001185,-130.087],
                       [0,0,0,1]],np.float64)
_T_B0_INV  = np.linalg.inv(_T_BASE_0)
_T_3EF_INV = np.linalg.inv(_T_3_EF)
_D1=75.0; _L2_MM=350.499

def _dh(theta,d_,a,alpha):
    c,s=math.cos(theta),math.sin(theta); ca,sa=math.cos(alpha),math.sin(alpha)
    return np.array([[c,-s*ca,s*sa,a*c],[s,c*ca,-c*sa,a*s],[0,sa,ca,d_],[0,0,0,1]],np.float64)

def _fk_mth_mm(qc,qp,qr):
    A1=_dh(-math.radians(qc),_D1,0,math.pi/2)
    A2=_dh(math.pi+math.radians(qp),0,_L2_MM,0)
    A3=_dh(math.radians(qr),0,0,0)
    return _T_BASE_0@A1@A2@A3@_T_3_EF

def _solve_qcqp_mth(qr_deg,px_mm,py_mm,pz_mm,max_iter=200,alpha=0.6,tol=1e-6):
    qc,qp=0.,0.; eps=1e-5
    for _ in range(max_iter):
        T=_fk_mth_mm(qc,qp,qr_deg); p=T[:3,3]
        e=np.array([px_mm-p[0],py_mm-p[1],pz_mm-p[2]])
        if np.linalg.norm(e)<tol: break
        Tp=_fk_mth_mm(qc+eps,qp,qr_deg)[:3,3]; Tq=_fk_mth_mm(qc,qp+eps,qr_deg)[:3,3]
        J=np.column_stack([(Tp-p)/eps,(Tq-p)/eps])
        try: dq=np.linalg.lstsq(J,e,rcond=None)[0]
        except: break
        qc=clamp(qc+alpha*dq[0],-90.,90.)
        qp=clamp(qp+alpha*dq[1],-90.,90.)
    return qc,qp

def ik_mth(x, y, z):
    """IK por inversión MTH. Entrada en metros. Retorna (qc°,qp°,qr°)."""
    px,py,pz = x*1000,y*1000,z*1000
    def err3d(qr_deg):
        qcd,qpd=_solve_qcqp_mth(qr_deg,px,py,pz)
        T=_fk_mth_mm(qcd,qpd,qr_deg)
        return float(np.linalg.norm(T[:3,3]-np.array([px,py,pz])))
    grid=np.linspace(-89,89,120); errs=np.array([err3d(q) for q in grid])
    mins=[i for i in range(1,len(errs)-1)
          if errs[i]<errs[i-1] and errs[i]<errs[i+1] and errs[i]<500]
    if not mins: mins=[int(np.argmin(errs))]
    gr=(math.sqrt(5)-1)/2; best_err,best_sol=float('inf'),None
    for idx in mins[:6]:
        lo,hi=max(-89.,grid[idx]-3.),min(89.,grid[idx]+3.)
        for _ in range(80):
            m1=hi-gr*(hi-lo); m2=lo+gr*(hi-lo)
            if err3d(m1)<err3d(m2): hi=m2
            else: lo=m1
            if hi-lo<1e-8: break
        qr_s=(lo+hi)/2; qcd,qpd=_solve_qcqp_mth(qr_s,px,py,pz,max_iter=400)
        T=_fk_mth_mm(qcd,qpd,qr_s)
        err=float(np.linalg.norm(T[:3,3]-np.array([px,py,pz])))
        if err<best_err: best_err=err; best_sol=(qcd,qpd,qr_s)
        if best_err<0.01: break
    if best_sol is None or best_err>2.0:
        raise ValueError(f"MTH: sin solución (err={best_err:.2f} mm)")
    qc,qp,qr=best_sol
    for v,name in [(qc,"qc"),(qp,"qp"),(qr,"qr")]:
        if abs(v)>90.1: raise ValueError(f"MTH: {name}={v:.1f}° fuera de ±90°")
    return qc, qp, qr

# =================================================================
#  IK  3 — GEOMÉTRICO (analítico puro, 2 soluciones)
# =================================================================
def ik_geometric_all(x, y, z):
    """
    IK geométrica analítica (SIN iteraciones). Devuelve lista de dicts con
    2 soluciones (ramas de rodilla).
    Cada dict: {rama_rodilla, qc_deg, qp_deg, qr_deg, err_mm, dentro_limites,
                s, yc, u, w, c2, s2, qr_eff_deg}
    """
    rho = math.hypot(x, y)
    if rho < d:
        raise ValueError(
            f"Punto no alcanzable.\n"
            f"  ρ = √(x²+y²) = {rho:.5f} m\n"
            f"  Se requiere ρ ≥ d = {d:.5f} m")

    s_abs = math.sqrt(max(rho**2 - d**2, 0.0))

    # Rama de cadera principal (yc = -s)
    s  = s_abs
    yc = -s
    qc = math.atan2(yc, -d) - math.atan2(y, x)
    qc = math.atan2(math.sin(qc), math.cos(qc))   # normalizar a (-π,π]

    u = s
    w = h - z
    c2 = clamp((u**2 + w**2 - L1**2 - L2**2) / (2*L1*L2), -1.0, 1.0)
    s2_abs = math.sqrt(max(0.0, 1.0 - c2**2))

    results = []
    for rama_rodilla, s2 in [("positiva", +s2_abs), ("negativa", -s2_abs)]:
        qr_eff = math.atan2(s2, c2)
        qp     = math.atan2(w, u) - math.atan2(L2*s2, L1 + L2*c2)
        qr     = qr_eff + phi

        qc_d = math.degrees(qc)
        qp_d = math.degrees(qp)
        qr_d = math.degrees(qr)

        within = (abs(qc_d) <= 90.1 and
                  abs(qp_d) <= 90.1 and
                  abs(qr_d) <= 90.1)

        xr, yr, zr = fk_geom_rad(qc, qp, qr)
        err = math.sqrt((x-xr)**2 + (y-yr)**2 + (z-zr)**2) * 1e3

        results.append({
            'rama_rodilla':   rama_rodilla,
            'qc_deg':         qc_d,
            'qp_deg':         qp_d,
            'qr_deg':         qr_d,
            'err_mm':         err,
            'dentro_limites': within,
            's':              s_abs,
            'yc':             yc,
            'u':              u,
            'w':              w,
            'c2':             c2,
            's2':             s2,
            'qr_eff_deg':     math.degrees(qr_eff),
        })

    return results


def ik_geometric_best(x, y, z):
    """Wrapper para comparación: retorna (qc°, qp°, qr°) de la mejor solución."""
    sols = ik_geometric_all(x, y, z)
    valid = [s for s in sols if s['dentro_limites']]
    best  = min(valid or sols, key=lambda s: s['err_mm'])
    if not best['dentro_limites']:
        raise ValueError(
            f"Geométrico: sin solución en ±90° "
            f"(err={best['err_mm']:.2f} mm, "
            f"qc={best['qc_deg']:.1f}° qp={best['qp_deg']:.1f}° qr={best['qr_deg']:.1f}°)")
    return best['qc_deg'], best['qp_deg'], best['qr_deg']


# =================================================================
#  IK  4 — GRADIENTE (J^T, BB1 o fijo)
# =================================================================
def ik_gradient(x, y, z, max_iter=500, epsilon=1e-5, alpha_fixed=0.6,
                mode='bb', alpha_min=1e-6, alpha_max=2.0,
                safeguard_beta=0.5, safeguard_max_tries=10, max_step=0.5):
    """IK por descenso del gradiente (J^T·e). Retorna (qc°,qp°,qr°)."""
    xd=np.array([x,y,z])
    STARTS=[np.zeros(3),np.radians([20,20,20]),np.radians([-20,20,-20]),
            np.radians([20,-20,20]),np.radians([40,0,0]),np.radians([0,40,0]),
            np.radians([0,0,40]),np.radians([45,45,0]),np.radians([0,45,-45])]
    best_q, best_err = None, float('inf')
    for q0 in STARTS:
        q=np.array(q0,float)
        prev_q=prev_grad=None
        for _ in range(max_iter):
            f=fk_geom_rad(*q); e=xd-f; err=float(np.linalg.norm(e))
            if err<epsilon: break
            J=np.zeros((3,3))
            for j in range(3):
                dv=np.zeros(3); dv[j]=1e-6
                J[:,j]=(fk_geom_rad(*(q+dv))-fk_geom_rad(*(q-dv)))/(2e-6)
            d_=J.T@e; grad=-d_
            if mode=='bb' and prev_q is not None:
                s=q-prev_q; y_=grad-prev_grad
                sty=float(s@y_); sts=float(s@s)
                if sts>1e-14 and abs(sty)>1e-14 and sty>0:
                    alpha=clamp(sts/sty,alpha_min,alpha_max)
                else: alpha=alpha_fixed
            else: alpha=alpha_fixed
            step=d_*alpha; nrm=float(np.linalg.norm(step))
            if nrm>max_step: step*=max_step/nrm
            q_try=np.clip(q+step,Q_MIN,Q_MAX)
            prev_q=q.copy(); prev_grad=grad.copy(); q=q_try
        f=fk_geom_rad(*q); err=float(np.linalg.norm(xd-f))
        if all(abs(qi)<=QLIM_RAD+1e-6 for qi in q) and err<best_err:
            best_err=err; best_q=q.copy()
        if best_err<epsilon: break
    if best_q is None or best_err>1e-2:
        raise ValueError(f"Gradiente: sin solución en ±90° (err={best_err*1e3:.1f} mm)")
    return math.degrees(best_q[0]),math.degrees(best_q[1]),math.degrees(best_q[2])

# =================================================================
#  IK  5 — NEWTON (Newton-Raphson pseudo-inversa)
# =================================================================
def ik_newton(x, y, z, epsilon=1e-5, max_iter=500, max_step=0.5):
    """IK por Newton-Raphson. Retorna (qc°,qp°,qr°)."""
    xd=np.array([x,y,z])
    STARTS=[np.zeros(3),np.radians([20,20,20]),np.radians([-20,20,-20]),
            np.radians([20,-20,20]),np.radians([40,0,0]),np.radians([0,40,0]),
            np.radians([0,0,40]),np.radians([45,45,0]),np.radians([0,45,-45])]
    best_q, best_err = None, float('inf')
    for q0 in STARTS:
        q=np.array(q0,float)
        for _ in range(max_iter):
            f=fk_geom_rad(*q); e=xd-f; err=float(np.linalg.norm(e))
            if err<epsilon: break
            J=np.zeros((3,3))
            for j in range(3):
                dv=np.zeros(3); dv[j]=1e-6
                J[:,j]=(fk_geom_rad(*(q+dv))-fk_geom_rad(*(q-dv)))/(2e-6)
            dq,_,_,_=np.linalg.lstsq(J,e,rcond=None)
            nrm=float(np.linalg.norm(dq))
            if nrm>max_step: dq*=max_step/nrm
            q=np.clip(q+dq,Q_MIN,Q_MAX)
        f=fk_geom_rad(*q); err=float(np.linalg.norm(xd-f))
        if all(abs(qi)<=QLIM_RAD+1e-6 for qi in q) and err<best_err:
            best_err=err; best_q=q.copy()
        if best_err<epsilon: break
    if best_q is None or best_err>1e-2:
        raise ValueError(f"Newton: sin solución en ±90° (err={best_err*1e3:.1f} mm)")
    return math.degrees(best_q[0]),math.degrees(best_q[1]),math.degrees(best_q[2])

# =================================================================
#  UTILIDADES
# =================================================================
def rot_to_rpy(R):
    pitch=math.atan2(-R[2,0],math.sqrt(R[0,0]**2+R[1,0]**2))
    if abs(math.cos(pitch))<1e-9: yaw=math.atan2(-R[1,2],R[1,1]); roll=0.
    else: yaw=math.atan2(R[1,0],R[0,0]); roll=math.atan2(R[2,1],R[2,2])
    return roll,pitch,yaw

def fmt4(T):
    return "\n".join("  ".join(f"{v:+.4f}" for v in row) for row in T)

# =================================================================
#  NODO ROS 2
# =================================================================
class PataNode(Node):
    def __init__(self):
        super().__init__("gui_unificada_node")
        self.pub=self.create_publisher(JointState,"/joint_states",10)
        self._deg={j:0. for j in JOINTS}
        self.create_timer(1./PUBLISH_HZ,self._publish)
    def set_deg(self,j,d): self._deg[j]=clamp(d,-90.,90.)
    def get_deg(self,j):   return self._deg[j]
    def _publish(self):
        msg=JointState(); msg.header.stamp=self.get_clock().now().to_msg()
        msg.name=JOINTS[:]; msg.position=[math.radians(self._deg[j]) for j in JOINTS]
        self.pub.publish(msg)

# =================================================================
#  ESTILO  (dark, mismo look que los GUIs individuales)
# =================================================================
STYLE="""
QMainWindow,QWidget{background:#0a0a0a;color:#d0d0d0;
  font-family:"Consolas","Courier New",monospace;font-size:12px;}
QGroupBox{border:1px solid #252525;border-radius:7px;margin-top:10px;
  padding:8px 6px 6px;font-size:11px;font-weight:bold;color:#4a9eff;}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}
QLabel{color:#d0d0d0;}
QSlider::groove:horizontal{height:5px;background:#111;border-radius:3px;}
QSlider::handle:horizontal{background:#4a9eff;border:none;
  width:16px;height:16px;margin:-6px 0;border-radius:8px;}
QSlider::sub-page:horizontal{background:#4a9eff;border-radius:3px;}
QLineEdit{background:#111;border:1px solid #252525;border-radius:4px;
  color:#d0d0d0;padding:3px 5px;min-width:64px;}
QLineEdit:focus{border-color:#4a9eff;}
QDoubleSpinBox{background:#111;border:1px solid #252525;border-radius:4px;
  color:#d0d0d0;padding:3px 5px;}
QDoubleSpinBox:focus{border-color:#4a9eff;}
QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{width:16px;background:#1a1a1a;border-radius:2px;}
QPushButton{background:#1a1a1a;border:none;border-radius:5px;
  color:#d0d0d0;padding:6px 14px;font-size:12px;}
QPushButton:hover{background:#222;}
QPushButton:pressed{background:#4a9eff;color:#000;}
QListWidget{background:#060606;border:1px solid #252525;
  border-radius:4px;color:#5ec46e;font-size:12px;}
QListWidget::item{padding:3px 6px;}
QListWidget::item:selected{background:#111;color:#4a9eff;}
QPlainTextEdit{background:#060606;border:1px solid #252525;border-radius:4px;
  color:#5ec46e;font-family:"Consolas","Courier New",monospace;font-size:12px;}
QTabWidget::pane{border:1px solid #252525;border-radius:6px;margin-top:-1px;}
QTabBar::tab{background:#111;color:#d0d0d0;padding:6px 16px;
  border-radius:5px 5px 0 0;margin-right:3px;font-size:12px;}
QTabBar::tab:selected{background:#1a1a1a;color:#4a9eff;}
QTabBar::tab:hover{background:#181818;}
QScrollBar:vertical{background:#0a0a0a;width:8px;border-radius:4px;}
QScrollBar::handle:vertical{background:#1a1a1a;border-radius:4px;min-height:20px;}
QTableWidget{background:#060606;border:1px solid #252525;gridline-color:#1e1e1e;}
QTableWidget::item{padding:4px 6px;}
QTableWidget::item:selected{background:#1a1a1a;color:#4a9eff;}
QHeaderView::section{background:#111;color:#4a9eff;padding:4px;
  border:1px solid #252525;font-size:11px;}
"""

_META={
    "joint_c":("qc","Coxa/Yaw",  "#e05c5c"),
    "joint_p":("qp","Fémur/Pitch","#4a9eff"),
    "joint_r":("qr","Tibia/Rod.", "#5ec46e"),
}

def _slider(color):
    sl=QtWidgets.QSlider(QtCore.Qt.Horizontal)
    sl.setStyleSheet(f"""
        QSlider::groove:horizontal{{height:5px;background:#111;border-radius:3px;}}
        QSlider::handle:horizontal{{background:{color};border:none;
          width:16px;height:16px;margin:-6px 0;border-radius:8px;}}
        QSlider::sub-page:horizontal{{background:{color};border-radius:3px;}}""")
    return sl

def _spin_xyz(v, lo=-1.5, hi=1.5):
    sb=QtWidgets.QDoubleSpinBox()
    sb.setRange(lo,hi); sb.setDecimals(5); sb.setSingleStep(0.005); sb.setValue(v)
    return sb

# =================================================================
#  VENTANA PRINCIPAL
# =================================================================
class UnifiedWindow(QtWidgets.QMainWindow):

    def __init__(self, node):
        super().__init__()
        self.node=node
        self.setWindowTitle("LA_PATA_SOLA  —  FK + 5 métodos IK  |  ROS 2")
        self.setMinimumSize(1150, 740)
        self.setStyleSheet(STYLE)

        self.tabs=QtWidgets.QTabWidget()
        self.tabs.addTab(self._tab_fk(),              "  Cinemática Directa  ")
        self.tabs.addTab(self._tab_ik_multi("algebraic"), "  Algebraico  ")
        self.tabs.addTab(self._tab_ik("mth"),          "  MTH  ")
        self.tabs.addTab(self._tab_ik_multi("geometric"), "  Geométrico  ")
        self.tabs.addTab(self._tab_ik("gradient"),     "  Gradiente  ")
        self.tabs.addTab(self._tab_ik("newton"),       "  Newton  ")
        self.tabs.addTab(self._tab_compare(),          "  Comparación  ")
        self.setCentralWidget(self.tabs)
        self._refresh_fk()

    # ──────────────────────────────────────────────────────────────
    #  TAB 1 — CINEMÁTICA DIRECTA
    # ──────────────────────────────────────────────────────────────
    def _tab_fk(self):
        w=QtWidgets.QWidget(); outer=QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10,10,10,10); outer.setSpacing(14)
        left=QtWidgets.QVBoxLayout(); left.setSpacing(10)
        right=QtWidgets.QVBoxLayout(); right.setSpacing(10)

        grp_j=QtWidgets.QGroupBox("Ángulos articulares   [−90° ≤ q ≤ +90°]  (límites URDF)")
        vj=QtWidgets.QVBoxLayout(grp_j); vj.setSpacing(6)
        self._sl={}; self._ed={}

        for j in JOINTS:
            sym,name,color=_META[j]
            top=QtWidgets.QHBoxLayout(); top.setSpacing(6)
            lsym=QtWidgets.QLabel(sym)
            lsym.setStyleSheet(f"color:{color};font-size:18px;font-weight:bold;min-width:24px;")
            lsym.setAlignment(QtCore.Qt.AlignCenter)
            lname=QtWidgets.QLabel(name)
            lname.setStyleSheet(f"color:{color};font-size:12px;")
            ed=QtWidgets.QLineEdit("0.0"); ed.setFixedWidth(72); ed.setAlignment(QtCore.Qt.AlignRight)
            lunit=QtWidgets.QLabel("°"); lunit.setStyleSheet("color:#555;font-size:12px;")
            top.addWidget(lsym); top.addWidget(lname,1); top.addWidget(ed); top.addWidget(lunit)

            bot=QtWidgets.QHBoxLayout(); bot.setSpacing(4)
            llo=QtWidgets.QLabel("−90°"); llo.setStyleSheet("color:#444;font-size:10px;min-width:32px;")
            llo.setAlignment(QtCore.Qt.AlignRight)
            sl=_slider(color); sl.setMinimum(-900); sl.setMaximum(900); sl.setValue(0)
            lhi=QtWidgets.QLabel("+90°"); lhi.setStyleSheet("color:#444;font-size:10px;min-width:32px;")
            bot.addWidget(llo); bot.addWidget(sl,1); bot.addWidget(lhi)

            def _sl_cb(val,joint=j,edit=ed):
                edit.setText(f"{val/10.:.1f}"); self.node.set_deg(joint,val/10.); self._refresh_fk()
            def _ed_cb(joint=j,s=sl):
                try: v=float(self._ed[joint].text())
                except ValueError: return
                v=clamp(v,-90.,90.)
                s.blockSignals(True); s.setValue(int(round(v*10))); s.blockSignals(False)
                self._ed[joint].setText(f"{v:.1f}"); self.node.set_deg(joint,v); self._refresh_fk()
            sl.valueChanged.connect(_sl_cb); ed.returnPressed.connect(_ed_cb); ed.editingFinished.connect(_ed_cb)

            blk=QtWidgets.QVBoxLayout(); blk.setSpacing(2); blk.addLayout(top); blk.addLayout(bot)
            vj.addLayout(blk)
            if j!=JOINTS[-1]:
                ln=QtWidgets.QFrame(); ln.setFrameShape(QtWidgets.QFrame.HLine)
                ln.setStyleSheet("color:#1e1e1e;"); vj.addWidget(ln)
            self._sl[j]=sl; self._ed[j]=ed

        left.addWidget(grp_j)

        grp_pos=QtWidgets.QGroupBox("Posición del efector  [m]  —  FK directa")
        gp=QtWidgets.QGridLayout(grp_pos); gp.setSpacing(6)
        self._fk_lx=QtWidgets.QLabel("—"); self._fk_ly=QtWidgets.QLabel("—"); self._fk_lz=QtWidgets.QLabel("—")
        for i,(sym,ww,col) in enumerate([("x =",self._fk_lx,"#e05c5c"),
                                          ("y =",self._fk_ly,"#5ec46e"),
                                          ("z =",self._fk_lz,"#4a9eff")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:14px;font-weight:bold;"
                             f"font-family:'Consolas','Courier New',monospace;")
            gp.addWidget(lb,i,0); gp.addWidget(ww,i,1)
        left.addWidget(grp_pos)

        br=QtWidgets.QHBoxLayout(); br.setSpacing(8)
        b0=QtWidgets.QPushButton("⟳  Zero")
        bv=QtWidgets.QPushButton("✓  Verificar q=0")
        for b in(b0,bv): b.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Fixed)
        br.addWidget(b0); br.addWidget(bv); left.addLayout(br)
        b0.clicked.connect(self._do_zero); bv.clicked.connect(self._do_verify)
        left.addStretch(1)

        grp_mat=QtWidgets.QGroupBox("Matrices HTM  —  tf2_echo base_link <link>")
        vm=QtWidgets.QVBoxLayout(grp_mat)
        self._fk_lst=QtWidgets.QListWidget(); self._fk_lst.setFixedHeight(95)
        self._fk_txt=QtWidgets.QPlainTextEdit(); self._fk_txt.setReadOnly(True); self._fk_txt.setMinimumHeight(230)
        self._fk_lst.currentRowChanged.connect(self._show_fk_mat)
        vm.addWidget(self._fk_lst); vm.addWidget(self._fk_txt)
        right.addWidget(grp_mat,1)

        note=QtWidgets.QLabel(
            "  FK:  T₀₁ = T_OC · Rz(−qc)\n"
            "       T₀₂ = T₀₁ · T_OP · Rz(−qp)\n"
            "       T₀₃ = T₀₂ · T_OR · Rz(−qr)\n"
            "       T₀₄ = T₀₃ · T_OE  ← efector (fijo)\n\n"
            "  q=(0,0,0) → efector ≈ (−0.130, −0.641, +0.226) m\n"
            "  ← tf2_echo base_link efector  ✓")
        note.setStyleSheet("color:#3a3a4a;font-size:11px;font-family:'Consolas','Courier New',monospace;")
        note.setWordWrap(True); right.addWidget(note)
        outer.addLayout(left,43); outer.addLayout(right,57)
        return w

    # ──────────────────────────────────────────────────────────────
    #  TAB IK CON MÚLTIPLES SOLUCIONES (Algebraico y Geométrico)
    # ──────────────────────────────────────────────────────────────
    def _tab_ik_multi(self, method):
        """
        Pestaña IK para métodos que devuelven múltiples soluciones
        (algebraic y geometric). Muestra lista de soluciones seleccionable
        y derivación/pasos en el panel derecho.
        """
        w=QtWidgets.QWidget(); outer=QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10,10,10,10); outer.setSpacing(14)
        left=QtWidgets.QVBoxLayout(); left.setSpacing(8)
        right=QtWidgets.QVBoxLayout(); right.setSpacing(8)

        TITLES={
            "algebraic":(
                "IK Algebraica — derivación algebraica completa (2 soluciones)",
                "#e05c5c",
                "Derivación algebraica paso a paso",
                "① x²+y²=d²+s²  →  s=±√(x²+y²−d²)\n\n"
                "② Sist.2×2 en (cos qc, sin qc) → atan2\n\n"
                "③ c₂=(u²+w²−L1²−L2²)/(2L1L2)\n   s₂=±√(1−c₂²)  [2 ramas rodilla]\n\n"
                "④ Sist.2×2 en (cos qp, sin qp) → atan2\n\n"
                "⑤ qr = atan2(s₂,c₂) + φ",
            ),
            "geometric":(
                "IK Geométrica — analítica directa SIN iteraciones (2 soluciones)",
                "#5ec46e",
                "Ecuaciones geométricas de la solución seleccionada",
                "Vista superior (XY):\n"
                "  ρ = √(x²+y²)\n"
                "  s = √(ρ²−d²)\n"
                "  qc = atan2(−s,−d) − atan2(y,x)\n\n"
                "Plano sagital 2R:\n"
                "  u = s,  w = h−z\n"
                "  c₂ = (u²+w²−L1²−L2²)/(2·L1·L2)\n"
                "  s₂ = ±√(1−c₂²)\n"
                "  qr_eff = atan2(s₂,c₂)\n"
                "  qp = atan2(w,u) − atan2(L2·s₂, L1+L2·c₂)\n"
                "  qr = qr_eff + φ",
            ),
        }
        title_txt, btn_color, steps_label, desc_txt = TITLES[method]

        title=QtWidgets.QLabel(title_txt)
        title.setStyleSheet(f"color:{btn_color};font-size:13px;font-weight:bold;")
        title.setWordWrap(True); left.addWidget(title)

        # Entrada xyz
        x0,y0,z0_=fk_pos(0,0,0)
        grp_in=QtWidgets.QGroupBox("Posición objetivo del efector  [m]")
        g_in=QtWidgets.QGridLayout(grp_in); g_in.setSpacing(8)
        ix=_spin_xyz(x0); iy=_spin_xyz(y0); iz=_spin_xyz(z0_)
        for i,(sym,ws,col) in enumerate([("x =",ix,"#e05c5c"),("y =",iy,"#5ec46e"),("z =",iz,"#4a9eff")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ws.setStyleSheet(f"color:{col};background:#111;border:1px solid #252525;border-radius:4px;padding:3px 5px;")
            g_in.addWidget(lb,i,0); g_in.addWidget(ws,i,1)
        ref=QtWidgets.QLabel(f"Reposo (0,0,0): ({x0:.5f}, {y0:.5f}, {z0_:.5f}) m")
        ref.setStyleSheet("color:#3a5a3a;font-size:11px;"); g_in.addWidget(ref,3,0,1,2)
        left.addWidget(grp_in)

        method_lbl = "ALGEBRAICO" if method == "algebraic" else "GEOMÉTRICO"
        btn=QtWidgets.QPushButton(f"Calcular soluciones IK  [{method_lbl}]  →")
        btn.setStyleSheet(f"background:{btn_color};color:#000;font-weight:bold;"
                          f"font-size:13px;padding:9px 16px;border-radius:6px;")
        left.addWidget(btn)

        # Lista de soluciones
        grp_sol=QtWidgets.QGroupBox("Soluciones  (✓ dentro de ±90°  |  ✗ fuera del límite URDF)")
        gsol=QtWidgets.QVBoxLayout(grp_sol)
        sol_list=QtWidgets.QListWidget(); sol_list.setFixedHeight(100)
        sol_list.setStyleSheet(
            "QListWidget{background:#060606;border:1px solid #252525;border-radius:4px;"
            "font-size:12px;font-family:'Consolas','Courier New',monospace;}"
            "QListWidget::item{padding:4px 8px;}"
            "QListWidget::item:selected{background:#1a1a1a;color:#4a9eff;}")
        gsol.addWidget(sol_list); left.addWidget(grp_sol)

        # Resultado ángulos
        grp_res=QtWidgets.QGroupBox("Solución seleccionada  —  ángulos articulares  [±90°]")
        g_res=QtWidgets.QGridLayout(grp_res); g_res.setSpacing(6)
        oqc=QtWidgets.QLabel("—"); oqp=QtWidgets.QLabel("—"); oqr=QtWidgets.QLabel("—")
        oerr=QtWidgets.QLabel(""); oerr.setStyleSheet("color:#e05c5c;font-weight:bold;"); oerr.setWordWrap(True)
        for i,(sym,ww,col) in enumerate([("qc =",oqc,"#e05c5c"),("qp =",oqp,"#4a9eff"),("qr =",oqr,"#5ec46e")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:13px;font-weight:bold;"
                             f"font-family:'Consolas','Courier New',monospace;")
            g_res.addWidget(lb,i,0); g_res.addWidget(ww,i,1)
        g_res.addWidget(oerr,3,0,1,2)
        btn_apply=QtWidgets.QPushButton("▶  Aplicar solución → RViz2")
        btn_apply.setStyleSheet(f"background:{btn_color};color:#000;font-weight:bold;"
                                f"font-size:12px;padding:6px 14px;border-radius:5px;")
        g_res.addWidget(btn_apply,4,0,1,2)
        left.addWidget(grp_res)

        # Verificación
        grp_v=QtWidgets.QGroupBox("Verificación  —  FK( IK(p) )  [m]")
        g_v=QtWidgets.QGridLayout(grp_v); g_v.setSpacing(6)
        vx=QtWidgets.QLabel("—"); vy=QtWidgets.QLabel("—"); vz=QtWidgets.QLabel("—"); ve=QtWidgets.QLabel("—")
        for i,(sym,ww,col) in enumerate([("x_rec =",vx,"#4dd0e1"),("y_rec =",vy,"#4dd0e1"),
                                          ("z_rec =",vz,"#4dd0e1"),("‖err‖ =",ve,"#5ec46e")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:12px;font-family:'Consolas','Courier New',monospace;")
            g_v.addWidget(lb,i,0); g_v.addWidget(ww,i,1)
        left.addWidget(grp_v); left.addStretch(1)

        # Panel derecho: derivación/pasos
        grp_steps=QtWidgets.QGroupBox(steps_label)
        vs=QtWidgets.QVBoxLayout(grp_steps)
        steps_txt=QtWidgets.QPlainTextEdit(); steps_txt.setReadOnly(True)
        vs.addWidget(steps_txt); right.addWidget(grp_steps,2)

        # Matriz HTM
        grp_mat=QtWidgets.QGroupBox("T(base→efector)  tras IK  ←  tf2_echo")
        vm=QtWidgets.QVBoxLayout(grp_mat)
        ik_mat=QtWidgets.QPlainTextEdit(); ik_mat.setReadOnly(True); ik_mat.setMaximumHeight(200)
        vm.addWidget(ik_mat); right.addWidget(grp_mat,1)

        note=QtWidgets.QLabel(desc_txt)
        note.setStyleSheet("color:#3a3a4a;font-size:11px;font-family:'Consolas','Courier New',monospace;")
        note.setWordWrap(True); right.addWidget(note)
        outer.addLayout(left,44); outer.addLayout(right,56)

        # Estado interno de soluciones para esta pestaña
        solutions_store = []

        def _solve():
            px=ix.value(); py=iy.value(); pz=iz.value()
            sol_list.clear(); solutions_store.clear()
            oqc.setText("—"); oqp.setText("—"); oqr.setText("—")
            oerr.setText(""); steps_txt.clear(); ik_mat.clear()
            for ww in(vx,vy,vz,ve): ww.setText("—")
            try:
                if method == "algebraic":
                    sols = ik_algebraic_all(px, py, pz)
                else:
                    sols = ik_geometric_all(px, py, pz)
            except ValueError as e:
                oerr.setText(f"Sin solución:\n{e}"); return

            solutions_store.extend(sols)
            for i, sol in enumerate(sols):
                sym_ok = "✓" if sol['dentro_limites'] else "✗"
                col    = "#5ec46e" if sol['dentro_limites'] else "#e05c5c"
                txt = (f"[{i+1}] {sym_ok}  Rodilla: {sol['rama_rodilla']:<10}"
                       f"qc={sol['qc_deg']:+6.1f}°  qp={sol['qp_deg']:+6.1f}°  "
                       f"qr={sol['qr_deg']:+6.1f}°  err={sol['err_mm']:.3f}mm")
                item=QtWidgets.QListWidgetItem(txt)
                item.setForeground(QtGui.QColor(col))
                sol_list.addItem(item)

            # Auto-seleccionar primera válida
            for i, sol in enumerate(sols):
                if sol['dentro_limites']:
                    sol_list.setCurrentRow(i); return
            sol_list.setCurrentRow(0)

        def _on_selected(idx):
            if idx < 0 or idx >= len(solutions_store): return
            sol = solutions_store[idx]
            qc_d = sol['qc_deg']; qp_d = sol['qp_deg']; qr_d = sol['qr_deg']

            oqc.setText(f"{qc_d:+.4f}°  ({math.radians(qc_d):+.6f} rad)")
            oqp.setText(f"{qp_d:+.4f}°  ({math.radians(qp_d):+.6f} rad)")
            oqr.setText(f"{qr_d:+.4f}°  ({math.radians(qr_d):+.6f} rad)")

            if sol['dentro_limites']:
                oerr.setText("✓ Solución dentro de los límites ±90° del URDF")
                oerr.setStyleSheet("color:#5ec46e;font-weight:bold;")
            else:
                oerr.setText("✗ Fuera de los límites ±90° del URDF")
                oerr.setStyleSheet("color:#e05c5c;font-weight:bold;")

            xr,yr,zr = fk_pos(qc_d, qp_d, qr_d)
            err = sol['err_mm']
            vx.setText(f"{xr:+.6f} m"); vy.setText(f"{yr:+.6f} m"); vz.setText(f"{zr:+.6f} m")
            col2 = "#5ec46e" if err < 5. else "#e05c5c"
            ve.setStyleSheet(f"color:{col2};font-weight:bold;")
            ve.setText(f"{err:.4f} mm")

            # Texto de pasos/ecuaciones
            if method == "algebraic":
                steps_txt.setPlainText(sol['pasos'])
            else:
                # Geométrico: construir texto de pasos
                px_=ix.value(); py_=iy.value(); pz_=iz.value()
                rho=math.hypot(px_,py_)
                s=sol['s']; u=sol['u']; w=sol['w']
                c2=sol['c2']; s2=sol['s2']
                steps = (
                    f"Solución [{idx+1}]: Rodilla = '{sol['rama_rodilla']}'\n"
                    f"{'─'*60}\n\n"
                    f"VISTA SUPERIOR  —  plano XY  (para qc):\n"
                    f"  ρ  = √(x²+y²) = √({px_:.5f}²+{py_:.5f}²) = {rho:.5f} m\n"
                    f"  s  = √(ρ²−d²) = √({rho:.5f}²−{d:.5f}²) = {s:.5f} m\n"
                    f"       [yc = −s = {sol['yc']:+.5f} m]\n"
                    f"  qc = atan2(−s,−d) − atan2(y,x)\n"
                    f"     = atan2({sol['yc']:+.5f},{-d:.5f}) − atan2({py_:.5f},{px_:.5f})\n"
                    f"     = {qc_d:+.4f}°\n\n"
                    f"PLANO SAGITAL  —  problema 2R  (para qp, qr):\n"
                    f"  u = s = {u:.5f} m\n"
                    f"  w = h − z = {h:.5f} − {pz_:.5f} = {w:.5f} m\n\n"
                    f"  Ley de cosenos para c₂:\n"
                    f"    c₂ = (u²+w²−L1²−L2²)/(2·L1·L2)\n"
                    f"       = ({u**2:.5f}+{w**2:.5f}−{L1**2:.5f}−{L2**2:.5f}) / {2*L1*L2:.5f}\n"
                    f"       = {c2:+.5f}\n"
                    f"    s₂ = ±√(1−c₂²) = ±{abs(s2):.5f}\n"
                    f"         [rama rodilla '{sol['rama_rodilla']}': s₂ = {s2:+.5f}]\n\n"
                    f"    qr_eff = atan2(s₂,c₂) = atan2({s2:.5f},{c2:.5f})\n"
                    f"           = {sol['qr_eff_deg']:+.4f}°\n\n"
                    f"    qp = atan2(w,u) − atan2(L2·s₂, L1+L2·c₂)\n"
                    f"       = atan2({w:.5f},{u:.5f}) − atan2({L2*s2:.5f},{L1+L2*c2:.5f})\n"
                    f"       = {qp_d:+.4f}°\n\n"
                    f"    qr = qr_eff + φ = {sol['qr_eff_deg']:+.4f}° + {math.degrees(phi):+.4f}°\n"
                    f"       = {qr_d:+.4f}°\n\n"
                    f"VERIFICACIÓN FK(IK):\n"
                    f"  p_rec = ({xr:.5f}, {yr:.5f}, {zr:.5f}) m\n"
                    f"  ‖error‖ = {err:.4f} mm"
                )
                steps_txt.setPlainText(steps)

            _,T=fk_chain(qc_d,qp_d,qr_d); roll,pitch,yaw=rot_to_rpy(T[:3,:3])
            ik_mat.setPlainText(
                f"Solución [{idx+1}]: Rodilla = '{sol['rama_rodilla']}'\n{'-'*44}\n"
                f"xyz [m]   = ({T[0,3]:+.6f}, {T[1,3]:+.6f}, {T[2,3]:+.6f})\n"
                f"rpy [°]   = ({math.degrees(roll):+.4f}, {math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\n"
                f"T (4×4):\n{fmt4(T)}\n\n"
                f"Verificar: ros2 run tf2_ros tf2_echo base_link efector")

        def _apply():
            idx=sol_list.currentRow()
            if idx < 0 or idx >= len(solutions_store): return
            sol = solutions_store[idx]
            qc_d=clamp(sol['qc_deg'],-90.,90.)
            qp_d=clamp(sol['qp_deg'],-90.,90.)
            qr_d=clamp(sol['qr_deg'],-90.,90.)
            self.node.set_deg("joint_c",qc_d); self.node.set_deg("joint_p",qp_d); self.node.set_deg("joint_r",qr_d)
            for j,v in zip(JOINTS,[qc_d,qp_d,qr_d]):
                self._sl[j].blockSignals(True); self._sl[j].setValue(int(round(v*10))); self._sl[j].blockSignals(False)
                self._ed[j].setText(f"{v:.1f}")
            self._refresh_fk()
            self.tabs.setCurrentIndex(0)

        btn.clicked.connect(_solve)
        sol_list.currentRowChanged.connect(_on_selected)
        btn_apply.clicked.connect(_apply)
        return w

    # ──────────────────────────────────────────────────────────────
    #  TABS IK SIMPLES (MTH, Gradiente, Newton)
    # ──────────────────────────────────────────────────────────────
    def _tab_ik(self, method):
        """Genera una pestaña IK genérica con entrada xyz y panel de resultados."""
        w=QtWidgets.QWidget(); outer=QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10,10,10,10); outer.setSpacing(14)
        left=QtWidgets.QVBoxLayout(); left.setSpacing(10)
        right=QtWidgets.QVBoxLayout(); right.setSpacing(10)

        TITLES={
            "mth":      ("IK por MTH — inversión de matrices DH",
                "W = T_BASE_0⁻¹ · T_obj · T_3_EF⁻¹\n"
                "qc = atan2(−W[0,2],−W[1,2])\n"
                "qp = asin((D1−W[2,3])/L2)\n"
                "qr = atan2(A3[1,0],A3[0,0])\n"
                "Bisección 1D + Gauss-Newton 2D para posición pura."),
            "gradient": ("IK Gradiente — descenso J^T con paso BB1",
                "d = Jᵀ·e   (dirección de descenso)\n"
                "α = BB1 espectral con safeguard\n"
                "q ← clip(q + α·d, −π/2, +π/2)\n"
                "9 semillas, ε = 1e-5 rad."),
            "newton":   ("IK Newton — Newton-Raphson pseudo-inversa",
                "J·Δq = e  (lstsq con clipping de paso)\n"
                "q ← clip(q + Δq, −π/2, +π/2)\n"
                "9 semillas, ε = 1e-5 rad, paso_max = 0.5 rad."),
        }
        title_txt, desc_txt = TITLES[method]
        title=QtWidgets.QLabel(title_txt)
        title.setStyleSheet("color:#4a9eff;font-size:13px;font-weight:bold;")
        title.setWordWrap(True); left.addWidget(title)

        x0,y0,z0_=fk_pos(0,0,0)
        grp_in=QtWidgets.QGroupBox("Posición objetivo del efector  [m]")
        g_in=QtWidgets.QGridLayout(grp_in); g_in.setSpacing(8)
        ix=_spin_xyz(x0); iy=_spin_xyz(y0); iz=_spin_xyz(z0_)
        for i,(sym,ws,col) in enumerate([("x =",ix,"#e05c5c"),("y =",iy,"#5ec46e"),("z =",iz,"#4a9eff")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ws.setStyleSheet(f"color:{col};background:#111;border:1px solid #252525;border-radius:4px;padding:3px 5px;")
            g_in.addWidget(lb,i,0); g_in.addWidget(ws,i,1)
        ref=QtWidgets.QLabel(f"Reposo (0,0,0): ({x0:.5f}, {y0:.5f}, {z0_:.5f}) m")
        ref.setStyleSheet("color:#3a5a3a;font-size:11px;"); g_in.addWidget(ref,3,0,1,2)
        left.addWidget(grp_in)

        COL_BTN={"mth":"#4a9eff","gradient":"#f4a261","newton":"#c77dff"}
        btn=QtWidgets.QPushButton(f"Resolver IK  [{method.upper()}]  →  qc  qp  qr")
        btn.setStyleSheet(f"background:{COL_BTN[method]};color:#000;font-weight:bold;"
                          f"font-size:13px;padding:9px 16px;border-radius:6px;")
        left.addWidget(btn)

        grp_res=QtWidgets.QGroupBox("Resultado  —  ángulos articulares  [±90°]")
        g_res=QtWidgets.QGridLayout(grp_res); g_res.setSpacing(6)
        oqc=QtWidgets.QLabel("-"); oqp=QtWidgets.QLabel("-"); oqr=QtWidgets.QLabel("-")
        oerr=QtWidgets.QLabel(""); oerr.setStyleSheet("color:#e05c5c;font-weight:bold;"); oerr.setWordWrap(True)
        for i,(sym,ww,col) in enumerate([("qc =",oqc,"#e05c5c"),("qp =",oqp,"#4a9eff"),("qr =",oqr,"#5ec46e")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:13px;font-weight:bold;"
                             f"font-family:'Consolas','Courier New',monospace;")
            g_res.addWidget(lb,i,0); g_res.addWidget(ww,i,1)
        g_res.addWidget(oerr,3,0,1,2); left.addWidget(grp_res)

        grp_v=QtWidgets.QGroupBox("Verificación  —  FK( IK(p) )  [m]")
        g_v=QtWidgets.QGridLayout(grp_v); g_v.setSpacing(6)
        vx=QtWidgets.QLabel("-"); vy=QtWidgets.QLabel("-"); vz=QtWidgets.QLabel("-"); ve=QtWidgets.QLabel("-")
        for i,(sym,ww,col) in enumerate([("x_rec =",vx,"#4dd0e1"),("y_rec =",vy,"#4dd0e1"),
                                          ("z_rec =",vz,"#4dd0e1"),("‖err‖ =",ve,"#5ec46e")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:12px;font-family:'Consolas','Courier New',monospace;")
            g_v.addWidget(lb,i,0); g_v.addWidget(ww,i,1)
        left.addWidget(grp_v); left.addStretch(1)

        grp_mat=QtWidgets.QGroupBox("T(base→efector)  tras IK  ←  tf2_echo")
        vm=QtWidgets.QVBoxLayout(grp_mat)
        ik_mat=QtWidgets.QPlainTextEdit(); ik_mat.setReadOnly(True)
        vm.addWidget(ik_mat); right.addWidget(grp_mat,1)
        note=QtWidgets.QLabel(desc_txt)
        note.setStyleSheet("color:#3a3a4a;font-size:11px;font-family:'Consolas','Courier New',monospace;")
        note.setWordWrap(True); right.addWidget(note)
        outer.addLayout(left,46); outer.addLayout(right,54)

        widgets=(ix,iy,iz,oqc,oqp,oqr,oerr,vx,vy,vz,ve,ik_mat)
        btn.clicked.connect(lambda _=False,m=method,W=widgets: self._solve_simple(m,W))
        return w

    # ──────────────────────────────────────────────────────────────
    #  TAB 7 — COMPARACIÓN
    # ──────────────────────────────────────────────────────────────
    def _tab_compare(self):
        w=QtWidgets.QWidget(); vl=QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(10,10,10,10); vl.setSpacing(10)
        title=QtWidgets.QLabel("Comparación simultánea de los 5 métodos IK")
        title.setStyleSheet("color:#4a9eff;font-size:14px;font-weight:bold;")
        vl.addWidget(title)

        grp_in=QtWidgets.QGroupBox("Posición objetivo [m]")
        g_in=QtWidgets.QHBoxLayout(grp_in); g_in.setSpacing(16)
        x0,y0,z0_=fk_pos(0,0,0)
        self._cx=_spin_xyz(x0); self._cy=_spin_xyz(y0); self._cz=_spin_xyz(z0_)
        for sym,ws,col in [("x =",self._cx,"#e05c5c"),("y =",self._cy,"#5ec46e"),("z =",self._cz,"#4a9eff")]:
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet(f"color:{col};font-size:12px;font-weight:bold;")
            ws.setStyleSheet(f"color:{col};background:#111;border:1px solid #252525;border-radius:4px;padding:3px 5px;")
            g_in.addWidget(lb); g_in.addWidget(ws)
        vl.addWidget(grp_in)

        btn=QtWidgets.QPushButton("▶  Comparar todos los métodos")
        btn.setStyleSheet("background:#4a9eff;color:#000;font-weight:bold;"
                          "font-size:13px;padding:9px 24px;border-radius:6px;")
        btn.clicked.connect(self._compare_all); vl.addWidget(btn)

        self._tbl=QtWidgets.QTableWidget(0,8)
        self._tbl.setHorizontalHeaderLabels(
            ["Método","qc (°)","qp (°)","qr (°)","‖err‖ (mm)","t (ms)","Estado","Acción"])
        self._tbl.horizontalHeader().setStretchLastSection(False)
        self._tbl.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.ResizeToContents)
        for c in range(1,7): self._tbl.horizontalHeader().setSectionResizeMode(c,QtWidgets.QHeaderView.Stretch)
        self._tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._tbl.verticalHeader().setVisible(False)
        vl.addWidget(self._tbl,1)

        self._cmp_detail=QtWidgets.QPlainTextEdit()
        self._cmp_detail.setReadOnly(True); self._cmp_detail.setMaximumHeight(180)
        vl.addWidget(self._cmp_detail)
        self._tbl.cellClicked.connect(self._compare_row_clicked)
        return w

    # ──────────────────────────────────────────────────────────────
    #  LÓGICA FK
    # ──────────────────────────────────────────────────────────────
    def _refresh_fk(self):
        qc=self.node.get_deg("joint_c"); qp=self.node.get_deg("joint_p"); qr=self.node.get_deg("joint_r")
        self._fk_frames,T=fk_chain(qc,qp,qr); p=T[:3,3]
        self._fk_lx.setText(f"{p[0]:+.6f} m"); self._fk_ly.setText(f"{p[1]:+.6f} m"); self._fk_lz.setText(f"{p[2]:+.6f} m")
        cur=max(self._fk_lst.currentRow(),0)
        self._fk_lst.blockSignals(True); self._fk_lst.clear()
        for name,_ in self._fk_frames: self._fk_lst.addItem(name)
        self._fk_lst.blockSignals(False)
        self._fk_lst.setCurrentRow(min(cur,len(self._fk_frames)-1))
        self._show_fk_mat()

    def _show_fk_mat(self):
        idx=self._fk_lst.currentRow()
        if idx<0 or not hasattr(self,"_fk_frames") or idx>=len(self._fk_frames): return
        name,T=self._fk_frames[idx]; p=T[:3,3]; roll,pitch,yaw=rot_to_rpy(T[:3,:3])
        self._fk_txt.setPlainText(
            f"{name}\n{'-'*52}\n"
            f"xyz  [m]   = ({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f})\n"
            f"rpy  [rad] = ({roll:+.6f}, {pitch:+.6f}, {yaw:+.6f})\n"
            f"rpy  [°]   = ({math.degrees(roll):+.4f}, {math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\n"
            f"T (4×4):\n{fmt4(T)}")

    # ──────────────────────────────────────────────────────────────
    #  LÓGICA IK SIMPLE (MTH, Gradiente, Newton)
    # ──────────────────────────────────────────────────────────────
    def _solve_simple(self, method, widgets):
        ix,iy,iz,oqc,oqp,oqr,oerr,vx,vy,vz,ve,ik_mat=widgets
        px=ix.value(); py=iy.value(); pz=iz.value()
        try:
            if method=="mth":
                qc_d,qp_d,qr_d=ik_mth(px,py,pz)
            elif method=="gradient":
                qc_d,qp_d,qr_d=ik_gradient(px,py,pz)
            elif method=="newton":
                qc_d,qp_d,qr_d=ik_newton(px,py,pz)
        except ValueError as e:
            for ww in(oqc,oqp,oqr): ww.setText("-")
            oerr.setText(f"Sin solución:\n{e}")
            ik_mat.setPlainText(f"Sin solución:\n{e}")
            for ww in(vx,vy,vz,ve): ww.setText("-")
            return

        oerr.setText("")
        oqc.setText(f"{qc_d:+.4f}°  ({math.radians(qc_d):+.6f} rad)")
        oqp.setText(f"{qp_d:+.4f}°  ({math.radians(qp_d):+.6f} rad)")
        oqr.setText(f"{qr_d:+.4f}°  ({math.radians(qr_d):+.6f} rad)")

        xr,yr,zr=fk_pos(qc_d,qp_d,qr_d)
        err=math.sqrt((px-xr)**2+(py-yr)**2+(pz-zr)**2)
        vx.setText(f"{xr:+.6f} m"); vy.setText(f"{yr:+.6f} m"); vz.setText(f"{zr:+.6f} m")
        col="#5ec46e" if err<5e-3 else "#e05c5c"
        ve.setStyleSheet(f"color:{col};font-weight:bold;")
        ve.setText(f"{err*1e3:.3f} mm  ({'OK' if err<5e-3 else 'WARN'})")

        _,T=fk_chain(qc_d,qp_d,qr_d); roll,pitch,yaw=rot_to_rpy(T[:3,:3])
        ik_mat.setPlainText(
            f"T(base→efector)  [{method.upper()}]:\n{'-'*44}\n"
            f"xyz [m]   = ({T[0,3]:+.6f}, {T[1,3]:+.6f}, {T[2,3]:+.6f})\n"
            f"rpy [°]   = ({math.degrees(roll):+.4f}, {math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\n"
            f"T (4×4):\n{fmt4(T)}\n\n"
            f"Verificar: ros2 run tf2_ros tf2_echo base_link efector")

        self.node.set_deg("joint_c",qc_d); self.node.set_deg("joint_p",qp_d); self.node.set_deg("joint_r",qr_d)
        for j,v in zip(JOINTS,[qc_d,qp_d,qr_d]):
            dv=clamp(v,-90.,90.)
            self._sl[j].blockSignals(True); self._sl[j].setValue(int(round(dv*10))); self._sl[j].blockSignals(False)
            self._ed[j].setText(f"{dv:.1f}")
        self._refresh_fk()

    # ──────────────────────────────────────────────────────────────
    #  LÓGICA COMPARACIÓN
    # ──────────────────────────────────────────────────────────────
    def _compare_all(self):
        px=self._cx.value(); py=self._cy.value(); pz=self._cz.value()
        METHODS=[
            ("Algebraico",  lambda: ik_algebraic_best(px,py,pz)),
            ("MTH",         lambda: ik_mth(px,py,pz)),
            ("Geométrico",  lambda: ik_geometric_best(px,py,pz)),
            ("Gradiente",   lambda: ik_gradient(px,py,pz)),
            ("Newton",      lambda: ik_newton(px,py,pz)),
        ]
        COLORS=["#e05c5c","#4a9eff","#5ec46e","#f4a261","#c77dff"]

        self._tbl.setRowCount(0)
        self._cmp_results=[]

        for row,(name,fn) in enumerate(METHODS):
            self._tbl.insertRow(row)
            t0=time.perf_counter()
            try:
                qc_d,qp_d,qr_d=fn()
                elapsed=(time.perf_counter()-t0)*1e3
                xr,yr,zr=fk_pos(qc_d,qp_d,qr_d)
                err=math.sqrt((px-xr)**2+(py-yr)**2+(pz-zr)**2)*1e3
                status="OK" if err<5. else "WARN"
                row_data=(qc_d,qp_d,qr_d,err,elapsed,status,True)
            except ValueError as e:
                elapsed=(time.perf_counter()-t0)*1e3
                row_data=(None,None,None,None,elapsed,str(e)[:60],False)

            col=COLORS[row]
            lbl=QtWidgets.QTableWidgetItem(name)
            lbl.setForeground(QtWidgets.QApplication.palette().text())
            self._tbl.setItem(row,0,lbl)

            if row_data[6]:
                qc_d,qp_d,qr_d,err,elapsed,status,_=row_data
                for c,txt in enumerate([f"{qc_d:+.2f}",f"{qp_d:+.2f}",f"{qr_d:+.2f}",
                                         f"{err:.3f}",f"{elapsed:.1f}",status]):
                    item=QtWidgets.QTableWidgetItem(txt)
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self._tbl.setItem(row,c+1,item)
                btn=QtWidgets.QPushButton("Aplicar")
                btn.setStyleSheet(f"background:{col};color:#000;font-size:11px;"
                                  f"border-radius:3px;padding:3px 8px;")
                btn.clicked.connect(lambda _=False,qc=qc_d,qp=qp_d,qr=qr_d: self._apply_solution(qc,qp,qr))
                self._tbl.setCellWidget(row,7,btn)
            else:
                _,_,_,_,elapsed,msg,_=row_data
                item=QtWidgets.QTableWidgetItem(f"—  {elapsed:.1f} ms  Sin solución")
                item.setForeground(QtWidgets.QApplication.palette().text())
                self._tbl.setSpan(row,1,1,7)
                self._tbl.setItem(row,1,item)

            self._cmp_results.append(row_data)

        self._tbl.resizeRowsToContents()

    def _compare_row_clicked(self, row, _col):
        if row>=len(self._cmp_results): return
        data=self._cmp_results[row]
        if not data[6]:
            self._cmp_detail.setPlainText(f"Sin solución:\n{data[5]}"); return
        qc_d,qp_d,qr_d,err,elapsed,status,_=data
        _,T=fk_chain(qc_d,qp_d,qr_d); roll,pitch,yaw=rot_to_rpy(T[:3,:3])
        self._cmp_detail.setPlainText(
            f"qc={qc_d:+.4f}°  qp={qp_d:+.4f}°  qr={qr_d:+.4f}°\n"
            f"Error posición: {err:.4f} mm    Tiempo: {elapsed:.2f} ms\n"
            f"T(base→efector):\n{fmt4(T)}")

    def _apply_solution(self, qc_d, qp_d, qr_d):
        self.node.set_deg("joint_c",qc_d); self.node.set_deg("joint_p",qp_d); self.node.set_deg("joint_r",qr_d)
        for j,v in zip(JOINTS,[qc_d,qp_d,qr_d]):
            dv=clamp(v,-90.,90.)
            self._sl[j].blockSignals(True); self._sl[j].setValue(int(round(dv*10))); self._sl[j].blockSignals(False)
            self._ed[j].setText(f"{dv:.1f}")
        self._refresh_fk()
        self.tabs.setCurrentIndex(0)

    # ──────────────────────────────────────────────────────────────
    #  BOTONES FK
    # ──────────────────────────────────────────────────────────────
    def _do_zero(self):
        for j in JOINTS:
            self.node.set_deg(j,0.); self._sl[j].blockSignals(True)
            self._sl[j].setValue(0); self._sl[j].blockSignals(False); self._ed[j].setText("0.0")
        self._refresh_fk()

    def _do_verify(self):
        x,y,z=fk_pos(0,0,0); _,T=fk_chain(0,0,0); roll,pitch,yaw=rot_to_rpy(T[:3,:3])
        QtWidgets.QMessageBox.information(self,"Verificación q=(0°,0°,0°) vs tf2_echo",
            f"FK  q=(0°,0°,0°):\n  x={x:+.6f} m\n  y={y:+.6f} m\n  z={z:+.6f} m\n\n"
            f"tf2_echo base_link efector:\n  Translation: [−0.130, −0.641, +0.226] m  ✓\n\n"
            f"RPY [rad]: ({roll:+.5f}, {pitch:+.5f}, {yaw:+.5f})\n\nT (4×4):\n{fmt4(T)}")


# =================================================================
#  MAIN
# =================================================================
def main(args=None):
    rclpy.init(args=args)
    node=PataNode()
    app=QtWidgets.QApplication([])
    win=UnifiedWindow(node); win.show()
    spin_t=QtCore.QTimer(); spin_t.timeout.connect(lambda: rclpy.spin_once(node,timeout_sec=0.0)); spin_t.start(10)
    signal.signal(signal.SIGINT,  lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    app.exec_()
    try: node.destroy_node()
    except: pass
    try: rclpy.shutdown()
    except: pass

if __name__=="__main__":
    main()