#!/usr/bin/env python3
"""
gui_newton.py — Cinemática Inversa Numérica (Método de Newton) para LA_PATA_SOLA
================================================================================

Este script implementa una GUI en PyQt5 que resuelve la cinemática inversa de
la pata del robot LA_PATA_SOLA utilizando el método de Newton con
estabilidad numérica. Además de la pestaña principal para la resolución
numérica, se incluye una pestaña de cinemática directa para ajustar
manualmente los ángulos y comprobar la posición resultante del efector.

La cinemática directa se calcula a partir del URDF del robot y los
parámetros geométricos extraídos del mismo.  El Jacobiano se estima por
diferencias finitas y el método de Newton aplica una pseudo-inversa con
limitación de paso para garantizar la estabilidad.

Uso:
    ros2 run parcial2 gui_newton
"""

import math
import signal
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5 import QtWidgets, QtCore

# ─────────────────── Parámetros del robot ───────────────────────
# Mismos valores que en gui_algebraic.py.  Todas las unidades están
# en metros o radianes.
z0    = 0.1439
dz_p  = 0.075
a     = 0.0403
L1    = 0.3505
L2    = np.hypot(0.2905, -0.0010066)
k1    = -0.00365
k2    = 0.093563
phi   = 0.026978 + math.atan2(-0.0010066, 0.2905)
h     = z0 + dz_p
d     = a + k1 + k2

JOINTS = ["joint_c", "joint_p", "joint_r"]
LIMITS_DEG = {
    "joint_c": (-90.0, 90.0),
    "joint_p": (-90.0, 90.0),
    "joint_r": (-90.0, 90.0),
}
HOME_DEG = {"joint_c": 0.0, "joint_p": 0.0, "joint_r": 0.0}
PUBLISH_HZ = 30.0

# Parametrización del método de Newton
EPSILON_DEFAULT   = 1e-4    # tolerancia del error cartesiano [m]
MAX_ITER_DEFAULT  = 200     # iteraciones máximas
H_DIFF            = 1e-6    # paso para diferencias finitas [rad]
MAX_STEP_DEFAULT  = 0.30    # paso máximo por iteración [rad]
COND_WARN_DEFAULT = 100.0   # umbral de número de condición para avisar singularidad
Q_MIN             = -math.pi/2  # límite inferior articular [rad]
Q_MAX             =  math.pi/2  # límite superior articular [rad]


# ─────────────────── Cinemática directa e inversa ───────────────

def fk_geom(qc: float, qp: float, qr: float) -> np.ndarray:
    """Cinemática directa geométrica.  Devuelve array [x, y, z]."""
    s_val = L1 * math.cos(qp) + L2 * math.cos(qp + qr - phi)
    z_val = h - L1 * math.sin(qp) - L2 * math.sin(qp + qr - phi)
    x_val = -d * math.cos(qc) - s_val * math.sin(qc)
    y_val =  d * math.sin(qc) - s_val * math.cos(qc)
    return np.array([x_val, y_val, z_val], dtype=float)


def rpy_to_mat(r: float, p: float, y_: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y_), math.sin(y_)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=float)
    return R


def make_transform(xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_mat(*rpy)
    T[:3, 3] = xyz
    return T


def fk_chain(qc: float, qp: float, qr: float) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    """Cinemática directa completa.  Devuelve lista de matrices y T final."""
    T01 = make_transform((0.0, 0.0, z0), (0.0, 0.0, 0.0)) @ make_transform((0.0, 0.0, 0.0), (0.0, 0.0, -qc))
    T12 = make_transform((-a, 0.0, dz_p), (math.pi/2, 0.0, -math.pi/2)) @ make_transform((0.0, 0.0, 0.0), (0.0, 0.0, -qp))
    T23 = make_transform((0.3505, -2.9381e-05, k1), (0.0, 0.0, 0.026978)) @ make_transform((0.0, 0.0, 0.0), (0.0, 0.0, -qr))
    T34 = make_transform((0.2905, -0.0010066, k2), (-0.18364, 1.5696, 1.3872))
    T1 = T01
    T2 = T01 @ T12
    T3 = T2  @ T23
    T4 = T3  @ T34
    frames = [
        ("T0→link_c",  T1),
        ("T0→link_p",  T2),
        ("T0→link_r",  T3),
        ("T0→efector", T4),
    ]
    return frames, T4


def jacobian_num(q: np.ndarray, h: float = H_DIFF) -> np.ndarray:
    """Jacobian numérico 3×3 por diferencias finitas centrales."""
    J = np.zeros((3, 3), dtype=float)
    for j in range(3):
        dq = np.zeros(3)
        dq[j] = h
        fp = fk_geom(*(q + dq))
        fm = fk_geom(*(q - dq))
        J[:, j] = (fp - fm) / (2.0 * h)
    return J


def newton_ik(
    xd: np.ndarray,
    q0: np.ndarray,
    epsilon: float,
    max_iter: int,
    max_step: float,
    cond_warn: float,
) -> Tuple[np.ndarray, int, float, bool, List[dict], List[int]]:
    """Resuelve q para fk_geom(q) ≈ xd mediante método de Newton.

    Devuelve: (q_sol, iters, err_final, convergio, log, sing_iters)
    """
    q = q0.copy()
    log = []
    sing_iters = []
    for k in range(max_iter):
        f = fk_geom(*q)
        e = xd - f
        err_norm = float(np.linalg.norm(e))
        J = jacobian_num(q)
        cond = float(np.linalg.cond(J))
        det  = float(np.linalg.det(J))
        singular_flag = cond > cond_warn
        log.append({
            "k":        k,
            "q_deg":    np.degrees(q).copy(),
            "err":      err_norm,
            "cond":     cond,
            "det":      det,
            "singular": singular_flag,
        })
        if singular_flag:
            sing_iters.append(k)
        if err_norm < epsilon:
            return q, k, err_norm, True, log, sing_iters
        dq, *_ = np.linalg.lstsq(J, e, rcond=None)
        norm_dq = float(np.linalg.norm(dq))
        if norm_dq > max_step:
            dq = dq * (max_step / norm_dq)
        q = np.clip(q + dq, Q_MIN, Q_MAX)
    # no convergió
    f = fk_geom(*q)
    e = xd - f
    return q, max_iter, float(np.linalg.norm(e)), False, log, sing_iters


# ─────────────────── Nodo ROS 2 ─────────────────────────────────
class NewtonIKNode(Node):
    def __init__(self) -> None:
        super().__init__("leg_kinematics_newton_gui")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self._rad = {j: 0.0 for j in JOINTS}
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)
    def set_deg(self, j: str, deg: float) -> None:
        lo, hi = LIMITS_DEG[j]
        self._rad[j] = math.radians(max(lo, min(hi, deg)))
    def set_rad(self, j: str, r: float) -> None:
        lo, hi = LIMITS_DEG[j]
        self._rad[j] = max(math.radians(lo), min(math.radians(hi), r))
    def get_deg(self, j: str) -> float:
        return math.degrees(self._rad[j])
    def get_rad(self, j: str) -> float:
        return self._rad[j]
    def _publish(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS[:]
        msg.position = [self._rad[j] for j in JOINTS]
        self.pub.publish(msg)


# ─────────────────── Estilo ─────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background: #000000;
    color: #d0d0d0;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #2a2a2a;
    border-radius: 7px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-size: 13px;
    font-weight: bold;
    color: #4a9eff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QLabel { color: #d0d0d0; }
QSlider::groove:horizontal {
    height: 5px; background: #111111; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #4a9eff; border: none;
    width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #4a9eff; border-radius: 3px; }
QLineEdit, QDoubleSpinBox, QSpinBox {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    color: #d0d0d0;
    padding: 3px 5px;
    min-width: 64px;
    font-size: 14px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: #4a9eff; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button, QSpinBox::up-button, QSpinBox::down-button {
    width: 16px; background: #1a1a1a; border-radius: 2px;
}
QPushButton {
    background: #1a1a1a; border: none; border-radius: 5px;
    color: #d0d0d0; padding: 7px 16px; font-size: 14px;
}
QPushButton:hover   { background: #222222; }
QPushButton:pressed { background: #4a9eff; color: #000000; }
QListWidget {
    background: #080808; border: 1px solid #2a2a2a;
    border-radius: 4px; color: #5ec46e; font-size: 13px;
}
QListWidget::item { padding: 4px 8px; }
QListWidget::item:selected { background: #111111; color: #4a9eff; }
QPlainTextEdit {
    background: #080808; border: 1px solid #2a2a2a;
    border-radius: 4px; color: #5ec46e;
    font-family: "Consolas", "Courier New", monospace; font-size: 13px;
}
QTabWidget::pane { border: 1px solid #2a2a2a; border-radius: 6px; margin-top: -1px; }
QTabBar::tab {
    background: #111111; color: #d0d0d0;
    padding: 8px 20px; border-radius: 5px 5px 0 0;
    margin-right: 3px; font-size: 13px;
}
QTabBar::tab:selected { background: #1a1a1a; color: #4a9eff; }
QTabBar::tab:hover    { background: #181818; }
QScrollBar:vertical { background: #000000; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #1a1a1a; border-radius: 4px; min-height: 20px; }
"""


def _colored_slider(color: str) -> QtWidgets.QSlider:
    sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    sl.setStyleSheet(f"""
        QSlider::groove:horizontal {{
            height: 5px; background: #111111; border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {color}; border: none;
            width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
        }}
        QSlider::sub-page:horizontal {{ background: {color}; border-radius: 3px; }}
    """)
    return sl


_JOINT_META = {
    "joint_c": {
        "sym": "α",
        "name": "Cadera",
        "axis": "eje Z (negativo)",
        "color": "#e05c5c",
        "sl_color": "#e05c5c",
    },
    "joint_p": {
        "sym": "β",
        "name": "Fémur",
        "axis": "eje Z (negativo)",
        "color": "#4a9eff",
        "sl_color": "#4a9eff",
    },
    "joint_r": {
        "sym": "λ",
        "name": "Rodilla",
        "axis": "eje Z (negativo)",
        "color": "#5ec46e",
        "sl_color": "#5ec46e",
    },
}

# Colores de aviso de singularidad
_SING_BG   = "#3a1a00"
_SING_FG   = "#ff9900"
_NOSING_BG = "#080808"
_NOSING_FG = "#5ec46e"


# ─────────────────── Ventana principal ─────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, node: NewtonIKNode) -> None:
        super().__init__()
        self.node = node
        self.setWindowTitle("Cinemática Inversa — Método de Newton (LA_PATA_SOLA)")
        self.setMinimumSize(1100, 740)
        self.setStyleSheet(STYLE)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_newton_tab(), "  Método de Newton (IK)  ")
        tabs.addTab(self._build_fk_tab(),     "  Cinemática Directa (FK)  ")
        self.setCentralWidget(tabs)
        self._do_home()

    # ── Helpers ─────────────────────────────────────────────────
    def _spin_xyz(self, val: float) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(-1.0, 1.0)
        sb.setDecimals(5)
        sb.setSingleStep(0.005)
        sb.setValue(val)
        return sb
    def _make_deg_spin(self, val: float) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(-180.0, 180.0)
        sb.setDecimals(2)
        sb.setSingleStep(5.0)
        sb.setValue(val)
        return sb

    # ── Pestaña de Newton ──────────────────────────────────────
    def _build_newton_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(14)
        left = QtWidgets.QVBoxLayout(); left.setSpacing(8)
        # Posición deseada
        grp_xd = QtWidgets.QGroupBox("Posición objetivo del pie  [m]")
        grid_xd = QtWidgets.QGridLayout(grp_xd); grid_xd.setSpacing(6)
        # Valores iniciales: posición en (0,0,0) articulaciones home
        x0, y0, z0_p = fk_geom(0.0, 0.0, 0.0)
        self._xd = self._spin_xyz(x0)
        self._yd = self._spin_xyz(y0)
        self._zd = self._spin_xyz(z0_p)
        for i, (sym, sp, color) in enumerate([
            ("x =", self._xd, "#e05c5c"),
            ("y =", self._yd, "#5ec46e"),
            ("z =", self._zd, "#4a9eff"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:14px;")
            sp.setStyleSheet(
                f"color:{color}; background:#111111; border:1px solid #2a2a2a; border-radius:4px; padding:3px 5px;"
            )
            grid_xd.addWidget(lbl, i, 0)
            grid_xd.addWidget(sp,  i, 1)
        left.addWidget(grp_xd)
        # Valor inicial q0
        grp_q0 = QtWidgets.QGroupBox("Valor inicial  q₀  [°]")
        grid_q0 = QtWidgets.QGridLayout(grp_q0); grid_q0.setSpacing(6)
        self._q0a = self._make_deg_spin(0.0)
        self._q0b = self._make_deg_spin(0.0)
        self._q0l = self._make_deg_spin(0.0)
        for i, (sym, sp, color) in enumerate([
            ("α₀:",  self._q0a, "#e05c5c"),
            ("β₀:",   self._q0b, "#4a9eff"),
            ("λ₀:", self._q0l, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet(f"color:{color}; font-size:14px;")
            grid_q0.addWidget(lbl, i, 0)
            grid_q0.addWidget(sp,  i, 1)
        left.addWidget(grp_q0)
        # Parámetros de Newton
        grp_p = QtWidgets.QGroupBox("Parámetros del Método de Newton")
        form_p = QtWidgets.QFormLayout(grp_p); form_p.setSpacing(6)
        self._spin_eps = QtWidgets.QDoubleSpinBox(); self._spin_eps.setRange(1e-10, 1e-1); self._spin_eps.setDecimals(8); self._spin_eps.setSingleStep(1e-5); self._spin_eps.setValue(EPSILON_DEFAULT)
        self._spin_maxiter = QtWidgets.QSpinBox(); self._spin_maxiter.setRange(1, 5000); self._spin_maxiter.setValue(MAX_ITER_DEFAULT)
        self._spin_maxstep = QtWidgets.QDoubleSpinBox(); self._spin_maxstep.setRange(0.01, math.pi); self._spin_maxstep.setDecimals(3); self._spin_maxstep.setSingleStep(0.05); self._spin_maxstep.setValue(MAX_STEP_DEFAULT)
        self._spin_cond = QtWidgets.QDoubleSpinBox(); self._spin_cond.setRange(10.0, 1e8); self._spin_cond.setDecimals(1); self._spin_cond.setSingleStep(50.0); self._spin_cond.setValue(COND_WARN_DEFAULT)
        form_p.addRow("Tolerancia  ε [m]:", self._spin_eps)
        form_p.addRow("Iteraciones máx:",    self._spin_maxiter)
        form_p.addRow("Paso máx [rad]:",     self._spin_maxstep)
        form_p.addRow("Umbral cond(J):",     self._spin_cond)
        left.addWidget(grp_p)
        # Botón de resolución
        btn_solve = QtWidgets.QPushButton("▶   Resolver  Cinemática Inversa  (Newton)")
        btn_solve.setStyleSheet(
            "background:#5ec46e; color:#000000; font-weight:bold; font-size:13px; padding:8px 16px; border-radius:6px;"
        )
        btn_solve.clicked.connect(self._solve_newton)
        left.addWidget(btn_solve)
        # Aviso de singularidad
        self._lbl_sing = QtWidgets.QLabel("")
        self._lbl_sing.setWordWrap(True)
        self._lbl_sing.setStyleSheet(
            f"color:{_SING_FG}; background:{_SING_BG}; border:1px solid #ff9900; border-radius:5px; padding:6px; font-size:13px; font-weight:bold;"
        )
        self._lbl_sing.setVisible(False)
        left.addWidget(self._lbl_sing)
        # Resultado angular
        grp_out = QtWidgets.QGroupBox("Resultado  —  ángulos calculados")
        grid_out = QtWidgets.QGridLayout(grp_out); grid_out.setSpacing(5)
        self._oa    = QtWidgets.QLabel("—")
        self._ob    = QtWidgets.QLabel("—")
        self._ol    = QtWidgets.QLabel("—")
        self._oit   = QtWidgets.QLabel("—")
        self._oer   = QtWidgets.QLabel("—")
        self._ocond = QtWidgets.QLabel("—")
        self._ostat = QtWidgets.QLabel("")
        self._ostat.setStyleSheet("font-size:13px; font-weight:bold;")
        for i, (sym, val_w, color) in enumerate([
            ("α =", self._oa, "#e05c5c"),
            ("β =", self._ob, "#4a9eff"),
            ("λ =", self._ol, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:14px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:15px; font-weight:bold; font-family:'Consolas','Courier New',monospace;"
            )
            grid_out.addWidget(lbl,   i, 0)
            grid_out.addWidget(val_w, i, 1)
        for i, (sym, val_w) in enumerate([
            ("Iteraciones:",  self._oit),
            ("Error final:",  self._oer),
            ("cond(J) final:", self._ocond),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:13px;")
            val_w.setStyleSheet(
                "color:#d0d0d0; font-size:13px; font-family:'Consolas','Courier New',monospace;"
            )
            grid_out.addWidget(lbl,   3 + i, 0)
            grid_out.addWidget(val_w, 3 + i, 1)
        grid_out.addWidget(self._ostat, 6, 0, 1, 2)
        left.addWidget(grp_out)
        # Verificación FK
        grp_v = QtWidgets.QGroupBox("Verificación  —  FK(IK(p))  →  posición reconstruida")
        grid_v = QtWidgets.QGridLayout(grp_v); grid_v.setSpacing(6)
        self._vx = QtWidgets.QLabel("—"); self._vy = QtWidgets.QLabel("—"); self._vz = QtWidgets.QLabel("—"); self._ve = QtWidgets.QLabel("—")
        self._ve.setStyleSheet("color:#5ec46e; font-weight:bold;")
        for i, (sym, val_w, color) in enumerate([
            ("x_rec =",   self._vx, "#4dd0e1"),
            ("y_rec =",   self._vy, "#4dd0e1"),
            ("z_rec =",   self._vz, "#4dd0e1"),
            ("‖error‖ =", self._ve, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:14px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:13px; font-family:'Consolas','Courier New',monospace;"
            )
            grid_v.addWidget(lbl,   i, 0)
            grid_v.addWidget(val_w, i, 1)
        left.addWidget(grp_v)
        left.addStretch(1)
        # Panel derecho: matriz y log
        right = QtWidgets.QVBoxLayout(); right.setSpacing(10)
        grp_mat = QtWidgets.QGroupBox("T0→efector  tras IK  (URDF)")
        vm = QtWidgets.QVBoxLayout(grp_mat)
        self._ik_mat = QtWidgets.QPlainTextEdit(); self._ik_mat.setReadOnly(True); self._ik_mat.setMinimumHeight(160)
        vm.addWidget(self._ik_mat)
        right.addWidget(grp_mat)
        grp_log = QtWidgets.QGroupBox("Log de iteraciones")
        vl = QtWidgets.QVBoxLayout(grp_log)
        self._log_txt = QtWidgets.QPlainTextEdit(); self._log_txt.setReadOnly(True); self._log_txt.setMinimumHeight(120)
        vl.addWidget(self._log_txt)
        right.addWidget(grp_log)
        note = QtWidgets.QLabel(
            "Método de Newton con estabilidad numérica:\n\n"
            "  q_{k+1} = q_k + dq,   dq = J⁺·(xd − f(q_k))\n\n"
            "  • J calculado por diferencias finitas centrales\n"
            "  • ||dq|| limitado a 'Paso máx' por iteración\n"
            "  • q clampeado a [-π/2, π/2] en cada iteración (URDF)\n"
            "  • Aviso si cond(J) > umbral (singularidad)"
        )
        note.setStyleSheet(
            "color:#d0d0d0; background:#000000; font-size:13px; font-family:'Consolas','Courier New',monospace; padding:6px;"
        )
        note.setWordWrap(True)
        right.addWidget(note)
        right.addStretch(1)
        outer.addLayout(left, 48)
        outer.addLayout(right, 52)
        return w

    # ── Pestaña FK ─────────────────────────────────────────────
    def _build_fk_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(14)
        left = QtWidgets.QVBoxLayout(); left.setSpacing(10)
        # Ángulos articulares
        grp_j = QtWidgets.QGroupBox("Ángulos articulares")
        vj = QtWidgets.QVBoxLayout(grp_j); vj.setSpacing(6)
        self._sl = {}; self._ed = {}
        for j in JOINTS:
            meta = _JOINT_META[j]
            top_row = QtWidgets.QHBoxLayout(); top_row.setSpacing(6)
            sym_lbl = QtWidgets.QLabel(meta["sym"])
            sym_lbl.setStyleSheet(
                f"color:{meta['color']}; font-size:22px; font-weight:bold; min-width:22px;"
            )
            sym_lbl.setAlignment(QtCore.Qt.AlignCenter)
            name_lbl = QtWidgets.QLabel(
                f"{meta['name']}   <span style='color:#666666;font-size:12px;'>{meta['axis']}</span>"
            )
            name_lbl.setTextFormat(QtCore.Qt.RichText)
            name_lbl.setStyleSheet(f"color:{meta['color']}; font-size:14px;")
            ed = QtWidgets.QLineEdit("0.0"); ed.setFixedWidth(70); ed.setAlignment(QtCore.Qt.AlignRight)
            deg_lbl = QtWidgets.QLabel("°"); deg_lbl.setStyleSheet("color:#555555; font-size:14px;")
            top_row.addWidget(sym_lbl); top_row.addWidget(name_lbl, 1); top_row.addWidget(ed); top_row.addWidget(deg_lbl)
            bot_row = QtWidgets.QHBoxLayout(); bot_row.setSpacing(4)
            lo, hi = LIMITS_DEG[j]
            lo_lbl = QtWidgets.QLabel(f"{int(lo)}°"); lo_lbl.setStyleSheet("color:#444444; font-size:12px; min-width:28px;"); lo_lbl.setAlignment(QtCore.Qt.AlignRight)
            sl = _colored_slider(meta["sl_color"]); sl.setMinimum(int(lo * 10)); sl.setMaximum(int(hi * 10)); sl.setValue(int(HOME_DEG[j] * 10))
            hi_lbl = QtWidgets.QLabel(f"{int(hi)}°"); hi_lbl.setStyleSheet("color:#444444; font-size:12px; min-width:28px;")
            bot_row.addWidget(lo_lbl); bot_row.addWidget(sl, 1); bot_row.addWidget(hi_lbl)
            def _sl_cb(val, joint=j, edit=ed):
                edit.setText(f"{val / 10.0:.1f}"); self.node.set_deg(joint, val / 10.0); self._refresh_fk()
            def _ed_cb(joint=j, s=sl):
                try:
                    v = float(self._ed[joint].text())
                except ValueError:
                    return
                lo2, hi2 = LIMITS_DEG[joint]
                v = max(lo2, min(hi2, v))
                s.blockSignals(True); s.setValue(int(round(v * 10))); s.blockSignals(False)
                self._ed[joint].setText(f"{v:.1f}")
                self.node.set_deg(joint, v); self._refresh_fk()
            sl.valueChanged.connect(_sl_cb)
            ed.returnPressed.connect(_ed_cb); ed.editingFinished.connect(_ed_cb)
            block = QtWidgets.QVBoxLayout(); block.setSpacing(2); block.addLayout(top_row); block.addLayout(bot_row); vj.addLayout(block)
            if j != JOINTS[-1]:
                sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.HLine); sep.setStyleSheet("color:#1e1e1e;"); vj.addWidget(sep)
            self._sl[j] = sl; self._ed[j] = ed
        left.addWidget(grp_j)
        # Posición del pie
        grp_pos = QtWidgets.QGroupBox("Posición del pie  [m]  —  FK geométrica")
        grid_pos = QtWidgets.QGridLayout(grp_pos); grid_pos.setSpacing(6); grid_pos.setColumnStretch(1, 1)
        self._fk_lx = QtWidgets.QLabel("—"); self._fk_ly = QtWidgets.QLabel("—"); self._fk_lz = QtWidgets.QLabel("—")
        for i, (lbl_txt, val_w, color) in enumerate([
            ("x =", self._fk_lx, "#e05c5c"), ("y =", self._fk_ly, "#5ec46e"), ("z =", self._fk_lz, "#4a9eff"),
        ]):
            lbl = QtWidgets.QLabel(lbl_txt); lbl.setStyleSheet("color:#555555; font-size:14px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:16px; font-weight:bold; font-family:'Consolas','Courier New',monospace;"
            )
            grid_pos.addWidget(lbl,   i, 0); grid_pos.addWidget(val_w, i, 1)
        left.addWidget(grp_pos)
        # Botones
        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(8)
        b0 = QtWidgets.QPushButton("⟳  Zero"); bh = QtWidgets.QPushButton("⌂  Home")
        for btn in (b0, bh):
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(b0); btn_row.addWidget(bh); b0.clicked.connect(self._do_zero); bh.clicked.connect(self._do_home)
        left.addLayout(btn_row); left.addStretch(1)
        # Panel derecho: matrices
        right = QtWidgets.QVBoxLayout(); right.setSpacing(10)
        grp_mat = QtWidgets.QGroupBox("Cadena de matrices (URDF)")
        vm = QtWidgets.QVBoxLayout(grp_mat); vm.setSpacing(6)
        self._lst = QtWidgets.QListWidget(); self._lst.setFixedHeight(80); self._lst.setStyleSheet(
            "QListWidget{font-size:12px;} QListWidget::item{padding:4px 8px;}"
        ); self._txt = QtWidgets.QPlainTextEdit(); self._txt.setReadOnly(True); self._txt.setMinimumHeight(200)
        self._lst.currentRowChanged.connect(self._show_mat)
        vm.addWidget(self._lst); vm.addWidget(self._txt); right.addWidget(grp_mat, 1)
        note = QtWidgets.QLabel(
            "  T0→efector coincide con tf2_ros echo base_link efector\n"
            "  Los valores se calculan a partir del URDF."
        )
        note.setStyleSheet("color:#444444; font-size:11px; font-family:'Consolas','Courier New',monospace;"); note.setWordWrap(True)
        right.addWidget(note)
        outer.addLayout(left, 45); outer.addLayout(right, 55)
        return w

    # ── Actualizar FK ─────────────────────────────────────────
    def _refresh_fk(self) -> None:
        qc = self.node.get_rad("joint_c"); qp = self.node.get_rad("joint_p"); qr = self.node.get_rad("joint_r")
        xyz = fk_geom(qc, qp, qr)
        self._fk_lx.setText(f"{xyz[0]:+.6f}"); self._fk_ly.setText(f"{xyz[1]:+.6f}"); self._fk_lz.setText(f"{xyz[2]:+.6f}")
        self._fk_frames, _ = fk_chain(qc, qp, qr)
        cur = max(self._lst.currentRow(), 0)
        self._lst.blockSignals(True); self._lst.clear();
        for name, _ in self._fk_frames: self._lst.addItem(name)
        self._lst.blockSignals(False); self._lst.setCurrentRow(min(cur, len(self._fk_frames) - 1)); self._show_mat()

    def _show_mat(self) -> None:
        idx = self._lst.currentRow()
        if idx < 0 or not hasattr(self, "_fk_frames") or idx >= len(self._fk_frames): return
        name, T = self._fk_frames[idx]
        p = T[:3, 3]; rpy = rot_to_rpy(T[:3, :3])
        self._txt.setPlainText(
            f"{name}\n{'─'*52}\n"
            f"xyz  [m]   = ({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f})\n"
            f"rpy  [rad] = ({rpy[0]:+.6f}, {rpy[1]:+.6f}, {rpy[2]:+.6f})\n"
            f"rpy  [°]   = ({math.degrees(rpy[0]):+.4f}, {math.degrees(rpy[1]):+.4f}, {math.degrees(rpy[2]):+.4f})\n\n"
            f"T (4×4):\n{fmt4(T)}"
        )

    # ── Resolver Newton ───────────────────────────────────────
    def _solve_newton(self) -> None:
        xd = np.array([self._xd.value(), self._yd.value(), self._zd.value()], dtype=float)
        q0 = np.radians(np.array([self._q0a.value(), self._q0b.value(), self._q0l.value()], dtype=float))
        eps      = self._spin_eps.value()
        max_iter = self._spin_maxiter.value()
        max_step = self._spin_maxstep.value()
        cond_w   = self._spin_cond.value()
        q_sol, iters, err_final, converged, log, sing_iters = newton_ik(xd, q0, eps, max_iter, max_step, cond_w)
        # Mostrar resultados
        self._oa.setText(f"{math.degrees(q_sol[0]):+.4f}°  ({q_sol[0]:+.6f} rad)")
        self._ob.setText(f"{math.degrees(q_sol[1]):+.4f}°  ({q_sol[1]:+.6f} rad)")
        self._ol.setText(f"{math.degrees(q_sol[2]):+.4f}°  ({q_sol[2]:+.6f} rad)")
        self._oit.setText(str(iters))
        self._oer.setText(f"{err_final:.2e} m")
        self._ocond.setText(f"{log[-1]['cond']:.2e}")
        self._ostat.setText("Convergió" if converged else "No convergió")
        # Mostrar singularidades
        if sing_iters:
            iters_str = ", ".join(str(i) for i in sing_iters)
            self._lbl_sing.setText(f"Jacobiano mal condicionado en iteraciones: {iters_str}")
            self._lbl_sing.setVisible(True)
        else:
            self._lbl_sing.setVisible(False)
        # Verificación FK(IK)
        xr, yr, zr = fk_geom(*q_sol)
        err = math.sqrt((xd[0]-xr)**2 + (xd[1]-yr)**2 + (xd[2]-zr)**2)
        self._vx.setText(f"{xr:+.6f}"); self._vy.setText(f"{yr:+.6f}"); self._vz.setText(f"{zr:+.6f}"); self._ve.setText(f"{err:.2e} m")
        # Matriz completa y log
        _, T = fk_chain(*q_sol)
        self._ik_mat.setPlainText(f"T0→efector:\n{fmt4(T)}")
        log_lines = []
        for entry in log:
            k = entry['k']; qd = entry['q_deg']; errn = entry['err']; cond = entry['cond']; det = entry['det']; sing = entry['singular']
            log_lines.append(
                f"k={k:3d}  q=[{qd[0]:+6.2f},{qd[1]:+6.2f},{qd[2]:+6.2f}]  |e|={errn:.2e}  cond={cond:.2e}  det={det:.2e}{'  SING' if sing else ''}"
            )
        self._log_txt.setPlainText("\n".join(log_lines))
        # Actualizar sliders y nodo
        self.node.set_rad("joint_c", q_sol[0]); self.node.set_rad("joint_p", q_sol[1]); self.node.set_rad("joint_r", q_sol[2]);
        self._sync_sliders(q_sol[0], q_sol[1], q_sol[2])

    def _sync_sliders(self, qc: float, qp: float, qr: float) -> None:
        for j, v in zip(JOINTS, [qc, qp, qr]):
            deg = max(LIMITS_DEG[j][0], min(LIMITS_DEG[j][1], math.degrees(v)))
            sl = self._sl[j]
            sl.blockSignals(True); sl.setValue(int(round(deg * 10))); sl.blockSignals(False)
            self._ed[j].setText(f"{deg:.1f}")
        self._refresh_fk()

    # ── Botones ───────────────────────────────────────────────
    def _do_zero(self) -> None:
        for j in JOINTS:
            self.node.set_deg(j, 0.0)
            sl = self._sl[j]; sl.blockSignals(True); sl.setValue(0); sl.blockSignals(False); self._ed[j].setText("0.0")
        self._refresh_fk()
    def _do_home(self) -> None:
        for j in JOINTS:
            ddeg = HOME_DEG[j]
            self.node.set_deg(j, ddeg)
            sl = self._sl[j]; sl.blockSignals(True); sl.setValue(int(ddeg * 10)); sl.blockSignals(False); self._ed[j].setText(f"{ddeg:.1f}")
        self._refresh_fk()


# ─────────────────── Utilidades de formato ───────────────────
def fmt4(T: np.ndarray) -> str:
    return "\n".join("  ".join(f"{v:+.4f}" for v in row) for row in T)


def rot_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    r11, r21, r31 = R[0,0], R[1,0], R[2,0]
    r32, r33 = R[2,1], R[2,2]
    pitch = math.atan2(-r31, math.sqrt(r11**2 + r21**2))
    if abs(math.cos(pitch)) < 1e-9:
        yaw = math.atan2(-R[1,2], R[1,1]); roll = 0.0
    else:
        yaw = math.atan2(r21, r11); roll = math.atan2(r32, r33)
    return roll, pitch, yaw


# ─────────────────── main ─────────────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    node = NewtonIKNode()
    app  = QtWidgets.QApplication([])
    win  = MainWindow(node)
    win.show()
    spin_t = QtCore.QTimer(); spin_t.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0)); spin_t.start(10)
    signal.signal(signal.SIGINT,  lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    app.exec_()
    try: node.destroy_node()
    except Exception: pass
    try: rclpy.shutdown()
    except Exception: pass


if __name__ == "__main__":
    main()