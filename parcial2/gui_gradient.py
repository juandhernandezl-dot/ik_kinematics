#!/usr/bin/env python3
"""
gui_gradient.py — Cinemática Inversa Numérica (Método del Gradiente) para LA_PATA_SOLA
=====================================================================================

Esta GUI permite resolver la cinemática inversa de la pata del robot
**LA_PATA_SOLA** mediante un método numérico de gradiente con búsqueda
espectral (Barzilai–Borwein) o tamaño de paso fijo.  El algoritmo de
gradiente se apoya en la cinemática directa geométrica extraída del URDF
del robot y calcula el Jacobiano de forma numérica por diferencias
finitas centrales.

La interfaz consta de dos pestañas: una para resolver la cinemática
inversa con parámetros ajustables (posición objetivo, valor inicial de
los ángulos, tolerancia, número máximo de iteraciones y configuración
del paso) y otra pestaña para visualizar la cinemática directa,
manipulando manualmente los ángulos articulares.  Se muestran las
matrices homogéneas de transformación (MTH) de cada eslabón así como
la posición y orientación del efector final.

Uso::

    ros2 run parcial2 gui_gradient

"""

import math
import signal
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5 import QtCore, QtWidgets

# ─────────────────── Parámetros del robot ───────────────────────
# Valores obtenidos a partir del URDF de LA_PATA_SOLA
z0   = 0.1439
dz_p = 0.075
a    = 0.0403
L1   = 0.3505
L2   = np.hypot(0.2905, -0.0010066)
k1   = -0.00365
k2   = 0.093563
phi  = 0.026978 + math.atan2(-0.0010066, 0.2905)
h    = z0 + dz_p
# Offset horizontal fijo (proyección en el plano) desde la cadera
d    = a + k1 + k2

# Nombres de las articulaciones coinciden con el URDF
JOINTS = ["joint_c", "joint_p", "joint_r"]

# Límites de cada articulación en grados (idénticos al URDF)
LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "joint_c": (-90.0, 90.0),
    "joint_p": (-90.0, 90.0),
    "joint_r": (-90.0, 90.0),
}

# Posición de home en grados (todas las articulaciones en cero)
HOME_DEG: Dict[str, float] = {"joint_c": 0.0, "joint_p": 0.0, "joint_r": 0.0}

# Frecuencia de publicación en /joint_states
PUBLISH_HZ = 30.0

# ─────────────────── Parámetros del método del gradiente ─────────
EPSILON_DEFAULT      = 1e-4     # tolerancia del error cartesiano [m]
MAX_ITER_DEFAULT     = 5000     # iteraciones máximas permitidas
H_DIFF_DEFAULT       = 1e-6     # paso para diferencias finitas [rad]
MAX_STEP_NORM_DEF    = 0.35     # límite de ||Δq|| por iteración [rad]

# Modo de alpha fijo
ALPHA_FIXED_DEF      = 5.0

# Modo Barzilai–Borwein (BB1)
ALPHA_MIN_DEF        = 1e-3
ALPHA_MAX_DEF        = 50.0
SAFEGUARD_BETA_DEF   = 0.5      # factor de reducción si no disminuye g(q)
SAFEGUARD_MAX_TRIES_DEF = 20     # máximo de intentos en el safeguard

# Límites articulares en radianes
Q_MIN = -math.pi / 2
Q_MAX =  math.pi / 2

# ─────────────────── Utilidades matemáticas ─────────────────────

def clamp(v: float, lo: float, hi: float) -> float:
    """Recorta v al rango [lo, hi]."""
    return max(lo, min(hi, v))


def rpy_to_mat(r: float, p: float, y_: float) -> np.ndarray:
    """Convierte roll, pitch, yaw a una matriz de rotación 3×3."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y_), math.sin(y_)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=float)


def make_transform(xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> np.ndarray:
    """Genera una matriz de transformación homogénea 4×4 a partir de una traslación y una rotación RPY."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_mat(*rpy)
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


# ─────────────────── Cinemática directa ─────────────────────────

def fk_geom(qc: float, qp: float, qr: float) -> np.ndarray:
    """Cinemática directa geométrica para LA_PATA_SOLA.

    Calcula la posición cartesiana del efector final a partir de los
    ángulos articulares qc (cadera), qp (parte superior) y qr (rodilla).
    Retorna un array [x, y, z].
    """
    s_val = L1 * math.cos(qp) + L2 * math.cos(qp + qr - phi)
    z_val = h - L1 * math.sin(qp) - L2 * math.sin(qp + qr - phi)
    x_val = -d * math.cos(qc) - s_val * math.sin(qc)
    y_val =  d * math.sin(qc) - s_val * math.cos(qc)
    return np.array([x_val, y_val, z_val], dtype=float)


def fk_chain(qc: float, qp: float, qr: float) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    """Cinemática directa completa basada en el URDF.

    Devuelve una lista de parejas (nombre, matriz 4×4) para cada eslabón
    y la matriz final T0→efector.
    """
    # Transformación base→link_c: traslación en z0 y rotación -qc sobre Z
    T01 = make_transform((0.0, 0.0, z0), (0.0, 0.0, 0.0)) @ make_transform((0.0, 0.0, 0.0), (0.0, 0.0, -qc))
    # Transformación link_c→link_p: traslación (-a, 0, dz_p), orientación inicial RPY(π/2,0,-π/2), luego rotación -qp
    T12 = make_transform((-a, 0.0, dz_p), (math.pi/2, 0.0, -math.pi/2)) @ make_transform((0.0, 0.0, 0.0), (0.0, 0.0, -qp))
    # Transformación link_p→link_r: traslación (0.3505, -2.9381e-05, k1), rotación fija RPY(0,0,0.026978), luego rotación -qr
    T23 = make_transform((0.3505, -2.9381e-05, k1), (0.0, 0.0, 0.026978)) @ make_transform((0.0, 0.0, 0.0), (0.0, 0.0, -qr))
    # Transformación link_r→efector: traslación final y orientación fija del efector
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


def jacobian_num(q: np.ndarray, h: float = H_DIFF_DEFAULT) -> np.ndarray:
    """Calcula el Jacobiano numérico 3×3 por diferencias finitas centrales.

    Cada columna se obtiene perturbando solo una articulación en ±h y
    evaluando la cinemática directa para estimar la derivada.
    """
    J = np.zeros((3, 3), dtype=float)
    for j in range(3):
        dq = np.zeros(3, dtype=float)
        dq[j] = h
        fp = fk_geom(*(q + dq))
        fm = fk_geom(*(q - dq))
        J[:, j] = (fp - fm) / (2.0 * h)
    return J


# ─────────────────── Funciones de optimización ──────────────────

def cost_from_error(e: np.ndarray) -> float:
    return 0.5 * float(np.dot(e, e))


def exact_jt_alpha(J: np.ndarray, d: np.ndarray, fallback: float) -> float:
    """Paso exacto sobre el modelo linealizado del método J^T.

    Si d = J^T e, entonces una opción robusta para la primera iteración es:
        α = (‖d‖²) / (‖J d‖²)
    En caso de que el denominador sea muy pequeño, se utiliza el valor de
    fallback proporcionado.
    """
    Jd = J @ d
    den = float(np.dot(Jd, Jd))
    num = float(np.dot(d, d))
    if den <= 1e-14 or num <= 1e-14:
        return float(fallback)
    return num / den


def bb1_alpha(prev_q: np.ndarray, q: np.ndarray, prev_grad: np.ndarray, grad: np.ndarray,
              alpha_min: float, alpha_max: float, fallback: float) -> float:
    """Paso espectral BB1 con clipping y fallback robusto."""
    s = q - prev_q
    y = grad - prev_grad
    sty = float(np.dot(s, y))
    sts = float(np.dot(s, s))
    if sts <= 1e-14 or abs(sty) <= 1e-14 or sty <= 0.0:
        return float(clamp(fallback, alpha_min, alpha_max))
    alpha = sts / sty
    if not np.isfinite(alpha):
        alpha = fallback
    return float(clamp(alpha, alpha_min, alpha_max))


def safe_step(q: np.ndarray, step: np.ndarray, max_step_norm: float) -> np.ndarray:
    """Limita la norma de step para evitar saltos excesivos durante la iteración."""
    step = np.array(step, dtype=float)
    nrm = float(np.linalg.norm(step))
    if nrm > max_step_norm and nrm > 1e-14:
        step *= max_step_norm / nrm
    return step


def gradient_ik(
    xd: np.ndarray,
    q0: np.ndarray,
    epsilon: float,
    max_iter: int,
    mode: str,
    alpha_fixed: float,
    alpha_min: float,
    alpha_max: float,
    safeguard_beta: float,
    safeguard_max_tries: int,
    max_step_norm: float,
    h_diff: float,
) -> Tuple[np.ndarray, int, float, bool, List[Dict], List[float], np.ndarray]:
    """Resuelve la cinemática inversa mediante el método del gradiente.

    Parámetros:
        xd:     posición objetivo [x,y,z]
        q0:     vector inicial de ángulos [qc, qp, qr] en radianes
        epsilon: tolerancia para la norma del error
        max_iter: iteraciones máximas permitidas
        mode:   'fixed' para paso fijo, 'bb' para BB1 con safeguard
        alpha_fixed: valor de α en modo fijo
        alpha_min, alpha_max: límites para α en modo BB
        safeguard_beta: factor de reducción en safeguard
        safeguard_max_tries: máximo de intentos en safeguard
        max_step_norm: límite de la norma del paso por iteración
        h_diff: paso para diferencias finitas en el Jacobiano

    Devuelve:
        (q_sol, iters, err_final, convergió, log, alpha_hist, J_final)
    """
    q = np.array(q0, dtype=float).copy()
    log: List[Dict] = []
    alpha_hist: List[float] = []
    prev_q: Optional[np.ndarray] = None
    prev_grad: Optional[np.ndarray] = None
    for k in range(max_iter):
        f = fk_geom(*q)
        e = xd - f
        err_norm = float(np.linalg.norm(e))
        g0 = cost_from_error(e)
        J = jacobian_num(q, h=h_diff)
        d = J.T @ e                # dirección de descenso
        grad = -d                  # gradiente de g(q)
        dir_norm = float(np.linalg.norm(d))
        if err_norm < epsilon:
            return q, k, err_norm, True, log, alpha_hist, J
        # cálculo de alpha según el modo
        if mode == 'bb':
            if prev_q is None or prev_grad is None:
                alpha = exact_jt_alpha(J, d, fallback=alpha_fixed)
                alpha = float(clamp(alpha, alpha_min, alpha_max))
                alpha_source = 'JT-exact'
            else:
                alpha = bb1_alpha(prev_q, q, prev_grad, grad,
                                  alpha_min=alpha_min,
                                  alpha_max=alpha_max,
                                  fallback=alpha_fixed)
                alpha_source = 'BB1'
        else:
            alpha = float(alpha_fixed)
            alpha_source = 'fixed'
        accepted_alpha = alpha
        safeguard_tries = 0
        # Propuesta inicial de paso
        step = safe_step(q=q, step=alpha * d, max_step_norm=max_step_norm)
        q_try = np.clip(q + step, Q_MIN, Q_MAX)
        e_try = xd - fk_geom(*q_try)
        g_try = cost_from_error(e_try)
        # Safeguard monotono solo para modo BB
        if mode == 'bb':
            while g_try > g0 and safeguard_tries < safeguard_max_tries:
                accepted_alpha *= safeguard_beta
                accepted_alpha = float(max(accepted_alpha, alpha_min))
                step = safe_step(q=q, step=accepted_alpha * d, max_step_norm=max_step_norm)
                q_try = np.clip(q + step, Q_MIN, Q_MAX)
                e_try = xd - fk_geom(*q_try)
                g_try = cost_from_error(e_try)
                safeguard_tries += 1
                alpha_source = 'BB1+safeguard'
        # Registrar log
        log.append({
            'k': k,
            'q_deg': np.degrees(q).copy(),
            'err': err_norm,
            'g': g0,
            'alpha': float(accepted_alpha),
            'alpha_source': alpha_source,
            'dir_norm': dir_norm,
            'step_norm': float(np.linalg.norm(step)),
            'safeguard_tries': safeguard_tries,
        })
        alpha_hist.append(float(accepted_alpha))
        prev_q = q.copy()
        prev_grad = grad.copy()
        q = q_try
    # no convergió en max_iter
    f = fk_geom(*q)
    e = xd - f
    J_final = jacobian_num(q, h=h_diff)
    return q, max_iter, float(np.linalg.norm(e)), False, log, alpha_hist, J_final


# ─────────────────── Formato de matrices ────────────────────────

def fmt4(T: np.ndarray) -> str:
    """Devuelve una cadena con el contenido de una matriz 4×4 con formato fijo."""
    return "\n".join("  ".join(f"{v:+.4f}" for v in row) for row in T)


def rot_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Convierte una matriz de rotación 3×3 a ángulos roll, pitch, yaw."""
    r11, r21, r31 = R[0, 0], R[1, 0], R[2, 0]
    r32, r33 = R[2, 1], R[2, 2]
    pitch = math.atan2(-r31, math.sqrt(r11 ** 2 + r21 ** 2))
    if abs(math.cos(pitch)) < 1e-9:
        yaw = math.atan2(-R[1, 2], R[1, 1])
        roll = 0.0
    else:
        yaw = math.atan2(r21, r11)
        roll = math.atan2(r32, r33)
    return roll, pitch, yaw


# ─────────────────── Nodo ROS 2 ─────────────────────────────────
class GradientIKNode(Node):
    def __init__(self) -> None:
        # Nombre de nodo acorde a la pata
        super().__init__('leg_kinematics_gradient_gui')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        # Almacena ángulos en radianes por articulación
        self._rad: Dict[str, float] = {j: 0.0 for j in JOINTS}
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)

    def set_deg(self, j: str, deg: float) -> None:
        lo, hi = LIMITS_DEG[j]
        self._rad[j] = math.radians(clamp(deg, lo, hi))

    def set_rad(self, j: str, r: float) -> None:
        lo, hi = LIMITS_DEG[j]
        self._rad[j] = clamp(r, math.radians(lo), math.radians(hi))

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


# ─────────────────── Estilos ────────────────────────────────────
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
QScrollArea, QScrollArea > QWidget > QWidget {
    background: #000000;
    border: none;
}
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
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
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
QRadioButton { color: #d0d0d0; spacing: 6px; }
QRadioButton::indicator { width: 14px; height: 14px; }
QRadioButton::indicator:checked { background: #4a9eff; border-radius: 7px; }
QRadioButton::indicator:unchecked {
    background: #111111; border: 1px solid #2a2a2a; border-radius: 7px;
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
        QSlider::sub-page:horizontal {{
            background: {color}; border-radius: 3px;
        }}
    """)
    return sl


def _wrap_vscroll(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    """Envuelve un widget en un QScrollArea vertical para permitir scroll."""
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    scroll.setWidget(widget)
    return scroll


# Metadatos de las articulaciones para la pestaña de FK
_JOINT_META = {
    'joint_c': {
        'sym': 'α',
        'name': 'Cadera',
        'axis': 'eje Z (negativo)',
        'color': '#e05c5c',
        'sl_color': '#e05c5c',
    },
    'joint_p': {
        'sym': 'β',
        'name': 'Fémur',
        'axis': 'eje Z (negativo)',
        'color': '#4a9eff',
        'sl_color': '#4a9eff',
    },
    'joint_r': {
        'sym': 'λ',
        'name': 'Rodilla',
        'axis': 'eje Z (negativo)',
        'color': '#5ec46e',
        'sl_color': '#5ec46e',
    },
}


# ─────────────────── Ventana principal ──────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, node: GradientIKNode) -> None:
        super().__init__()
        self.node = node
        self.setWindowTitle("Cinemática Inversa – Método del Gradiente – LA_PATA_SOLA")
        self.setMinimumSize(1160, 760)
        self.setStyleSheet(STYLE)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_gradient_tab(), "  Método del Gradiente (IK)  ")
        tabs.addTab(self._build_fk_tab(), "  Cinemática Directa (FK)  ")
        self.setCentralWidget(tabs)

        # Establecer posición home al inicio
        self._do_home()

    # ── Helpers para spin boxes ─────────────────────────────────
    def _spin_xyz(self, val: float) -> QtWidgets.QDoubleSpinBox:
        """Crea un spin box para una coordenada cartesiana.

        La amplitud se ajusta para cubrir el espacio de trabajo de LA_PATA_SOLA.  El
        rango por defecto era ±0.6 m en la versión original, pero el
        valor de la componente y en la posición home (aprox. −0.64 m) quedaba
        fuera de este intervalo.  Se amplía el rango a ±1.0 m para permitir
        introducir posiciones realistas y reproducir exactamente la cinemática
        directa en las pruebas.
        """
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(-1.0, 1.0)
        sb.setDecimals(5)
        sb.setSingleStep(0.005)
        sb.setValue(val)
        return sb

    def _make_deg_spin(self, val: float) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(-90.0, 90.0)
        sb.setDecimals(2)
        sb.setSingleStep(5.0)
        sb.setValue(val)
        return sb

    # ── Construye la pestaña de Gradiente ───────────────────────
    def _build_gradient_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(14)
        # Columna izquierda (con scroll)
        left_widget = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 4, 0)
        left.setSpacing(8)
        # Posición objetivo
        grp_xd = QtWidgets.QGroupBox("Posición objetivo del pie  [m]")
        grid_xd = QtWidgets.QGridLayout(grp_xd)
        grid_xd.setSpacing(6)
        # Valor inicial de la FK en home
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
                f"color:{color}; background:#111111; border:1px solid #2a2a2a;"
                " border-radius:4px; padding:3px 5px;"
            )
            grid_xd.addWidget(lbl, i, 0)
            grid_xd.addWidget(sp, i, 1)
        left.addWidget(grp_xd)
        # Valor inicial de q0
        grp_q0 = QtWidgets.QGroupBox("Valor inicial  q₀  [°]")
        grid_q0 = QtWidgets.QGridLayout(grp_q0)
        grid_q0.setSpacing(6)
        self._q0c = self._make_deg_spin(0.0)
        self._q0p = self._make_deg_spin(0.0)
        self._q0r = self._make_deg_spin(0.0)
        for i, (sym, sp, color) in enumerate([
            ("α₀ (cadera):", self._q0c, "#e05c5c"),
            ("β₀ (fémur):",  self._q0p, "#4a9eff"),
            ("λ₀ (rodilla):", self._q0r, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet(f"color:{color}; font-size:14px;")
            grid_q0.addWidget(lbl, i, 0)
            grid_q0.addWidget(sp, i, 1)
        left.addWidget(grp_q0)
        # Configuración de alpha
        grp_alpha = QtWidgets.QGroupBox("Tamaño de paso α (solo para Gradiente)")
        v_alpha = QtWidgets.QVBoxLayout(grp_alpha)
        v_alpha.setSpacing(6)
        row_radio = QtWidgets.QHBoxLayout()
        self._rb_fixed = QtWidgets.QRadioButton("Alpha fijo")
        self._rb_bb    = QtWidgets.QRadioButton("Barzilai–Borwein (BB1)")
        self._rb_bb.setChecked(True)
        row_radio.addWidget(self._rb_fixed)
        row_radio.addWidget(self._rb_bb)
        v_alpha.addLayout(row_radio)
        # Grupo para alpha fijo
        self._grp_fixed = QtWidgets.QGroupBox("Alpha fijo")
        self._grp_fixed.setStyleSheet("QGroupBox{color:#888888; font-size:12px;}")
        form_fixed = QtWidgets.QFormLayout(self._grp_fixed)
        form_fixed.setSpacing(4)
        self._spin_alpha_fixed = QtWidgets.QDoubleSpinBox()
        self._spin_alpha_fixed.setRange(1e-5, 100.0)
        self._spin_alpha_fixed.setDecimals(5)
        self._spin_alpha_fixed.setSingleStep(0.5)
        self._spin_alpha_fixed.setValue(ALPHA_FIXED_DEF)
        self._spin_alpha_fixed.setToolTip(
            "Solo recomendado para pruebas.\n"
            "Si α es muy alto puede divergir;\nsi es muy pequeño convergerá lento."
        )
        form_fixed.addRow("α:", self._spin_alpha_fixed)
        v_alpha.addWidget(self._grp_fixed)
        # Grupo para BB1
        self._grp_bb = QtWidgets.QGroupBox("BB1 + safeguard monótono (recomendado)")
        self._grp_bb.setStyleSheet("QGroupBox{color:#888888; font-size:12px;}")
        form_bb = QtWidgets.QFormLayout(self._grp_bb)
        form_bb.setSpacing(4)
        self._spin_alpha_min = QtWidgets.QDoubleSpinBox()
        self._spin_alpha_min.setRange(1e-6, 10.0)
        self._spin_alpha_min.setDecimals(6)
        self._spin_alpha_min.setSingleStep(1e-3)
        self._spin_alpha_min.setValue(ALPHA_MIN_DEF)
        self._spin_alpha_min.setToolTip("Clip inferior para α_BB")
        self._spin_alpha_max = QtWidgets.QDoubleSpinBox()
        self._spin_alpha_max.setRange(0.1, 500.0)
        self._spin_alpha_max.setDecimals(3)
        self._spin_alpha_max.setSingleStep(1.0)
        self._spin_alpha_max.setValue(ALPHA_MAX_DEF)
        self._spin_alpha_max.setToolTip("Clip superior para α_BB")
        self._spin_safeguard_beta = QtWidgets.QDoubleSpinBox()
        self._spin_safeguard_beta.setRange(0.05, 0.99)
        self._spin_safeguard_beta.setDecimals(3)
        self._spin_safeguard_beta.setSingleStep(0.05)
        self._spin_safeguard_beta.setValue(SAFEGUARD_BETA_DEF)
        self._spin_safeguard_beta.setToolTip("Factor de reducción si el paso no baja g(q)")
        self._spin_safeguard_tries = QtWidgets.QSpinBox()
        self._spin_safeguard_tries.setRange(1, 100)
        self._spin_safeguard_tries.setValue(SAFEGUARD_MAX_TRIES_DEF)
        self._spin_safeguard_tries.setToolTip("Máximos intentos del safeguard")
        form_bb.addRow("α_min:", self._spin_alpha_min)
        form_bb.addRow("α_max:", self._spin_alpha_max)
        form_bb.addRow("β_safeguard:", self._spin_safeguard_beta)
        form_bb.addRow("intentos safeguard:", self._spin_safeguard_tries)
        v_alpha.addWidget(self._grp_bb)
        # Conexiones para habilitar/deshabilitar grupos
        self._rb_fixed.toggled.connect(self._on_mode_changed)
        self._rb_bb.toggled.connect(self._on_mode_changed)
        self._on_mode_changed()
        left.addWidget(grp_alpha)
        # Otros parámetros
        grp_p = QtWidgets.QGroupBox("Otros parámetros")
        form_p = QtWidgets.QFormLayout(grp_p)
        form_p.setSpacing(6)
        self._spin_eps = QtWidgets.QDoubleSpinBox()
        self._spin_eps.setRange(1e-10, 1e-1)
        self._spin_eps.setDecimals(8)
        self._spin_eps.setSingleStep(1e-5)
        self._spin_eps.setValue(EPSILON_DEFAULT)
        self._spin_maxiter = QtWidgets.QSpinBox()
        self._spin_maxiter.setRange(1, 50000)
        self._spin_maxiter.setValue(MAX_ITER_DEFAULT)
        self._spin_maxstep = QtWidgets.QDoubleSpinBox()
        self._spin_maxstep.setRange(0.01, math.pi)
        self._spin_maxstep.setDecimals(3)
        self._spin_maxstep.setSingleStep(0.05)
        self._spin_maxstep.setValue(MAX_STEP_NORM_DEF)
        self._spin_maxstep.setToolTip("Limita ||Δq|| por iteración para mayor estabilidad")
        self._spin_hdiff = QtWidgets.QDoubleSpinBox()
        self._spin_hdiff.setRange(1e-8, 1e-2)
        self._spin_hdiff.setDecimals(8)
        self._spin_hdiff.setSingleStep(1e-6)
        self._spin_hdiff.setValue(H_DIFF_DEFAULT)
        self._spin_hdiff.setToolTip(
            "Paso usado para el Jacobiano NUMÉRICO:\n"
            "J[:,j] = (f(q+h e_j) − f(q−h e_j))/(2h)"
        )
        form_p.addRow("Tolerancia  ε [m]:", self._spin_eps)
        form_p.addRow("Iteraciones máx.:", self._spin_maxiter)
        form_p.addRow("||Δq|| máx. [rad]:", self._spin_maxstep)
        form_p.addRow("h Jacobiano [rad]:", self._spin_hdiff)
        left.addWidget(grp_p)
        # Botón resolver
        btn_solve = QtWidgets.QPushButton("▶   Resolver  Cinemática Inversa  (Gradiente)")
        btn_solve.setStyleSheet(
            "background:#4a9eff; color:#000000; font-weight:bold;"
            " font-size:13px; padding:8px 16px; border-radius:6px;"
        )
        btn_solve.clicked.connect(self._solve_gradient)
        left.addWidget(btn_solve)
        # Salida de los ángulos calculados
        grp_out = QtWidgets.QGroupBox("Resultado  —  ángulos calculados")
        grid_out = QtWidgets.QGridLayout(grp_out)
        grid_out.setSpacing(5)
        self._oc = QtWidgets.QLabel("—")
        self._op = QtWidgets.QLabel("—")
        self._or = QtWidgets.QLabel("—")
        self._oit = QtWidgets.QLabel("—")
        self._oer = QtWidgets.QLabel("—")
        self._oalpha = QtWidgets.QLabel("—")
        self._omode = QtWidgets.QLabel("—")
        self._ostat = QtWidgets.QLabel("")
        self._ostat.setStyleSheet("font-size:13px; font-weight:bold;")
        for i, (sym, val_w, color) in enumerate([
            ("α =", self._oc, "#e05c5c"),
            ("β =", self._op, "#4a9eff"),
            ("λ =", self._or, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:14px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:15px; font-weight:bold;"
                " font-family:'Consolas','Courier New',monospace;"
            )
            grid_out.addWidget(lbl, i, 0)
            grid_out.addWidget(val_w, i, 1)
        for i, (sym, val_w) in enumerate([
            ("Iteraciones:", self._oit),
            ("Error final:", self._oer),
            ("α final:", self._oalpha),
            ("Modo:", self._omode),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:13px;")
            val_w.setStyleSheet(
                "color:#d0d0d0; font-size:13px;"
                " font-family:'Consolas','Courier New',monospace;"
            )
            grid_out.addWidget(lbl, 3 + i, 0)
            grid_out.addWidget(val_w, 3 + i, 1)
        grid_out.addWidget(self._ostat, 7, 0, 1, 2)
        left.addWidget(grp_out)
        # Verificación FK después de IK
        grp_v = QtWidgets.QGroupBox("Verificación  —  FK(IK(p))  →  posición reconstruida")
        grid_v = QtWidgets.QGridLayout(grp_v)
        grid_v.setSpacing(6)
        self._vx = QtWidgets.QLabel("—")
        self._vy = QtWidgets.QLabel("—")
        self._vz = QtWidgets.QLabel("—")
        self._ve = QtWidgets.QLabel("—")
        self._ve.setStyleSheet("color:#5ec46e; font-weight:bold;")
        for i, (sym, val_w, color) in enumerate([
            ("x_rec =", self._vx, "#4dd0e1"),
            ("y_rec =", self._vy, "#4dd0e1"),
            ("z_rec =", self._vz, "#4dd0e1"),
            ("‖error‖ =", self._ve, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:14px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:13px;"
                " font-family:'Consolas','Courier New',monospace;"
            )
            grid_v.addWidget(lbl, i, 0)
            grid_v.addWidget(val_w, i, 1)
        left.addWidget(grp_v)
        left.addStretch(1)
        # Columna derecha
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        # MTH después de IK
        grp_mat = QtWidgets.QGroupBox("T0→efector  tras IK  (cadena URDF)")
        vm = QtWidgets.QVBoxLayout(grp_mat)
        self._ik_mat = QtWidgets.QPlainTextEdit()
        self._ik_mat.setReadOnly(True)
        self._ik_mat.setMinimumHeight(160)
        vm.addWidget(self._ik_mat)
        right.addWidget(grp_mat)
        # Jacobiano final
        grp_jnum = QtWidgets.QGroupBox("Jacobiano final usado por el solver  (NUMÉRICO)")
        vjnum = QtWidgets.QVBoxLayout(grp_jnum)
        self._jnum_txt = QtWidgets.QPlainTextEdit()
        self._jnum_txt.setReadOnly(True)
        self._jnum_txt.setMinimumHeight(160)
        vjnum.addWidget(self._jnum_txt)
        right.addWidget(grp_jnum)
        # Log de iteraciones
        grp_log = QtWidgets.QGroupBox("Log de iteraciones")
        vl = QtWidgets.QVBoxLayout(grp_log)
        self._log_txt = QtWidgets.QPlainTextEdit()
        self._log_txt.setReadOnly(True)
        self._log_txt.setMinimumHeight(160)
        vl.addWidget(self._log_txt)
        right.addWidget(grp_log)
        # Nota explicativa
        note = QtWidgets.QLabel(
            "Método del Gradiente:\n\n"
            "  q_{k+1} = q_k + α_k·J(q_k)^T·(x_d − f(q_k))\n\n"
            "Jacobiano empleado en esta GUI:\n"
            "  • NUMÉRICO por diferencias finitas centrales\n"
            "  • No analítico\n"
            "  • J[:,j] = (f(q+h e_j) − f(q−h e_j))/(2h)\n\n"
            "Tamaño de paso implementado para Gradiente:\n"
            "  • Barzilai–Borwein (BB1) como modo recomendado\n"
            "  • Alpha fijo para pruebas\n"
            "  • Salvaguarda monótona para estabilidad\n\n"
            "Modelado de LA_PATA_SOLA:\n"
            "  x = −d·cos(q_c) − s·sin(q_c)\n"
            "  y =  d·sin(q_c) − s·cos(q_c)\n"
            "  z =  h − L₁·sin(q_p) − L₂·sin(q_p + q_r − φ)\n"
            "  s =  L₁·cos(q_p) + L₂·cos(q_p + q_r − φ)\n"
        )
        note.setStyleSheet(
            "color:#d0d0d0; background:#000000; font-size:13px;"
            " font-family:'Consolas','Courier New',monospace; padding:6px;"
        )
        note.setWordWrap(True)
        right.addWidget(note)
        right.addStretch(1)
        # Agregar columnas al layout principal
        outer.addWidget(_wrap_vscroll(left_widget), 48)
        outer.addLayout(right, 52)
        return w

    # ── Construye la pestaña de FK ─────────────────────────────
    def _build_fk_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(14)
        # Columna izquierda
        left_widget = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 4, 0)
        left.setSpacing(10)
        grp_j = QtWidgets.QGroupBox("Ángulos articulares")
        vj = QtWidgets.QVBoxLayout(grp_j)
        vj.setSpacing(6)
        self._sl: Dict[str, QtWidgets.QSlider] = {}
        self._ed: Dict[str, QtWidgets.QLineEdit] = {}
        for j in JOINTS:
            meta = _JOINT_META[j]
            top_row = QtWidgets.QHBoxLayout()
            top_row.setSpacing(6)
            sym_lbl = QtWidgets.QLabel(meta['sym'])
            sym_lbl.setStyleSheet(
                f"color:{meta['color']}; font-size:22px; font-weight:bold; min-width:22px;"
            )
            sym_lbl.setAlignment(QtCore.Qt.AlignCenter)
            name_lbl = QtWidgets.QLabel(
                f"{meta['name']}   <span style='color:#666666;font-size:12px;'>"
                f"{meta['axis']}</span>"
            )
            name_lbl.setTextFormat(QtCore.Qt.RichText)
            name_lbl.setStyleSheet(f"color:{meta['color']}; font-size:14px;")
            ed = QtWidgets.QLineEdit("0.0")
            ed.setFixedWidth(70)
            ed.setAlignment(QtCore.Qt.AlignRight)
            deg_lbl = QtWidgets.QLabel("°")
            deg_lbl.setStyleSheet("color:#555555; font-size:14px;")
            top_row.addWidget(sym_lbl)
            top_row.addWidget(name_lbl, 1)
            top_row.addWidget(ed)
            top_row.addWidget(deg_lbl)
            # Slider
            bot_row = QtWidgets.QHBoxLayout()
            bot_row.setSpacing(4)
            lo, hi = LIMITS_DEG[j]
            lo_lbl = QtWidgets.QLabel(f"{int(lo)}°")
            lo_lbl.setStyleSheet("color:#444444; font-size:12px; min-width:28px;")
            lo_lbl.setAlignment(QtCore.Qt.AlignRight)
            sl = _colored_slider(meta['sl_color'])
            sl.setMinimum(int(lo * 10))
            sl.setMaximum(int(hi * 10))
            sl.setValue(0)
            hi_lbl = QtWidgets.QLabel(f"{int(hi)}°")
            hi_lbl.setStyleSheet("color:#444444; font-size:12px; min-width:28px;")
            bot_row.addWidget(lo_lbl)
            bot_row.addWidget(sl, 1)
            bot_row.addWidget(hi_lbl)
            # Callbacks para slider y edit
            def _sl_cb(val, joint=j, edit=ed):
                edit.setText(f"{val / 10.0:.1f}")
                self.node.set_deg(joint, val / 10.0)
                self._refresh_fk()
            def _ed_cb(joint=j, s=sl):
                try:
                    v = float(self._ed[joint].text())
                except ValueError:
                    return
                lo2, hi2 = LIMITS_DEG[joint]
                v = clamp(v, lo2, hi2)
                s.blockSignals(True)
                s.setValue(int(round(v * 10)))
                s.blockSignals(False)
                self._ed[joint].setText(f"{v:.1f}")
                self.node.set_deg(joint, v)
                self._refresh_fk()
            sl.valueChanged.connect(_sl_cb)
            ed.returnPressed.connect(_ed_cb)
            ed.editingFinished.connect(_ed_cb)
            block = QtWidgets.QVBoxLayout()
            block.setSpacing(2)
            block.addLayout(top_row)
            block.addLayout(bot_row)
            vj.addLayout(block)
            if j != JOINTS[-1]:
                line = QtWidgets.QFrame()
                line.setFrameShape(QtWidgets.QFrame.HLine)
                line.setStyleSheet("color:#1e1e1e;")
                vj.addWidget(line)
            self._sl[j] = sl
            self._ed[j] = ed
        left.addWidget(grp_j)
        # Posición del pie en FK
        grp_pos = QtWidgets.QGroupBox("Posición del pie  [m]  —  FK geométrica")
        grid_pos = QtWidgets.QGridLayout(grp_pos)
        grid_pos.setSpacing(6)
        grid_pos.setColumnStretch(1, 1)
        self._fk_lx = QtWidgets.QLabel("—")
        self._fk_ly = QtWidgets.QLabel("—")
        self._fk_lz = QtWidgets.QLabel("—")
        for i, (lbl_txt, val_w, color) in enumerate([
            ("x =", self._fk_lx, "#e05c5c"),
            ("y =", self._fk_ly, "#5ec46e"),
            ("z =", self._fk_lz, "#4a9eff"),
        ]):
            lbl = QtWidgets.QLabel(lbl_txt)
            lbl.setStyleSheet("color:#555555; font-size:14px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:16px; font-weight:bold;"
                " font-family:'Consolas','Courier New',monospace;"
            )
            grid_pos.addWidget(lbl, i, 0)
            grid_pos.addWidget(val_w, i, 1)
        left.addWidget(grp_pos)
        # Botones Zero y Home
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        b0 = QtWidgets.QPushButton("⟳  Zero")
        bh = QtWidgets.QPushButton("⌂  Home (0°)")
        for btn in (b0, bh):
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(b0)
        btn_row.addWidget(bh)
        left.addLayout(btn_row)
        b0.clicked.connect(self._do_zero)
        bh.clicked.connect(self._do_home)
        left.addStretch(1)
        # Columna derecha de FK
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        grp_mat = QtWidgets.QGroupBox("Matrices MTH  —  iguales a tf2_echo")
        vm = QtWidgets.QVBoxLayout(grp_mat)
        vm.setSpacing(6)
        self._fk_lst = QtWidgets.QListWidget()
        self._fk_lst.setFixedHeight(80)
        self._fk_lst.setStyleSheet(
            "QListWidget{font-size:12px;} QListWidget::item{padding:4px 8px;}"
        )
        self._fk_txt = QtWidgets.QPlainTextEdit()
        self._fk_txt.setReadOnly(True)
        self._fk_txt.setMinimumHeight(220)
        self._fk_lst.currentRowChanged.connect(self._show_fk_mat)
        vm.addWidget(self._fk_lst)
        vm.addWidget(self._fk_txt)
        right.addWidget(grp_mat, 1)
        note = QtWidgets.QLabel(
            "FK geométrica usada por la GUI:\n\n"
            "  s  =  L₁·cos(q_p) + L₂·cos(q_p + q_r − φ)\n"
            "  x  = −d·cos(q_c) − s·sin(q_c)\n"
            "  y  =  d·sin(q_c) − s·cos(q_c)\n"
            "  z  =  h − L₁·sin(q_p) − L₂·sin(q_p + q_r − φ)\n\n"
            "El Jacobiano del solver de gradiente se obtiene\n"
            "NUMÉRICAMENTE a partir de esta misma FK."
        )
        note.setStyleSheet(
            "color:#d0d0d0; background:#000000; font-size:13px;"
            " font-family:'Consolas','Courier New',monospace; padding:6px;"
        )
        note.setWordWrap(True)
        right.addWidget(note)
        outer.addWidget(_wrap_vscroll(left_widget), 45)
        outer.addLayout(right, 55)
        return w

    # ── Gestión de modos de alpha ───────────────────────────────
    def _on_mode_changed(self) -> None:
        use_bb = self._rb_bb.isChecked()
        self._grp_fixed.setEnabled(not use_bb)
        self._grp_bb.setEnabled(use_bb)

    # ── Lógica para resolver gradiente ───────────────────────────
    def _solve_gradient(self) -> None:
        xd = np.array([self._xd.value(), self._yd.value(), self._zd.value()], dtype=float)
        q0 = np.radians([self._q0c.value(), self._q0p.value(), self._q0r.value()])
        eps = self._spin_eps.value()
        max_iter = self._spin_maxiter.value()
        mode = 'bb' if self._rb_bb.isChecked() else 'fixed'
        alpha_fixed = self._spin_alpha_fixed.value()
        alpha_min = self._spin_alpha_min.value()
        alpha_max = self._spin_alpha_max.value()
        safeguard_beta = self._spin_safeguard_beta.value()
        safeguard_tries = self._spin_safeguard_tries.value()
        max_step_norm = self._spin_maxstep.value()
        h_diff = self._spin_hdiff.value()
        try:
            q_sol, iters, err, ok, log, alpha_hist, J_final = gradient_ik(
                xd=xd,
                q0=q0,
                epsilon=eps,
                max_iter=max_iter,
                mode=mode,
                alpha_fixed=alpha_fixed,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                safeguard_beta=safeguard_beta,
                safeguard_max_tries=safeguard_tries,
                max_step_norm=max_step_norm,
                h_diff=h_diff,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
        q_deg = np.degrees(q_sol)
        self._oc.setText(f"{q_deg[0]:+.4f}°   ({q_sol[0]:+.6f} rad)")
        self._op.setText(f"{q_deg[1]:+.4f}°   ({q_sol[1]:+.6f} rad)")
        self._or.setText(f"{q_deg[2]:+.4f}°   ({q_sol[2]:+.6f} rad)")
        self._oit.setText(str(iters))
        self._oer.setText(f"{err:.2e} m")
        self._omode.setText("BB1 + safeguard" if mode == 'bb' else "Alpha fijo")
        if alpha_hist:
            self._oalpha.setText(
                f"final={alpha_hist[-1]:.6f}   media={float(np.mean(alpha_hist)):.6f}"
            )
        else:
            self._oalpha.setText("—")
        if ok:
            self._ostat.setText("✔  CONVERGIÓ")
            self._ostat.setStyleSheet("color:#5ec46e; font-size:13px; font-weight:bold;")
        else:
            self._ostat.setText("✘  NO convergió  (máx. iteraciones alcanzado)")
            self._ostat.setStyleSheet("color:#e05c5c; font-size:13px; font-weight:bold;")
        xyz_rec = fk_geom(*q_sol)
        err_n = float(np.linalg.norm(xd - xyz_rec))
        self._vx.setText(f"{xyz_rec[0]:+.6f}")
        self._vy.setText(f"{xyz_rec[1]:+.6f}")
        self._vz.setText(f"{xyz_rec[2]:+.6f}")
        self._ve.setText(f"{err_n:.2e} m")
        _, T = fk_chain(*q_sol)
        self._ik_mat.setPlainText(f"T0→efector:\n{fmt4(T)}")
        self._jnum_txt.setPlainText(
            "Jacobiano final (NUMÉRICO):\n"
            f"h = {h_diff:.2e} rad\n\n"
            f"{fmt4(J_final)}\n\n"
            "Cada columna se obtuvo por diferencias finitas centrales\n"
            "perturbando solo una articulación."
        )
        # Mostrar log (últimas 35 iteraciones)
        log_lines: List[str] = []
        disp = log[-35:]
        if len(log) > 35:
            log_lines.append(f"(mostrando últimas 35 de {len(log)} iteraciones)")
        for dct in disp:
            log_lines.append(
                f"k={dct['k']:4d}  err={dct['err']:.3e}  g={dct['g']:.3e}"
                f"  α={dct['alpha']:.4e}  src={dct['alpha_source']}"
                f"  ||d||={dct['dir_norm']:.3e}  ||Δq||={dct['step_norm']:.3e}"
                f"  sg={dct['safeguard_tries']:2d}"
                f"  α_deg={dct['q_deg'][0]:+.1f}°"
                f"  β_deg={dct['q_deg'][1]:+.1f}°"
                f"  λ_deg={dct['q_deg'][2]:+.1f}°"
            )
        self._log_txt.setPlainText("\n".join(log_lines))
        # Publicar ángulos a través del nodo
        self.node.set_rad('joint_c', q_sol[0])
        self.node.set_rad('joint_p', q_sol[1])
        self.node.set_rad('joint_r', q_sol[2])
        if not ok:
            QtWidgets.QMessageBox.warning(
                self,
                "Sin convergencia",
                f"El gradiente no convergió en {max_iter} iteraciones.\n"
                f"Error actual: {err:.2e} m\n\n"
                "Sugerencias:\n"
                "  • Aumente el número de iteraciones\n"
                "  • Pruebe otro q₀\n"
                "  • Ajuste α fijo o cambie a modo BB1\n"
                "  • Revise h del Jacobiano numérico",
            )

    # ── Lógica de FK ───────────────────────────────────────────
    def _refresh_fk(self) -> None:
        qc = self.node.get_rad('joint_c')
        qp = self.node.get_rad('joint_p')
        qr = self.node.get_rad('joint_r')
        xyz = fk_geom(qc, qp, qr)
        self._fk_lx.setText(f"{xyz[0]:+.6f}")
        self._fk_ly.setText(f"{xyz[1]:+.6f}")
        self._fk_lz.setText(f"{xyz[2]:+.6f}")
        self._fk_frames, _ = fk_chain(qc, qp, qr)
        cur = max(self._fk_lst.currentRow(), 0)
        self._fk_lst.blockSignals(True)
        self._fk_lst.clear()
        for name, _ in self._fk_frames:
            self._fk_lst.addItem(name)
        self._fk_lst.blockSignals(False)
        self._fk_lst.setCurrentRow(min(cur, len(self._fk_frames) - 1))
        self._show_fk_mat()

    def _show_fk_mat(self) -> None:
        if not hasattr(self, '_fk_frames'):
            return
        idx = self._fk_lst.currentRow()
        if idx < 0 or idx >= len(self._fk_frames):
            return
        name, T = self._fk_frames[idx]
        p = T[:3, 3]
        roll, pitch, yaw = rot_to_rpy(T[:3, :3])
        self._fk_txt.setPlainText(
            f"{name}\n{'─' * 52}\n"
            f"xyz  [m]   = ({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f})\n"
            f"rpy  [rad] = ({roll:+.6f}, {pitch:+.6f}, {yaw:+.6f})\n"
            f"rpy  [°]   = ({math.degrees(roll):+.4f}, "
            f"{math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\n"
            f"T (4×4):\n{fmt4(T)}"
        )

    def _do_zero(self) -> None:
        for j in JOINTS:
            self.node.set_deg(j, 0.0)
            self._sl[j].blockSignals(True)
            self._sl[j].setValue(0)
            self._sl[j].blockSignals(False)
            self._ed[j].setText("0.0")
        self._refresh_fk()

    def _do_home(self) -> None:
        for j in JOINTS:
            d_deg = HOME_DEG[j]
            self.node.set_deg(j, d_deg)
            sl = self._sl[j]
            sl.blockSignals(True)
            sl.setValue(int(d_deg * 10))
            sl.blockSignals(False)
            self._ed[j].setText(f"{d_deg:.1f}")
        self._refresh_fk()


# ─────────────────── Punto de entrada ───────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    node = GradientIKNode()
    app = QtWidgets.QApplication([])
    win = MainWindow(node)
    win.show()
    spin_t = QtCore.QTimer()
    spin_t.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    spin_t.start(10)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    app.exec_()
    try:
        node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()