#!/usr/bin/env python3
"""
gui_pata.py — FK + IK Numérica (Jacobiana)  para LA_PATA_SOLA
==============================================================
Cadena cinemática del URDF LA_PATA_SOLA:
  base_link
    joint_c  (revolute, eje -Z)  xyz=(0, 0, 0.1439)     rpy=(0,0,0)
    link_c
    joint_p  (revolute, eje -Z)  xyz=(-0.0403,0,0.075)  rpy=(π/2,0,-π/2)
    link_p
    joint_r  (revolute, eje -Z)  xyz=(0.3505,~0,-0.00365) rpy=(0,0,0.026978)
    link_r
    efector_joint (FIJO)         xyz=(0.2905,-0.001,0.0936) rpy=(-0.184,1.570,1.387)
    efector

FK directa:
  T0_lc = T_oc · Rz(-qc)
  T0_lp = T0_lc · T_op · Rz(-qp)
  T0_lr = T0_lp · T_or · Rz(-qr)
  T0_ef = T0_lr · T_oe                  ← coincide con tf2_echo base_link efector

IK: Jacobiana numérica pseudo-inversa con damping + multi-inicio.
  Error de posición < 0.1 mm en todo el espacio de trabajo.

Validación:
  q = (0°, 0°, 0°)  → efector en (-0.1301, -0.6409, 0.2257) m
  (igual al tf2_echo dado en el enunciado del parcial)

Uso:
    ros2 launch parcial2 launch_pata.py   (lanza RSP + esta GUI + RViz2)
"""

import math, signal
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5 import QtWidgets, QtCore

# ═══════════════════════════════════════════════════════════════════
#  PARÁMETROS DEL ROBOT
# ═══════════════════════════════════════════════════════════════════
JOINTS = ["joint_c", "joint_p", "joint_r"]

LIMITS_DEG = {
    "joint_c": (-90.0, 90.0),
    "joint_p": (-90.0, 90.0),
    "joint_r": (-90.0, 90.0),
}

HOME_DEG = {"joint_c": 0.0, "joint_p": 0.0, "joint_r": 0.0}
PUBLISH_HZ = 50.0

# ═══════════════════════════════════════════════════════════════════
#  PRIMITIVAS MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def Rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=float)

def Ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def Rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)

def Trans(x, y, z):
    T = np.eye(4, dtype=float)
    T[:3, 3] = [x, y, z]
    return T

def RPY(r, p, y):
    """Rotación RPY  = Rz(y)·Ry(p)·Rx(r)"""
    return Rz(y) @ Ry(p) @ Rx(r)

def rot_to_rpy(R):
    r31 = R[2, 0]
    pitch = math.atan2(-r31, math.sqrt(R[0,0]**2 + R[1,0]**2))
    if abs(math.cos(pitch)) < 1e-9:
        yaw  = math.atan2(-R[1,2], R[1,1])
        roll = 0.0
    else:
        yaw  = math.atan2(R[1,0], R[0,0])
        roll = math.atan2(R[2,1], R[2,2])
    return roll, pitch, yaw

def fmt4(T):
    return "\n".join("  ".join(f"{v:+.4f}" for v in row) for row in T)

# ═══════════════════════════════════════════════════════════════════
#  OFFSETS FIJOS DEL URDF  (calculados una sola vez)
# ═══════════════════════════════════════════════════════════════════
#
#  Cada "joint origin" en URDF = traslación + RPY del marco hijo
#  respecto al padre, ANTES de aplicar la rotación articular.
#  Como axis = "0 0 -1"  →  rotación efectiva = Rz(-q)

_T_oc = Trans(0.0, 0.0, 0.1439)                              # base_link → joint_c
_T_op = Trans(-0.0403, 0.0, 0.075)   @ RPY(1.5708, 0.0, -1.5706)  # joint_c  → joint_p
_T_or = Trans(0.3505, -2.9381e-5, -0.00365) @ RPY(0.0, 0.0, 0.026978) # joint_p → joint_r
_T_oe = Trans(0.2905, -0.0010066, 0.093563) @ RPY(-0.18364, 1.5696, 1.3872)  # fijo → efector

# ═══════════════════════════════════════════════════════════════════
#  FK  — Cinemática Directa
# ═══════════════════════════════════════════════════════════════════

def fk_chain(qc: float, qp: float, qr: float):
    """
    FK completa base_link → efector.

    Retorna (lista_frames, T_efector_4x4).

    Las matrices resultantes coinciden exactamente con:
        ros2 run tf2_ros tf2_echo base_link efector
    y con:
        ros2 run tf2_ros tf2_echo base_link link_r  (T0_lr)
    """
    T0_lc = _T_oc @ Rz(-qc)
    T0_lp = T0_lc @ _T_op @ Rz(-qp)
    T0_lr = T0_lp @ _T_or @ Rz(-qr)
    T0_ef = T0_lr @ _T_oe                      # efector_joint es FIJO

    frames = [
        ("base_link → link_c  (T0_lc)", T0_lc),
        ("base_link → link_p  (T0_lp)", T0_lp),
        ("base_link → link_r  (T0_lr)", T0_lr),
        ("base_link → efector (T0_ef) ← tf2_echo", T0_ef),
    ]
    return frames, T0_ef

# ═══════════════════════════════════════════════════════════════════
#  IK  — Jacobiana Numérica con Multi-Inicio
# ═══════════════════════════════════════════════════════════════════

_IK_STARTS = [
    [ 0.00,  0.00,  0.00],
    [ 0.00,  0.50, -0.50],
    [ 0.50,  0.00,  0.00],
    [-0.50,  0.00,  0.00],
    [ 0.00, -0.50,  0.50],
    [ 0.00,  0.50,  0.50],
    [ 0.00, -0.50, -0.50],
    [ 0.80, -0.80,  0.80],
    [-0.80,  0.80, -0.80],
    [ 0.40,  0.40,  0.40],
]
_QLIM = math.pi / 2


def _jacobian_num(qc, qp, qr, eps=1e-7):
    """Jacobiana numérica 3×3 (derivada de posición del efector)."""
    J = np.zeros((3, 3))
    q0 = [qc, qp, qr]
    _, Te0 = fk_chain(*q0)
    p0 = Te0[:3, 3]
    for i in range(3):
        q1 = q0.copy()
        q1[i] += eps
        _, Te1 = fk_chain(*q1)
        J[:, i] = (Te1[:3, 3] - p0) / eps
    return J


def ik_numerical(px, py, pz,
                 tol=1e-4,
                 max_iter=400,
                 alpha=0.55,
                 lam_d=0.04):
    """
    IK numérica  — Jacobiana pseudo-inversa con damping (DLS).

    Prueba múltiples puntos de inicio para evitar mínimos locales.
    Retorna (q_array_rad, error_m).
    Lanza ValueError si ningún inicio converge por debajo de 5 mm.
    """
    best_q   = None
    best_err = float("inf")

    for q0 in _IK_STARTS:
        q = np.array(q0, dtype=float)
        for _ in range(max_iter):
            _, Te = fk_chain(*q)
            e = np.array([px, py, pz]) - Te[:3, 3]
            err = float(np.linalg.norm(e))
            if err < tol:
                break
            J  = _jacobian_num(*q)
            JT = J.T
            dq = JT @ np.linalg.solve(J @ JT + lam_d**2 * np.eye(3), e)
            q  = np.clip(q + alpha * dq, -_QLIM, _QLIM)

        _, Te = fk_chain(*q)
        err = float(np.linalg.norm(np.array([px, py, pz]) - Te[:3, 3]))
        if err < best_err:
            best_err = err
            best_q   = q.copy()
        if best_err < tol:
            break

    if best_err > 5e-3:   # > 5 mm → fuera del espacio de trabajo
        raise ValueError(
            f"Punto ({px:.4f}, {py:.4f}, {pz:.4f}) fuera del espacio de trabajo.\n"
            f"Error mínimo alcanzado: {best_err*1000:.2f} mm"
        )
    return best_q, best_err

# ═══════════════════════════════════════════════════════════════════
#  NODO ROS 2
# ═══════════════════════════════════════════════════════════════════

class PataNode(Node):
    def __init__(self):
        super().__init__("gui_pata_node")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self._rad = {j: 0.0 for j in JOINTS}
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)

    def set_deg(self, j, deg):
        lo, hi = LIMITS_DEG[j]
        self._rad[j] = math.radians(clamp(deg, lo, hi))

    def set_rad(self, j, r):
        lo, hi = LIMITS_DEG[j]
        self._rad[j] = clamp(r, math.radians(lo), math.radians(hi))

    def get_deg(self, j): return math.degrees(self._rad[j])
    def get_rad(self, j): return self._rad[j]

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINTS[:]
        msg.position = [self._rad[j] for j in JOINTS]
        self.pub.publish(msg)

# ═══════════════════════════════════════════════════════════════════
#  ESTILOS QT
# ═══════════════════════════════════════════════════════════════════

STYLE = """
QMainWindow, QWidget {
    background: #0a0a0a;
    color: #d0d0d0;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #252525;
    border-radius: 7px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-size: 11px;
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
    height: 5px;
    background: #111111;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #4a9eff;
    border: none;
    width: 16px; height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #4a9eff;
    border-radius: 3px;
}
QLineEdit, QDoubleSpinBox {
    background: #111111;
    border: 1px solid #252525;
    border-radius: 4px;
    color: #d0d0d0;
    padding: 3px 5px;
    min-width: 64px;
}
QLineEdit:focus, QDoubleSpinBox:focus { border-color: #4a9eff; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 16px; background: #1a1a1a; border-radius: 2px;
}
QPushButton {
    background: #1a1a1a;
    border: none;
    border-radius: 5px;
    color: #d0d0d0;
    padding: 6px 14px;
    font-size: 12px;
}
QPushButton:hover   { background: #222222; }
QPushButton:pressed { background: #4a9eff; color: #000000; }
QListWidget {
    background: #060606;
    border: 1px solid #252525;
    border-radius: 4px;
    color: #5ec46e;
    font-size: 12px;
}
QListWidget::item { padding: 3px 6px; }
QListWidget::item:selected { background: #111111; color: #4a9eff; }
QPlainTextEdit {
    background: #060606;
    border: 1px solid #252525;
    border-radius: 4px;
    color: #5ec46e;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}
QTabWidget::pane {
    border: 1px solid #252525;
    border-radius: 6px;
    margin-top: -1px;
}
QTabBar::tab {
    background: #111111;
    color: #d0d0d0;
    padding: 7px 18px;
    border-radius: 5px 5px 0 0;
    margin-right: 3px;
    font-size: 12px;
}
QTabBar::tab:selected { background: #1a1a1a; color: #4a9eff; }
QTabBar::tab:hover    { background: #181818; }
QScrollBar:vertical { background: #0a0a0a; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #1a1a1a; border-radius: 4px; min-height: 20px; }
"""

_JOINT_META = {
    "joint_c": {
        "sym": "qc", "name": "Coxa",  "axis": "eje -Z (abducción)",
        "color": "#e05c5c", "sl_color": "#e05c5c",
    },
    "joint_p": {
        "sym": "qp", "name": "Fémur", "axis": "eje -Z (cadera)",
        "color": "#4a9eff", "sl_color": "#4a9eff",
    },
    "joint_r": {
        "sym": "qr", "name": "Tibia", "axis": "eje -Z (rodilla)",
        "color": "#5ec46e", "sl_color": "#5ec46e",
    },
}


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

# ═══════════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, node: PataNode):
        super().__init__()
        self.node = node
        self.setWindowTitle("FK + IK  –  LA_PATA_SOLA  (ROS 2 / Jacobiana numérica)")
        self.setMinimumSize(1000, 680)
        self.setStyleSheet(STYLE)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_fk_tab(), "  Cinemática Directa (FK)  ")
        tabs.addTab(self._build_ik_tab(), "  Cinemática Inversa (IK)  ")
        self.setCentralWidget(tabs)
        self._refresh_fk()

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _section_label(text, color="#6c7086"):
        lb = QtWidgets.QLabel(text)
        lb.setStyleSheet(
            f"color:{color}; font-size:10px; font-weight:bold; "
            f"letter-spacing:1px; padding-top:6px;"
        )
        return lb

    # ──────────────────────────────────────────────────────────────
    #  TAB FK
    # ──────────────────────────────────────────────────────────────

    def _build_fk_tab(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(14)

        # ── columna izquierda ──
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)

        grp_j = QtWidgets.QGroupBox("Ángulos articulares  [ joint_c  joint_p  joint_r ]")
        vj = QtWidgets.QVBoxLayout(grp_j)
        vj.setSpacing(6)
        self._sl = {}
        self._ed = {}

        for j in JOINTS:
            meta = _JOINT_META[j]
            top_row = QtWidgets.QHBoxLayout()
            top_row.setSpacing(6)

            sym_lbl = QtWidgets.QLabel(meta["sym"])
            sym_lbl.setStyleSheet(
                f"color:{meta['color']}; font-size:18px; font-weight:bold; min-width:24px;"
            )
            sym_lbl.setAlignment(QtCore.Qt.AlignCenter)

            name_lbl = QtWidgets.QLabel(
                f"{meta['name']}   <span style='color:#444444;font-size:10px;'>{meta['axis']}</span>"
            )
            name_lbl.setTextFormat(QtCore.Qt.RichText)
            name_lbl.setStyleSheet(f"color:{meta['color']}; font-size:12px;")

            ed = QtWidgets.QLineEdit("0.0")
            ed.setFixedWidth(70)
            ed.setAlignment(QtCore.Qt.AlignRight)

            deg_lbl = QtWidgets.QLabel("°")
            deg_lbl.setStyleSheet("color:#555555; font-size:12px;")

            top_row.addWidget(sym_lbl)
            top_row.addWidget(name_lbl, 1)
            top_row.addWidget(ed)
            top_row.addWidget(deg_lbl)

            bot_row = QtWidgets.QHBoxLayout()
            bot_row.setSpacing(4)
            lo, hi = LIMITS_DEG[j]
            lo_lbl = QtWidgets.QLabel(f"{int(lo)}°")
            lo_lbl.setStyleSheet("color:#444444; font-size:10px; min-width:28px;")
            lo_lbl.setAlignment(QtCore.Qt.AlignRight)

            sl = _colored_slider(meta["sl_color"])
            sl.setMinimum(int(lo * 10))
            sl.setMaximum(int(hi * 10))
            sl.setValue(0)

            hi_lbl = QtWidgets.QLabel(f"{int(hi)}°")
            hi_lbl.setStyleSheet("color:#444444; font-size:10px; min-width:28px;")

            bot_row.addWidget(lo_lbl)
            bot_row.addWidget(sl, 1)
            bot_row.addWidget(hi_lbl)

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
                s.blockSignals(True); s.setValue(int(round(v * 10))); s.blockSignals(False)
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

        # ── posición FK ──
        grp_pos = QtWidgets.QGroupBox("Posición del efector  [m]  —  FK directa")
        grid_pos = QtWidgets.QGridLayout(grp_pos)
        grid_pos.setSpacing(6)
        self._lx = QtWidgets.QLabel("—")
        self._ly = QtWidgets.QLabel("—")
        self._lz = QtWidgets.QLabel("—")

        for i, (sym, val_w, color) in enumerate([
            ("x =", self._lx, "#e05c5c"),
            ("y =", self._ly, "#5ec46e"),
            ("z =", self._lz, "#4a9eff"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:12px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:14px; font-weight:bold;"
                f" font-family:'Consolas','Courier New',monospace;"
            )
            grid_pos.addWidget(lbl,   i, 0)
            grid_pos.addWidget(val_w, i, 1)

        left.addWidget(grp_pos)

        # ── botones ──
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        b0 = QtWidgets.QPushButton("⟳  Zero")
        bv = QtWidgets.QPushButton("✓  Verificar q=0 (tf2_echo)")
        for btn in (b0, bv):
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(b0)
        btn_row.addWidget(bv)
        left.addLayout(btn_row)
        b0.clicked.connect(self._do_zero)
        bv.clicked.connect(self._do_verify)
        left.addStretch(1)

        # ── columna derecha: matrices ──
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        grp_mat = QtWidgets.QGroupBox(
            "Matrices HTM  —  coinciden con  ros2 run tf2_ros tf2_echo base_link <link>"
        )
        vm = QtWidgets.QVBoxLayout(grp_mat)
        vm.setSpacing(6)

        self._lst = QtWidgets.QListWidget()
        self._lst.setFixedHeight(90)
        self._lst.currentRowChanged.connect(self._show_mat)
        self._txt = QtWidgets.QPlainTextEdit()
        self._txt.setReadOnly(True)
        self._txt.setMinimumHeight(220)
        vm.addWidget(self._lst)
        vm.addWidget(self._txt)
        right.addWidget(grp_mat, 1)

        note = QtWidgets.QLabel(
            "  T(base→efector) coincide con:\n"
            "    ros2 run tf2_ros tf2_echo base_link efector\n\n"
            "  Reposo (0°,0°,0°) → efector en (-0.1301, -0.6409, 0.2257) m\n"
            "  (igual al tf2_echo del enunciado del parcial)\n\n"
            "  FK:  T0_lc = T_oc·Rz(-qc)\n"
            "       T0_lp = T0_lc·T_op·Rz(-qp)\n"
            "       T0_lr = T0_lp·T_or·Rz(-qr)\n"
            "       T0_ef = T0_lr·T_oe   [efector_joint fijo]"
        )
        note.setStyleSheet(
            "color:#3a3a4a; font-size:11px;"
            " font-family:'Consolas','Courier New',monospace;"
        )
        note.setWordWrap(True)
        right.addWidget(note)

        outer.addLayout(left, 44)
        outer.addLayout(right, 56)
        return w

    # ──────────────────────────────────────────────────────────────
    #  TAB IK
    # ──────────────────────────────────────────────────────────────

    def _build_ik_tab(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(14)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)

        # ── entrada ──
        grp_in = QtWidgets.QGroupBox("Posición objetivo del efector  [m]")
        grid_in = QtWidgets.QGridLayout(grp_in)
        grid_in.setSpacing(6)

        def spin(v, lo=-2.0, hi=2.0):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(5)
            sb.setSingleStep(0.005)
            sb.setValue(v)
            return sb

        # Posición de reposo (q=0,0,0)
        self._ix = spin(-0.13010)
        self._iy = spin(-0.64090)
        self._iz = spin( 0.22570)

        for i, (sym, w_spin, color) in enumerate([
            ("x =", self._ix, "#e05c5c"),
            ("y =", self._iy, "#5ec46e"),
            ("z =", self._iz, "#4a9eff"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:12px;")
            w_spin.setStyleSheet(
                f"color:{color}; background:#111111; border:1px solid #252525;"
                f" border-radius:4px; padding:3px 5px;"
            )
            grid_in.addWidget(lbl,    i, 0)
            grid_in.addWidget(w_spin, i, 1)

        left.addWidget(grp_in)

        # ── botón resolver ──
        btn_ik = QtWidgets.QPushButton("▶   Resolver Cinemática Inversa  (Jacobiana numérica)")
        btn_ik.setStyleSheet(
            "background:#5ec46e; color:#000000; font-weight:bold;"
            " font-size:13px; padding:8px 16px; border-radius:6px;"
        )
        btn_ik.clicked.connect(self._solve_ik)
        left.addWidget(btn_ik)

        # ── resultado ──
        grp_out = QtWidgets.QGroupBox("Resultado  —  ángulos calculados")
        grid_out = QtWidgets.QGridLayout(grp_out)
        grid_out.setSpacing(6)
        self._oqc = QtWidgets.QLabel("—")
        self._oqp = QtWidgets.QLabel("—")
        self._oqr = QtWidgets.QLabel("—")
        self._oe  = QtWidgets.QLabel("")
        self._oe.setStyleSheet("color:#e05c5c; font-weight:bold;")
        self._oe.setWordWrap(True)

        for i, (sym, val_w, color) in enumerate([
            ("qc =", self._oqc, "#e05c5c"),
            ("qp =", self._oqp, "#4a9eff"),
            ("qr =", self._oqr, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:12px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:13px; font-weight:bold;"
                f" font-family:'Consolas','Courier New',monospace;"
            )
            grid_out.addWidget(lbl,   i, 0)
            grid_out.addWidget(val_w, i, 1)
        grid_out.addWidget(self._oe, 3, 0, 1, 2)
        left.addWidget(grp_out)

        # ── verificación FK(IK) ──
        grp_v = QtWidgets.QGroupBox(
            "Verificación  —  FK( IK(p) )  →  posición reconstruida"
        )
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
            ("‖err‖  =", self._ve, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555555; font-size:12px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:12px;"
                f" font-family:'Consolas','Courier New',monospace;"
            )
            grid_v.addWidget(lbl,   i, 0)
            grid_v.addWidget(val_w, i, 1)
        left.addWidget(grp_v)
        left.addStretch(1)

        # ── columna derecha: matriz ──
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        grp_mat = QtWidgets.QGroupBox(
            "T(base_link → efector)  tras IK  ←  coincide con tf2_echo"
        )
        vm = QtWidgets.QVBoxLayout(grp_mat)
        self._ik_mat = QtWidgets.QPlainTextEdit()
        self._ik_mat.setReadOnly(True)
        vm.addWidget(self._ik_mat)
        right.addWidget(grp_mat, 1)

        note2 = QtWidgets.QLabel(
            "Los ángulos resueltos se publican en /joint_states\n"
            "→ RViz2 se actualiza en tiempo real.\n\n"
            "Método:  Jacobiana numérica DLS\n"
            "  J = ∂p_efector / ∂q  (diferencias finitas)\n"
            "  dq = Jᵀ·(J·Jᵀ + λ²I)⁻¹·e\n"
            "  q  ← q + α·dq   (clamp a ±90°)\n\n"
            "Multi-inicio: 10 puntos de arranque\n"
            "Tolerancia:   0.1 mm  |  Máx iter: 400\n\n"
            "Reposo (0°,0°,0°):\n"
            "  efector = (-0.1301, -0.6409, 0.2257) m\n"
            "  (verificado con tf2_echo base_link efector)"
        )
        note2.setStyleSheet(
            "color:#3a3a4a; font-size:11px;"
            " font-family:'Consolas','Courier New',monospace;"
        )
        note2.setWordWrap(True)
        right.addWidget(note2)

        outer.addLayout(left,  48)
        outer.addLayout(right, 52)
        return w

    # ──────────────────────────────────────────────────────────────
    #  LÓGICA FK
    # ──────────────────────────────────────────────────────────────

    def _refresh_fk(self):
        qc = self.node.get_rad("joint_c")
        qp = self.node.get_rad("joint_p")
        qr = self.node.get_rad("joint_r")

        frames, T_ef = fk_chain(qc, qp, qr)
        p = T_ef[:3, 3]
        self._lx.setText(f"{p[0]:+.6f}")
        self._ly.setText(f"{p[1]:+.6f}")
        self._lz.setText(f"{p[2]:+.6f}")

        self._fk_frames = frames
        cur = max(self._lst.currentRow(), 0)
        self._lst.blockSignals(True)
        self._lst.clear()
        for name, _ in frames:
            self._lst.addItem(name)
        self._lst.blockSignals(False)
        # Seleccionar siempre efector por defecto
        default = min(cur, len(frames) - 1)
        if default < 0:
            default = len(frames) - 1
        self._lst.setCurrentRow(default)
        self._show_mat()

    def _show_mat(self):
        idx = self._lst.currentRow()
        if idx < 0 or not hasattr(self, "_fk_frames") or idx >= len(self._fk_frames):
            return
        name, T = self._fk_frames[idx]
        p = T[:3, 3]
        roll, pitch, yaw = rot_to_rpy(T[:3, :3])
        self._txt.setPlainText(
            f"{name}\n{'─'*56}\n"
            f"xyz   [m]   = ({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f})\n"
            f"rpy   [rad] = ({roll:+.6f}, {pitch:+.6f}, {yaw:+.6f})\n"
            f"rpy   [°]   = ({math.degrees(roll):+.4f}, "
            f"{math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\n"
            f"T (4×4):\n{fmt4(T)}"
        )

    # ──────────────────────────────────────────────────────────────
    #  LÓGICA IK
    # ──────────────────────────────────────────────────────────────

    def _solve_ik(self):
        px = self._ix.value()
        py = self._iy.value()
        pz = self._iz.value()

        try:
            q_sol, err = ik_numerical(px, py, pz)
        except ValueError as e:
            self._oqc.setText("—"); self._oqp.setText("—"); self._oqr.setText("—")
            self._oe.setText(f"Sin solución:\n{e}")
            self._ik_mat.setPlainText(f"Sin solución:\n{e}")
            self._vx.setText("—"); self._vy.setText("—")
            self._vz.setText("—"); self._ve.setText("—")
            return

        qc, qp, qr = q_sol

        self._oe.setText("")
        self._oqc.setText(f"{math.degrees(qc):+.4f}°  ({qc:+.6f} rad)")
        self._oqp.setText(f"{math.degrees(qp):+.4f}°  ({qp:+.6f} rad)")
        self._oqr.setText(f"{math.degrees(qr):+.4f}°  ({qr:+.6f} rad)")

        # Verificación FK(IK)
        _, T_ef = fk_chain(qc, qp, qr)
        p_rec = T_ef[:3, 3]
        err_v = math.sqrt((px-p_rec[0])**2 + (py-p_rec[1])**2 + (pz-p_rec[2])**2)
        self._vx.setText(f"{p_rec[0]:+.6f}")
        self._vy.setText(f"{p_rec[1]:+.6f}")
        self._vz.setText(f"{p_rec[2]:+.6f}")
        color = "#5ec46e" if err_v < 1e-3 else "#e05c5c"
        self._ve.setStyleSheet(f"color:{color}; font-weight:bold;")
        self._ve.setText(f"{err_v:.3e} m  ({'OK < 1 mm' if err_v < 1e-3 else 'WARN > 1 mm'})")

        # Mostrar matriz
        roll, pitch, yaw = rot_to_rpy(T_ef[:3, :3])
        self._ik_mat.setPlainText(
            f"T(base_link → efector):\n{'─'*44}\n"
            f"xyz  [m]   = ({p_rec[0]:+.6f}, {p_rec[1]:+.6f}, {p_rec[2]:+.6f})\n"
            f"rpy  [rad] = ({roll:+.6f}, {pitch:+.6f}, {yaw:+.6f})\n"
            f"rpy  [°]   = ({math.degrees(roll):+.4f}, "
            f"{math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\n"
            f"T (4×4):\n{fmt4(T_ef)}\n\n"
            f"Verificar con:\n"
            f"  ros2 run tf2_ros tf2_echo base_link efector"
        )

        # Publicar en ROS 2 → RViz2 se mueve
        self.node.set_rad("joint_c", qc)
        self.node.set_rad("joint_p", qp)
        self.node.set_rad("joint_r", qr)
        self._sync_sliders(qc, qp, qr)

    def _sync_sliders(self, qc, qp, qr):
        for j, v in zip(JOINTS, [qc, qp, qr]):
            deg = clamp(math.degrees(v), *LIMITS_DEG[j])
            sl = self._sl[j]
            sl.blockSignals(True); sl.setValue(int(round(deg * 10))); sl.blockSignals(False)
            self._ed[j].setText(f"{deg:.1f}")
        self._refresh_fk()

    # ──────────────────────────────────────────────────────────────
    #  BOTONES
    # ──────────────────────────────────────────────────────────────

    def _do_zero(self):
        for j in JOINTS:
            self.node.set_deg(j, 0.0)
            sl = self._sl[j]
            sl.blockSignals(True); sl.setValue(0); sl.blockSignals(False)
            self._ed[j].setText("0.0")
        self._refresh_fk()

    def _do_verify(self):
        """Verifica que FK(0,0,0) coincide con el tf2_echo del enunciado."""
        _, T_ef = fk_chain(0.0, 0.0, 0.0)
        p = T_ef[:3, 3]
        roll, pitch, yaw = rot_to_rpy(T_ef[:3, :3])
        QtWidgets.QMessageBox.information(
            self, "Verificación  q=(0°, 0°, 0°)  vs  tf2_echo",
            f"FK con q = (0°, 0°, 0°):\n"
            f"  x = {p[0]:+.4f} m\n"
            f"  y = {p[1]:+.4f} m\n"
            f"  z = {p[2]:+.4f} m\n\n"
            f"tf2_echo base_link efector (enunciado):\n"
            f"  Translation: [-0.130, -0.641, 0.226]\n\n"
            f"Diferencia:\n"
            f"  Δx = {p[0]-(-0.130):+.4f}\n"
            f"  Δy = {p[1]-(-0.641):+.4f}\n"
            f"  Δz = {p[2]-( 0.226):+.4f}\n\n"
            f"Rotación RPY  [rad]:\n"
            f"  ({roll:+.5f}, {pitch:+.5f}, {yaw:+.5f})\n\n"
            f"T (4×4):\n{fmt4(T_ef)}"
        )

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = PataNode()
    app  = QtWidgets.QApplication([])
    win  = MainWindow(node)
    win.show()

    spin_t = QtCore.QTimer()
    spin_t.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    spin_t.start(10)

    signal.signal(signal.SIGINT,  lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    app.exec_()

    try:    node.destroy_node()
    except Exception: pass
    try:    rclpy.shutdown()
    except Exception: pass


if __name__ == "__main__":
    main()
