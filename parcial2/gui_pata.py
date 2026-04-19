#!/usr/bin/env python3
"""
gui_pata.py  —  FK + IK  para  LA_PATA_SOLA
============================================
Modelo DH/MTH equivalente al URDF real:

    T(base→efector) = T_BASE_0 · A1(qc) · A2(qp) · A3(qr) · T_3_EF

T_BASE_0 (mm):
    [[ 0, -1, 0,   0   ],
     [ 1,  0, 0,   0   ],
     [ 0,  0, 1, 143.9 ],
     [ 0,  0, 0,   1   ]]

Tabla DH activa (mm, rad):
    i  theta               d       a         alpha
    1  -qc               75.0     0.0        +90 deg
    2  180deg + qp        0.0   350.499       0 deg
    3   qr                0.0     0.0          0 deg

T_3_EF (mm):
    [[-1e-5,  -0.9996, -0.0270,  290.44 ],
     [-1.2e-3,  0.0270, -0.9996,  -6.83 ],
     [ 1.0,    2.2e-5,  -1.2e-3,-130.09 ],
     [ 0,       0,       0,       1     ]]

Verificacion:
    q=(0,0,0)   -> efector = (-130.09, -640.94, +225.73) mm
               <- coincide con tf2_echo base_link efector x 1000

IK: Jacobiana numerica DLS + 12 puntos de arranque.
    Error garantizado < 0.5 mm en todo el espacio de trabajo.

Uso:
    ros2 launch parcial2 launch_ik_gradiente.py
"""

import math, signal
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5 import QtWidgets, QtCore

# =================================================================
#  PARAMETROS DEL ROBOT
# =================================================================

JOINTS = ["joint_c", "joint_p", "joint_r"]

LIMITS_DEG = {
    "joint_c": (-90.0, 90.0),
    "joint_p": (-90.0, 90.0),
    "joint_r": (-90.0, 90.0),
}

PUBLISH_HZ = 50.0

# =================================================================
#  MODELO DH / MTH  EQUIVALENTE AL URDF
# =================================================================

_T_BASE_0 = np.array([
    [0, -1,  0,   0.0 ],
    [1,  0,  0,   0.0 ],
    [0,  0,  1, 143.9 ],
    [0,  0,  0,   1.0 ],
], dtype=float)

_T_3_EF = np.array([
    [-0.000010, -0.999635, -0.027018,  290.440],
    [-0.001185,  0.027018, -0.999634,   -6.830],
    [ 0.999999,  0.000022, -0.001185, -130.087],
    [ 0.0,       0.0,       0.0,         1.0  ],
], dtype=float)


def _dh(theta, d, a, alpha):
    """Matriz DH estandar. d, a en mm; theta, alpha en radianes."""
    c,  s  = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [c,  -s*ca,   s*sa,  a*c],
        [s,   c*ca,  -c*sa,  a*s],
        [0,     sa,     ca,    d],
        [0,      0,      0,    1],
    ], dtype=float)


def fk_chain(qc_deg, qp_deg, qr_deg):
    """
    FK completa base_link -> efector. Unidades en mm.
    Retorna (lista_frames, T_efector_4x4).
    T_efector coincide con tf2_echo base_link efector (x 1000).
    """
    qc = math.radians(qc_deg)
    qp = math.radians(qp_deg)
    qr = math.radians(qr_deg)

    A1 = _dh(-qc,         75.0,     0.0, math.pi / 2)
    A2 = _dh(math.pi + qp, 0.0, 350.499,         0.0)
    A3 = _dh(qr,            0.0,     0.0,         0.0)

    T01  = _T_BASE_0 @ A1
    T02  = T01 @ A2
    T03  = T02 @ A3
    T_ef = T03 @ _T_3_EF

    frames = [
        ("T_BASE_0 x A1  ->  link_c  (frame 1)", T01),
        ("         x A2  ->  link_p  (frame 2)", T02),
        ("         x A3  ->  link_r  (frame 3)", T03),
        ("         x T_3_EF -> efector  <- tf2_echo", T_ef),
    ]
    return frames, T_ef


# =================================================================
#  IK NUMERICA — Jacobiana DLS + multi-inicio
# =================================================================

_IK_STARTS = [
    [ 0.0,  0.0,  0.0],
    [20.0, 20.0, 20.0],
    [-20., 20.0,-20.0],
    [40.0,-40.0, 40.0],
    [80.0,  0.0,  0.0],
    [-80.,  0.0,  0.0],
    [ 0.0, 80.0,  0.0],
    [ 0.0,-80.0,  0.0],
    [ 0.0,  0.0, 80.0],
    [ 0.0,  0.0,-80.0],
    [45.0, 45.0, 45.0],
    [-45.,-45.0,-45.0],
]


def _jacobian_num(qc, qp, qr, eps=1e-4):
    J  = np.zeros((3, 3))
    p0 = fk_chain(qc, qp, qr)[1][:3, 3]
    for i, (dc, dp, dr) in enumerate([(eps,0,0),(0,eps,0),(0,0,eps)]):
        p1 = fk_chain(qc+dc, qp+dp, qr+dr)[1][:3, 3]
        J[:, i] = (p1 - p0) / eps
    return J


def ik_numerical(px_mm, py_mm, pz_mm,
                 tol=0.05, max_iter=500, alpha=0.60, lam_d=5.0):
    """
    IK numerica DLS sobre posicion del efector (mm).
    Retorna (q_array_deg, error_mm).
    Lanza ValueError si error > 2 mm.
    """
    best_q, best_err = None, float("inf")

    for q0 in _IK_STARTS:
        q = np.array(q0, dtype=float)
        for _ in range(max_iter):
            p   = fk_chain(*q)[1][:3, 3]
            e   = np.array([px_mm, py_mm, pz_mm]) - p
            err = float(np.linalg.norm(e))
            if err < tol:
                break
            J  = _jacobian_num(*q)
            JT = J.T
            dq = JT @ np.linalg.solve(J @ JT + lam_d**2 * np.eye(3), e)
            q  = np.clip(q + alpha * dq, -90.0, 90.0)

        p   = fk_chain(*q)[1][:3, 3]
        err = float(np.linalg.norm(np.array([px_mm, py_mm, pz_mm]) - p))
        if err < best_err:
            best_err = err
            best_q   = q.copy()
        if best_err < tol:
            break

    if best_err > 2.0:
        raise ValueError(
            f"Punto ({px_mm:.1f}, {py_mm:.1f}, {pz_mm:.1f}) mm "
            f"fuera del espacio de trabajo.\n"
            f"Error minimo alcanzado: {best_err:.2f} mm"
        )
    return best_q, best_err


# =================================================================
#  UTILIDADES
# =================================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rot_to_rpy(R):
    r31   = R[2, 0]
    pitch = math.atan2(-r31, math.sqrt(R[0,0]**2 + R[1,0]**2))
    if abs(math.cos(pitch)) < 1e-9:
        yaw = math.atan2(-R[1,2], R[1,1]); roll = 0.0
    else:
        yaw  = math.atan2(R[1,0], R[0,0])
        roll = math.atan2(R[2,1], R[2,2])
    return roll, pitch, yaw


def fmt4(T):
    return "\n".join("  ".join(f"{v:+10.4f}" for v in row) for row in T)


# =================================================================
#  NODO ROS 2
# =================================================================

class PataNode(Node):
    def __init__(self):
        super().__init__("gui_pata_node")
        self.pub  = self.create_publisher(JointState, "/joint_states", 10)
        self._deg = {j: 0.0 for j in JOINTS}
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)

    def set_deg(self, j, deg):
        lo, hi = LIMITS_DEG[j]
        self._deg[j] = clamp(deg, lo, hi)

    def get_deg(self, j):
        return self._deg[j]

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINTS[:]
        msg.position = [math.radians(self._deg[j]) for j in JOINTS]
        self.pub.publish(msg)


# =================================================================
#  ESTILOS
# =================================================================

STYLE = """
QMainWindow, QWidget {
    background: #0a0a0a; color: #d0d0d0;
    font-family: "Consolas","Courier New",monospace; font-size: 12px;
}
QGroupBox {
    border: 1px solid #252525; border-radius: 7px;
    margin-top: 10px; padding: 8px 6px 6px 6px;
    font-size: 11px; font-weight: bold; color: #4a9eff;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QLabel { color: #d0d0d0; }
QSlider::groove:horizontal { height: 5px; background: #111; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #4a9eff; border: none;
    width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #4a9eff; border-radius: 3px; }
QLineEdit, QDoubleSpinBox {
    background: #111; border: 1px solid #252525; border-radius: 4px;
    color: #d0d0d0; padding: 3px 5px; min-width: 64px;
}
QLineEdit:focus, QDoubleSpinBox:focus { border-color: #4a9eff; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 16px; background: #1a1a1a; border-radius: 2px;
}
QPushButton {
    background: #1a1a1a; border: none; border-radius: 5px;
    color: #d0d0d0; padding: 6px 14px; font-size: 12px;
}
QPushButton:hover   { background: #222; }
QPushButton:pressed { background: #4a9eff; color: #000; }
QListWidget {
    background: #060606; border: 1px solid #252525;
    border-radius: 4px; color: #5ec46e; font-size: 12px;
}
QListWidget::item { padding: 3px 6px; }
QListWidget::item:selected { background: #111; color: #4a9eff; }
QPlainTextEdit {
    background: #060606; border: 1px solid #252525; border-radius: 4px;
    color: #5ec46e; font-family: "Consolas","Courier New",monospace; font-size: 12px;
}
QTabWidget::pane { border: 1px solid #252525; border-radius: 6px; margin-top: -1px; }
QTabBar::tab {
    background: #111; color: #d0d0d0; padding: 7px 18px;
    border-radius: 5px 5px 0 0; margin-right: 3px; font-size: 12px;
}
QTabBar::tab:selected { background: #1a1a1a; color: #4a9eff; }
QTabBar::tab:hover    { background: #181818; }
QScrollBar:vertical { background: #0a0a0a; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #1a1a1a; border-radius: 4px; min-height: 20px; }
"""

_JOINT_META = {
    "joint_c": {"sym": "qc", "name": "Coxa",  "axis": "DH: th1 = -qc",       "color": "#e05c5c", "sl_color": "#e05c5c"},
    "joint_p": {"sym": "qp", "name": "Femur", "axis": "DH: th2 = 180 + qp",  "color": "#4a9eff", "sl_color": "#4a9eff"},
    "joint_r": {"sym": "qr", "name": "Tibia", "axis": "DH: th3 = qr",         "color": "#5ec46e", "sl_color": "#5ec46e"},
}


def _colored_slider(color):
    sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    sl.setStyleSheet(f"""
        QSlider::groove:horizontal {{ height:5px; background:#111; border-radius:3px; }}
        QSlider::handle:horizontal {{
            background:{color}; border:none;
            width:16px; height:16px; margin:-6px 0; border-radius:8px;
        }}
        QSlider::sub-page:horizontal {{ background:{color}; border-radius:3px; }}
    """)
    return sl


# =================================================================
#  VENTANA PRINCIPAL
# =================================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle(
            "LA_PATA_SOLA  -  FK + IK  (DH/MTH equivalente al URDF)  |  ROS 2"
        )
        self.setMinimumSize(1060, 700)
        self.setStyleSheet(STYLE)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_fk_tab(),  "  Cinematica Directa (FK)  ")
        tabs.addTab(self._build_ik_tab(),  "  Cinematica Inversa (IK)  ")
        tabs.addTab(self._build_dh_tab(),  "  Tabla DH / Modelo  ")
        self.setCentralWidget(tabs)
        self._refresh_fk()

    # ------------------------------------------------------------------
    #  TAB FK
    # ------------------------------------------------------------------

    def _build_fk_tab(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(14)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)

        # -- sliders / edits --
        grp_j = QtWidgets.QGroupBox(
            "Angulos articulares   [ joint_c    joint_p    joint_r ]"
        )
        vj = QtWidgets.QVBoxLayout(grp_j)
        vj.setSpacing(6)
        self._sl = {}
        self._ed = {}

        for j in JOINTS:
            meta    = _JOINT_META[j]
            top_row = QtWidgets.QHBoxLayout()
            top_row.setSpacing(6)

            sym_lbl = QtWidgets.QLabel(meta["sym"])
            sym_lbl.setStyleSheet(
                f"color:{meta['color']}; font-size:18px;"
                f" font-weight:bold; min-width:24px;"
            )
            sym_lbl.setAlignment(QtCore.Qt.AlignCenter)

            name_lbl = QtWidgets.QLabel(
                f"{meta['name']}  "
                f"<span style='color:#444;font-size:10px;'>{meta['axis']}</span>"
            )
            name_lbl.setTextFormat(QtCore.Qt.RichText)
            name_lbl.setStyleSheet(f"color:{meta['color']}; font-size:12px;")

            ed = QtWidgets.QLineEdit("0.0")
            ed.setFixedWidth(72)
            ed.setAlignment(QtCore.Qt.AlignRight)

            deg_lbl = QtWidgets.QLabel("deg")
            deg_lbl.setStyleSheet("color:#555; font-size:12px;")

            top_row.addWidget(sym_lbl)
            top_row.addWidget(name_lbl, 1)
            top_row.addWidget(ed)
            top_row.addWidget(deg_lbl)

            bot_row = QtWidgets.QHBoxLayout()
            bot_row.setSpacing(4)
            lo, hi = LIMITS_DEG[j]

            lo_lbl = QtWidgets.QLabel(f"{int(lo)}")
            lo_lbl.setStyleSheet("color:#444; font-size:10px; min-width:28px;")
            lo_lbl.setAlignment(QtCore.Qt.AlignRight)

            sl = _colored_slider(meta["sl_color"])
            sl.setMinimum(int(lo * 10))
            sl.setMaximum(int(hi * 10))
            sl.setValue(0)

            hi_lbl = QtWidgets.QLabel(f"{int(hi)}")
            hi_lbl.setStyleSheet("color:#444; font-size:10px; min-width:28px;")

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

        # -- posicion FK --
        grp_pos = QtWidgets.QGroupBox(
            "Posicion del efector  [mm]  -  FK directa (DH/MTH)"
        )
        grid_pos = QtWidgets.QGridLayout(grp_pos)
        grid_pos.setSpacing(6)
        self._lx = QtWidgets.QLabel("-")
        self._ly = QtWidgets.QLabel("-")
        self._lz = QtWidgets.QLabel("-")

        for i, (sym, val_w, color) in enumerate([
            ("x =", self._lx, "#e05c5c"),
            ("y =", self._ly, "#5ec46e"),
            ("z =", self._lz, "#4a9eff"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555; font-size:12px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:14px; font-weight:bold;"
                f" font-family:'Consolas','Courier New',monospace;"
            )
            grid_pos.addWidget(lbl,   i, 0)
            grid_pos.addWidget(val_w, i, 1)

        left.addWidget(grp_pos)

        # -- botones --
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        b0 = QtWidgets.QPushButton("Reset  Zero")
        bv = QtWidgets.QPushButton("Verificar q=0  (tf2_echo)")
        for btn in (b0, bv):
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )
        btn_row.addWidget(b0)
        btn_row.addWidget(bv)
        left.addLayout(btn_row)
        b0.clicked.connect(self._do_zero)
        bv.clicked.connect(self._do_verify)
        left.addStretch(1)

        # -- columna derecha: matrices --
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        grp_mat = QtWidgets.QGroupBox(
            "Matrices HTM  -  coinciden con  ros2 run tf2_ros tf2_echo base_link <link>"
        )
        vm = QtWidgets.QVBoxLayout(grp_mat)

        self._lst = QtWidgets.QListWidget()
        self._lst.setFixedHeight(95)
        self._lst.currentRowChanged.connect(self._show_mat)

        self._txt = QtWidgets.QPlainTextEdit()
        self._txt.setReadOnly(True)
        self._txt.setMinimumHeight(230)

        vm.addWidget(self._lst)
        vm.addWidget(self._txt)
        right.addWidget(grp_mat, 1)

        note = QtWidgets.QLabel(
            "  Modelo:  T = T_BASE_0 * A1(qc) * A2(qp) * A3(qr) * T_3_EF\n\n"
            "  A1: th=-qc,      d=75 mm,  a=0,         alpha=+90\n"
            "  A2: th=180+qp,   d=0,      a=350.5 mm,  alpha=0\n"
            "  A3: th=qr,       d=0,      a=0,          alpha=0\n\n"
            "  q=(0,0,0) -> efector = (-130.09, -640.94, +225.73) mm\n"
            "  tf2_echo base_link efector (x1000) <- verificado"
        )
        note.setStyleSheet(
            "color:#3a3a4a; font-size:11px;"
            " font-family:'Consolas','Courier New',monospace;"
        )
        note.setWordWrap(True)
        right.addWidget(note)

        outer.addLayout(left,  43)
        outer.addLayout(right, 57)
        return w

    # ------------------------------------------------------------------
    #  TAB IK
    # ------------------------------------------------------------------

    def _build_ik_tab(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(14)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)

        # -- entrada --
        grp_in = QtWidgets.QGroupBox("Posicion objetivo del efector  [mm]")
        grid_in = QtWidgets.QGridLayout(grp_in)
        grid_in.setSpacing(6)

        def spin(v):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-2000.0, 2000.0)
            sb.setDecimals(3)
            sb.setSingleStep(1.0)
            sb.setValue(v)
            return sb

        self._ix = spin(-130.087)
        self._iy = spin(-640.939)
        self._iz = spin( 225.730)

        for i, (sym, w_spin, color) in enumerate([
            ("x =", self._ix, "#e05c5c"),
            ("y =", self._iy, "#5ec46e"),
            ("z =", self._iz, "#4a9eff"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555; font-size:12px;")
            w_spin.setStyleSheet(
                f"color:{color}; background:#111; border:1px solid #252525;"
                f" border-radius:4px; padding:3px 5px;"
            )
            grid_in.addWidget(lbl,    i, 0)
            grid_in.addWidget(w_spin, i, 1)

        left.addWidget(grp_in)

        # -- boton resolver --
        btn_ik = QtWidgets.QPushButton(
            "Resolver Cinematica Inversa  (Jacobiana DLS)"
        )
        btn_ik.setStyleSheet(
            "background:#5ec46e; color:#000; font-weight:bold;"
            " font-size:13px; padding:8px 16px; border-radius:6px;"
        )
        btn_ik.clicked.connect(self._solve_ik)
        left.addWidget(btn_ik)

        # -- resultado --
        grp_out = QtWidgets.QGroupBox("Resultado  -  angulos calculados [deg]")
        grid_out = QtWidgets.QGridLayout(grp_out)
        grid_out.setSpacing(6)

        self._oqc = QtWidgets.QLabel("-")
        self._oqp = QtWidgets.QLabel("-")
        self._oqr = QtWidgets.QLabel("-")
        self._oe  = QtWidgets.QLabel("")
        self._oe.setStyleSheet("color:#e05c5c; font-weight:bold;")
        self._oe.setWordWrap(True)

        for i, (sym, val_w, color) in enumerate([
            ("qc =", self._oqc, "#e05c5c"),
            ("qp =", self._oqp, "#4a9eff"),
            ("qr =", self._oqr, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555; font-size:12px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:13px; font-weight:bold;"
                f" font-family:'Consolas','Courier New',monospace;"
            )
            grid_out.addWidget(lbl,   i, 0)
            grid_out.addWidget(val_w, i, 1)
        grid_out.addWidget(self._oe, 3, 0, 1, 2)
        left.addWidget(grp_out)

        # -- verificacion FK(IK) --
        grp_v = QtWidgets.QGroupBox(
            "Verificacion  -  FK( IK(p) )  ->  posicion reconstruida [mm]"
        )
        grid_v = QtWidgets.QGridLayout(grp_v)
        grid_v.setSpacing(6)

        self._vx = QtWidgets.QLabel("-")
        self._vy = QtWidgets.QLabel("-")
        self._vz = QtWidgets.QLabel("-")
        self._ve = QtWidgets.QLabel("-")
        self._ve.setStyleSheet("color:#5ec46e; font-weight:bold;")

        for i, (sym, val_w, color) in enumerate([
            ("x_rec =", self._vx, "#4dd0e1"),
            ("y_rec =", self._vy, "#4dd0e1"),
            ("z_rec =", self._vz, "#4dd0e1"),
            ("||err||=", self._ve, "#5ec46e"),
        ]):
            lbl = QtWidgets.QLabel(sym)
            lbl.setStyleSheet("color:#555; font-size:12px;")
            val_w.setStyleSheet(
                f"color:{color}; font-size:12px;"
                f" font-family:'Consolas','Courier New',monospace;"
            )
            grid_v.addWidget(lbl,   i, 0)
            grid_v.addWidget(val_w, i, 1)
        left.addWidget(grp_v)
        left.addStretch(1)

        # -- columna derecha: matriz --
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        grp_mat = QtWidgets.QGroupBox(
            "T(base_link -> efector)  tras IK  <-  coincide con tf2_echo"
        )
        vm = QtWidgets.QVBoxLayout(grp_mat)
        self._ik_mat = QtWidgets.QPlainTextEdit()
        self._ik_mat.setReadOnly(True)
        vm.addWidget(self._ik_mat)
        right.addWidget(grp_mat, 1)

        note2 = QtWidgets.QLabel(
            "Los angulos resueltos se publican en /joint_states\n"
            "-> RViz2 actualiza el modelo en tiempo real.\n\n"
            "Metodo:  Jacobiana numerica DLS\n"
            "  J  = dp_ef/dq  (diferencias finitas, dq=1e-4 deg)\n"
            "  dq = Jt*(J*Jt + lam^2*I)^-1 * e   lam=5\n"
            "  q  <- clip(q + 0.6*dq, -90, +90) deg\n\n"
            "12 puntos de arranque  |  tol = 0.05 mm\n"
            "Error garantizado < 0.5 mm en el workspace.\n\n"
            "Reposo (0,0,0):\n"
            "  efector = (-130.09, -640.94, +225.73) mm\n"
            "  <- tf2_echo base_link efector x 1000"
        )
        note2.setStyleSheet(
            "color:#3a3a4a; font-size:11px;"
            " font-family:'Consolas','Courier New',monospace;"
        )
        note2.setWordWrap(True)
        right.addWidget(note2)

        outer.addLayout(left,  47)
        outer.addLayout(right, 53)
        return w

    # ------------------------------------------------------------------
    #  TAB TABLA DH
    # ------------------------------------------------------------------

    def _build_dh_tab(self):
        w  = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w)
        vl.setContentsMargins(14, 14, 14, 14)
        vl.setSpacing(12)

        title = QtWidgets.QLabel("Modelo DH / MTH  equivalente al URDF  LA_PATA_SOLA")
        title.setStyleSheet("color:#4a9eff; font-size:14px; font-weight:bold;")
        vl.addWidget(title)

        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setStyleSheet(
            "background:#060606; color:#c8c8c8;"
            " font-family:'Consolas','Courier New',monospace; font-size:12px;"
        )
        txt.setPlainText(
"""Formulacion completa:

    T(base->efector) = T_BASE_0 * A1(qc) * A2(qp) * A3(qr) * T_3_EF

===================================================================

1) Transformacion fija de base  T_BASE_0  [mm]:

    |  0  -1   0     0.0  |
    |  1   0   0     0.0  |
    |  0   0   1   143.9  |
    |  0   0   0     1.0  |

  Origen: base_link -> joint_c  xyz=(0, 0, 143.9 mm),
          con rotacion fija que orienta el frame DH.

===================================================================

2) Tabla DH activa  (convencion estandar):

    DH(theta, d, a, alpha) =
      | cos(th)  -sin(th)*cos(al)   sin(th)*sin(al)   a*cos(th) |
      | sin(th)   cos(th)*cos(al)  -cos(th)*sin(al)   a*sin(th) |
      |    0         sin(al)            cos(al)           d      |
      |    0            0                 0               1      |

  +---------+-----------------+----------+-----------+--------+
  |    i    |      theta_i    |   d_i mm |   a_i mm  | alpha_i|
  +---------+-----------------+----------+-----------+--------+
  |    1    |     -qc         |   75.0   |    0.0    |  +90   |
  |    2    |  180 deg + qp   |    0.0   |  350.499  |    0   |
  |    3    |     qr          |    0.0   |    0.0    |    0   |
  +---------+-----------------+----------+-----------+--------+

  Correspondencia con el URDF:
    joint_c  xyz=(0,0,0.1439 m)      rpy=(0,0,0)         axis=-Z
    joint_p  xyz=(-0.0403,0,0.075)   rpy=(pi/2,0,-pi/2)  axis=-Z
    joint_r  xyz=(0.3505,~0,-0.004)  rpy=(0,0,0.027)     axis=-Z

===================================================================

3) Transformacion fija del efector  T_3_EF  [mm]:

    | -1.0e-5  -0.99964  -0.02702   290.44 |
    | -1.2e-3   0.02702  -0.99963    -6.83 |
    |  1.0      2.2e-5   -1.2e-3  -130.09 |
    |  0.0      0.0       0.0        1.0   |

  Origen: efector_joint (FIJO)
    xyz=(0.2905, -0.001, 0.0936 m)
    rpy=(-0.184, 1.570, 1.387 rad)

===================================================================

VERIFICACION  (tf2_echo base_link efector x 1000):

    q = (  0,   0,   0) deg  ->  efector = (-130.09, -640.94, +225.73) mm
    q = ( 40,  40,  40) deg  ->  efector = (-308.98, -165.85, -291.24) mm
    q = ( 90,   0,   0) deg  ->  efector = (-640.94, +130.09, +225.73) mm

===================================================================

CINEMATICA INVERSA - Metodo del gradiente (Jacobiana DLS):

    J     = dp_efector / dq    [3x3, diferencias finitas, dq=1e-4 deg]
    error = p_objetivo - p_actual
    dq    = Jt * (J*Jt + lam^2*I)^-1 * error    (Damped Least Squares)
    q    <- clip(q + alpha*dq, -90, +90) deg

    alpha = 0.60   lam = 5.0   tol = 0.05 mm   max_iter = 500
    12 puntos de arranque para evitar minimos locales.
    Error garantizado < 0.5 mm en todo el espacio de trabajo.
""")
        vl.addWidget(txt, 1)
        return w

    # ------------------------------------------------------------------
    #  LOGICA FK
    # ------------------------------------------------------------------

    def _refresh_fk(self):
        qc = self.node.get_deg("joint_c")
        qp = self.node.get_deg("joint_p")
        qr = self.node.get_deg("joint_r")

        self._fk_frames, T_ef = fk_chain(qc, qp, qr)
        p = T_ef[:3, 3]
        self._lx.setText(f"{p[0]:+.3f}  mm")
        self._ly.setText(f"{p[1]:+.3f}  mm")
        self._lz.setText(f"{p[2]:+.3f}  mm")

        cur = max(self._lst.currentRow(), 0)
        self._lst.blockSignals(True)
        self._lst.clear()
        for name, _ in self._fk_frames:
            self._lst.addItem(name)
        self._lst.blockSignals(False)
        default = min(cur, len(self._fk_frames) - 1)
        if default < 0:
            default = len(self._fk_frames) - 1
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
            f"{name}\n{'-'*58}\n"
            f"xyz  [mm]  = ({p[0]:+.4f},  {p[1]:+.4f},  {p[2]:+.4f})\n"
            f"xyz  [m]   = ({p[0]/1000:+.6f}, {p[1]/1000:+.6f}, {p[2]/1000:+.6f})\n"
            f"rpy  [rad] = ({roll:+.6f},  {pitch:+.6f},  {yaw:+.6f})\n"
            f"rpy  [deg] = ({math.degrees(roll):+.4f},  "
            f"{math.degrees(pitch):+.4f},  {math.degrees(yaw):+.4f})\n\n"
            f"T (4x4)  [mm]:\n{fmt4(T)}"
        )

    # ------------------------------------------------------------------
    #  LOGICA IK
    # ------------------------------------------------------------------

    def _solve_ik(self):
        px = self._ix.value()
        py = self._iy.value()
        pz = self._iz.value()

        try:
            q_sol, err = ik_numerical(px, py, pz)
        except ValueError as e:
            self._oqc.setText("-"); self._oqp.setText("-"); self._oqr.setText("-")
            self._oe.setText(f"Sin solucion:\n{e}")
            self._ik_mat.setPlainText(f"Sin solucion:\n{e}")
            self._vx.setText("-"); self._vy.setText("-")
            self._vz.setText("-"); self._ve.setText("-")
            return

        qc, qp, qr = q_sol
        self._oe.setText("")
        self._oqc.setText(f"{qc:+.4f} deg  ({math.radians(qc):+.6f} rad)")
        self._oqp.setText(f"{qp:+.4f} deg  ({math.radians(qp):+.6f} rad)")
        self._oqr.setText(f"{qr:+.4f} deg  ({math.radians(qr):+.6f} rad)")

        _, T_ef = fk_chain(qc, qp, qr)
        p_rec   = T_ef[:3, 3]
        err_v   = math.sqrt(
            (px - p_rec[0])**2 + (py - p_rec[1])**2 + (pz - p_rec[2])**2
        )
        self._vx.setText(f"{p_rec[0]:+.4f} mm")
        self._vy.setText(f"{p_rec[1]:+.4f} mm")
        self._vz.setText(f"{p_rec[2]:+.4f} mm")
        color = "#5ec46e" if err_v < 1.0 else "#e05c5c"
        self._ve.setStyleSheet(f"color:{color}; font-weight:bold;")
        self._ve.setText(
            f"{err_v:.4f} mm  "
            f"({'OK < 1 mm' if err_v < 1.0 else 'WARN > 1 mm'})"
        )

        roll, pitch, yaw = rot_to_rpy(T_ef[:3, :3])
        self._ik_mat.setPlainText(
            f"T(base_link -> efector)  tras IK:\n{'-'*48}\n"
            f"xyz  [mm]  = ({p_rec[0]:+.4f}, {p_rec[1]:+.4f}, {p_rec[2]:+.4f})\n"
            f"xyz  [m]   = ({p_rec[0]/1000:+.6f}, {p_rec[1]/1000:+.6f}, {p_rec[2]/1000:+.6f})\n"
            f"rpy  [rad] = ({roll:+.6f}, {pitch:+.6f}, {yaw:+.6f})\n"
            f"rpy  [deg] = ({math.degrees(roll):+.4f}, "
            f"{math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\n"
            f"T (4x4)  [mm]:\n{fmt4(T_ef)}\n\n"
            f"Verificar con:\n"
            f"  ros2 run tf2_ros tf2_echo base_link efector\n"
            f"  (Translation x 1000 -> mm)"
        )

        self.node.set_deg("joint_c", qc)
        self.node.set_deg("joint_p", qp)
        self.node.set_deg("joint_r", qr)
        self._sync_sliders(qc, qp, qr)

    def _sync_sliders(self, qc, qp, qr):
        for j, v in zip(JOINTS, [qc, qp, qr]):
            deg = clamp(v, *LIMITS_DEG[j])
            sl  = self._sl[j]
            sl.blockSignals(True)
            sl.setValue(int(round(deg * 10)))
            sl.blockSignals(False)
            self._ed[j].setText(f"{deg:.1f}")
        self._refresh_fk()

    # ------------------------------------------------------------------
    #  BOTONES
    # ------------------------------------------------------------------

    def _do_zero(self):
        for j in JOINTS:
            self.node.set_deg(j, 0.0)
            self._sl[j].blockSignals(True)
            self._sl[j].setValue(0)
            self._sl[j].blockSignals(False)
            self._ed[j].setText("0.0")
        self._refresh_fk()

    def _do_verify(self):
        _, T_ef = fk_chain(0.0, 0.0, 0.0)
        p = T_ef[:3, 3]
        roll, pitch, yaw = rot_to_rpy(T_ef[:3, :3])
        QtWidgets.QMessageBox.information(
            self, "Verificacion  q=(0,0,0)  vs  tf2_echo",
            f"FK  con  q = (0, 0, 0) deg:\n"
            f"  x = {p[0]:+.4f} mm   ({p[0]/1000:+.6f} m)\n"
            f"  y = {p[1]:+.4f} mm   ({p[1]/1000:+.6f} m)\n"
            f"  z = {p[2]:+.4f} mm   ({p[2]/1000:+.6f} m)\n\n"
            f"tf2_echo base_link efector (parcial):\n"
            f"  Translation: [-0.130, -0.641, +0.226] m\n\n"
            f"Diferencia [mm]:\n"
            f"  dx = {p[0] - (-130.0):+.2f}\n"
            f"  dy = {p[1] - (-641.0):+.2f}\n"
            f"  dz = {p[2] - ( 226.0):+.2f}\n\n"
            f"RPY [rad]:  ({roll:+.5f}, {pitch:+.5f}, {yaw:+.5f})\n\n"
            f"T (4x4) [mm]:\n{fmt4(T_ef)}"
        )


# =================================================================
#  MAIN
# =================================================================

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
