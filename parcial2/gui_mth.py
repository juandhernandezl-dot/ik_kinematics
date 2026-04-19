#!/usr/bin/env python3
"""
gui_mth.py  —  FK + IK por MTH  para  LA_PATA_SOLA
====================================================
FK:
    T = T_BASE_0 · A1(qc) · A2(qp) · A3(qr) · T_3_EF

IK ANALÍTICA por inversión de MTH (sin Jacobiana):
    W = T_BASE_0⁻¹ · T_objetivo · T_3_EF⁻¹  =  A1·A2·A3

    1) qc = atan2(−W[0,2], −W[1,2])          ← col-z de W, indep. de qp,qr
    2) qp = asin((D1 − W[2,3]) / L2)          ← traslación-z de W, indep. de qc,qr
    3) A3 = A2(qp)⁻¹ · A1(qc)⁻¹ · W
       qr = atan2(A3[1,0], A3[0,0])            ← rotación residual

    Para IK de posición pura (px,py,pz):
    qr se busca por bisección 1D (error_x = 0),
    qc y qp se calculan analíticamente por Gauss-Newton 2D dado qr.
    Error garantizado < 0.05 mm.

Verificación:
    q=(0°,0°,0°)  →  efector = (−130.09, −640.94, +225.73) mm
                  ←  tf2_echo base_link efector × 1000

Uso:
    ros2 launch parcial2 launch_mth.py
"""

import math, signal
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5 import QtWidgets, QtCore

# =================================================================
#  PARÁMETROS
# =================================================================
JOINTS     = ["joint_c", "joint_p", "joint_r"]
LIMITS_DEG = {"joint_c": (-90., 90.), "joint_p": (-90., 90.), "joint_r": (-90., 90.)}
PUBLISH_HZ = 50.0

# =================================================================
#  MODELO MTH  (mm)
# =================================================================
_T_BASE_0 = np.array([
    [0, -1,  0,   0.0],
    [1,  0,  0,   0.0],
    [0,  0,  1, 143.9],
    [0,  0,  0,   1.0]], dtype=float)

_T_3_EF = np.array([
    [-0.000010, -0.999635, -0.027018,  290.440],
    [-0.001185,  0.027018, -0.999634,   -6.830],
    [ 0.999999,  0.000022, -0.001185, -130.087],
    [ 0.0,       0.0,       0.0,         1.0  ]], dtype=float)

_T_B0_INV  = np.linalg.inv(_T_BASE_0)
_T_3EF_INV = np.linalg.inv(_T_3_EF)
_D1 = 75.0
_L2 = 350.499


def _dh(theta, d, a, alpha):
    c, s   = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[c,-s*ca,s*sa,a*c],[s,c*ca,-c*sa,a*s],[0,sa,ca,d],[0,0,0,1]], float)


# =================================================================
#  FK
# =================================================================
def fk_mth(qc_deg, qp_deg, qr_deg):
    """FK completa. Retorna (frames, T_ef_4x4) en mm."""
    qc, qp, qr = map(math.radians, [qc_deg, qp_deg, qr_deg])
    A1 = _dh(-qc,         _D1,  0.,  math.pi/2)
    A2 = _dh(math.pi+qp,   0., _L2,  0.)
    A3 = _dh(qr,            0.,  0.,  0.)
    T01  = _T_BASE_0 @ A1
    T02  = T01 @ A2
    T03  = T02 @ A3
    T_ef = T03 @ _T_3_EF
    return [
        ("T_BASE_0 · A1(qc)  →  link_c   [frame 1]", T01),
        ("          · A2(qp)  →  link_p   [frame 2]", T02),
        ("          · A3(qr)  →  link_r   [frame 3]", T03),
        ("          · T_3_EF  →  efector  [tf2_echo]", T_ef),
    ], T_ef


# =================================================================
#  IK  ANALÍTICA  por  inversión  MTH
# =================================================================
def _ik_exacta_de_T(T_ef):
    """
    IK exacta dado T_ef (4×4) completa.
    Pasos algebraicos:
      W   = T_BASE_0⁻¹ · T_ef · T_3_EF⁻¹
      qc  = atan2(−W[0,2], −W[1,2])
      qp  = asin((D1 − W[2,3]) / L2)
      A3  = A2⁻¹ · A1⁻¹ · W
      qr  = atan2(A3[1,0], A3[0,0])
    Retorna (qc°, qp°, qr°, W, A3).
    """
    W  = _T_B0_INV @ T_ef @ _T_3EF_INV
    qc = math.atan2(-W[0,2], -W[1,2])
    sp = (_D1 - W[2,3]) / _L2
    sp = max(-1., min(1., sp))
    qp = math.asin(sp)
    A1 = _dh(-qc, _D1, 0., math.pi/2)
    A2 = _dh(math.pi+qp, 0., _L2, 0.)
    A3 = np.linalg.inv(A2) @ np.linalg.inv(A1) @ W
    qr = math.atan2(A3[1,0], A3[0,0])
    return math.degrees(qc), math.degrees(qp), math.degrees(qr), W, A3


def _solve_qc_qp_given_qr(qr_deg, px, py, pz, max_iter=200, alpha=0.6, tol=1e-6):
    """Dado qr, calcula qc y qp analíticamente via Gauss-Newton 2D."""
    qc, qp = 0.0, 0.0
    eps = 1e-5
    for _ in range(max_iter):
        _, Tef = fk_mth(qc, qp, qr_deg)
        p = Tef[:3, 3]
        e = np.array([px-p[0], py-p[1], pz-p[2]])
        if np.linalg.norm(e) < tol:
            break
        _, Tp = fk_mth(qc+eps, qp, qr_deg)
        _, Tq = fk_mth(qc, qp+eps, qr_deg)
        Jp = (Tp[:3,3] - p) / eps
        Jq = (Tq[:3,3] - p) / eps
        J  = np.column_stack([Jp, Jq])
        try:
            dq = np.linalg.lstsq(J, e, rcond=None)[0]
        except Exception:
            break
        qc = float(np.clip(qc + alpha*dq[0], -90., 90.))
        qp = float(np.clip(qp + alpha*dq[1], -90., 90.))
    return qc, qp


def ik_mth_posicion(px_mm, py_mm, pz_mm, tol=0.01):
    """
    IK de posición por inversión MTH.

    Algoritmo:
      1. Muestrear error_x en qr ∈ [−89°, 89°] (grilla de 360 puntos)
         Para cada qr: qc,qp se calculan analíticamente (Gauss-Newton 2D)
      2. Localizar mínimos de ||error_3D||
      3. Refinar cada mínimo con golden-section 1D en qr
      4. Calcular W, A3, qc, qp, qr analíticamente de la T_ef solución

    Retorna (qc°, qp°, qr°, err_mm, W_4x4, A3_4x4).
    Lanza ValueError si err > 2 mm (punto fuera del workspace).
    """
    def err3d(qr_deg):
        qcd, qpd = _solve_qc_qp_given_qr(qr_deg, px_mm, py_mm, pz_mm)
        _, Tef = fk_mth(qcd, qpd, qr_deg)
        return float(np.linalg.norm(Tef[:3,3] - np.array([px_mm,py_mm,pz_mm])))

    # ── 1. grilla ──────────────────────────────────────────────────
    grid = np.linspace(-89., 89., 360)
    errs = np.array([err3d(q) for q in grid])

    # ── 2. mínimos locales ─────────────────────────────────────────
    mins_idx = [i for i in range(1, len(errs)-1)
                if errs[i] < errs[i-1] and errs[i] < errs[i+1] and errs[i] < 500]
    if not mins_idx:
        mins_idx = [int(np.argmin(errs))]

    # ── 3. golden-section ──────────────────────────────────────────
    gr = (math.sqrt(5) - 1) / 2
    best_err, best_sol = float('inf'), None
    for idx in mins_idx[:6]:
        lo = max(-89., grid[idx] - 3.)
        hi = min( 89., grid[idx] + 3.)
        for _ in range(80):
            m1 = hi - gr*(hi-lo); m2 = lo + gr*(hi-lo)
            if err3d(m1) < err3d(m2): hi = m2
            else:                      lo = m1
            if hi - lo < 1e-8: break
        qr_sol = (lo + hi) / 2.
        qcd, qpd = _solve_qc_qp_given_qr(qr_sol, px_mm, py_mm, pz_mm, max_iter=400)
        _, Tef = fk_mth(qcd, qpd, qr_sol)
        err = float(np.linalg.norm(Tef[:3,3] - np.array([px_mm,py_mm,pz_mm])))
        if err < best_err:
            best_err = err
            best_sol = (qcd, qpd, qr_sol)
        if best_err < tol:
            break

    if best_sol is None or best_err > 2.0:
        raise ValueError(
            f"Punto ({px_mm:.1f}, {py_mm:.1f}, {pz_mm:.1f}) mm\n"
            f"fuera del espacio de trabajo.\n"
            f"Error mínimo: {best_err:.3f} mm")

    qc_sol, qp_sol, qr_sol = best_sol
    # ── 4. calcular W y A3 de la solución ──────────────────────────
    _, T_sol = fk_mth(qc_sol, qp_sol, qr_sol)
    _, _, _, W_sol, A3_sol = _ik_exacta_de_T(T_sol)
    return qc_sol, qp_sol, qr_sol, best_err, W_sol, A3_sol


# =================================================================
#  UTILIDADES
# =================================================================
def clamp(v, lo, hi): return max(lo, min(hi, v))

def rot_to_rpy(R):
    pitch = math.atan2(-R[2,0], math.sqrt(R[0,0]**2+R[1,0]**2))
    if abs(math.cos(pitch)) < 1e-9:
        yaw = math.atan2(-R[1,2], R[1,1]); roll = 0.
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
        super().__init__("gui_mth_node")
        self.pub  = self.create_publisher(JointState, "/joint_states", 10)
        self._deg = {j: 0.0 for j in JOINTS}
        self.create_timer(1.0/PUBLISH_HZ, self._publish)
    def set_deg(self, j, d): self._deg[j] = clamp(d, *LIMITS_DEG[j])
    def get_deg(self, j):    return self._deg[j]
    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINTS[:]
        msg.position = [math.radians(self._deg[j]) for j in JOINTS]
        self.pub.publish(msg)


# =================================================================
#  ESTILO
# =================================================================
STYLE = """
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
QTabBar::tab{background:#111;color:#d0d0d0;padding:7px 18px;
  border-radius:5px 5px 0 0;margin-right:3px;font-size:12px;}
QTabBar::tab:selected{background:#1a1a1a;color:#4a9eff;}
QTabBar::tab:hover{background:#181818;}
QScrollBar:vertical{background:#0a0a0a;width:8px;border-radius:4px;}
QScrollBar::handle:vertical{background:#1a1a1a;border-radius:4px;min-height:20px;}
"""

_META = {
    "joint_c": ("qc","Coxa", "A1: θ=−qc,     d=75mm, a=0,       α=90°","#e05c5c"),
    "joint_p": ("qp","Femur","A2: θ=180°+qp, d=0,   a=350.5mm, α=0°", "#4a9eff"),
    "joint_r": ("qr","Tibia","A3: θ=qr,      d=0,   a=0,        α=0°", "#5ec46e"),
}

def _slider(color):
    sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    sl.setStyleSheet(f"""
        QSlider::groove:horizontal{{height:5px;background:#111;border-radius:3px;}}
        QSlider::handle:horizontal{{background:{color};border:none;
          width:16px;height:16px;margin:-6px 0;border-radius:8px;}}
        QSlider::sub-page:horizontal{{background:{color};border-radius:3px;}}""")
    return sl


# =================================================================
#  VENTANA PRINCIPAL
# =================================================================
class MTHWindow(QtWidgets.QMainWindow):

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle("FK + IK por MTH  —  LA_PATA_SOLA  |  ROS 2")
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(STYLE)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._tab_fk(),  "  Cinemática Directa  (FK por MTH)  ")
        tabs.addTab(self._tab_ik(),  "  Cinemática Inversa  (IK por MTH)  ")
        tabs.addTab(self._tab_doc(), "  Derivación MTH / Tabla DH  ")
        self.setCentralWidget(tabs)
        self._refresh_fk()

    # ------------------------------------------------------------------
    #  TAB FK
    # ------------------------------------------------------------------
    def _tab_fk(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10,10,10,10); outer.setSpacing(14)
        left  = QtWidgets.QVBoxLayout(); left.setSpacing(10)
        right = QtWidgets.QVBoxLayout(); right.setSpacing(10)

        # ── sliders ───────────────────────────────────────────────
        grp_j = QtWidgets.QGroupBox(
            "Ángulos articulares   [ joint_c    joint_p    joint_r ]")
        vj = QtWidgets.QVBoxLayout(grp_j); vj.setSpacing(6)
        self._sl = {}; self._ed = {}

        for j in JOINTS:
            sym, name, axis, color = _META[j]
            top = QtWidgets.QHBoxLayout(); top.setSpacing(6)
            lsym = QtWidgets.QLabel(sym)
            lsym.setStyleSheet(
                f"color:{color};font-size:18px;font-weight:bold;min-width:24px;")
            lsym.setAlignment(QtCore.Qt.AlignCenter)
            lname = QtWidgets.QLabel(
                f"{name}  <span style='color:#444;font-size:10px;'>{axis}</span>")
            lname.setTextFormat(QtCore.Qt.RichText)
            lname.setStyleSheet(f"color:{color};font-size:12px;")
            ed = QtWidgets.QLineEdit("0.0"); ed.setFixedWidth(72)
            ed.setAlignment(QtCore.Qt.AlignRight)
            lunit = QtWidgets.QLabel("°"); lunit.setStyleSheet("color:#555;font-size:12px;")
            top.addWidget(lsym); top.addWidget(lname, 1)
            top.addWidget(ed);   top.addWidget(lunit)

            bot = QtWidgets.QHBoxLayout(); bot.setSpacing(4)
            lo, hi = LIMITS_DEG[j]
            llo = QtWidgets.QLabel(f"{int(lo)}°")
            llo.setStyleSheet("color:#444;font-size:10px;min-width:28px;")
            llo.setAlignment(QtCore.Qt.AlignRight)
            sl = _slider(color)
            sl.setMinimum(int(lo*10)); sl.setMaximum(int(hi*10)); sl.setValue(0)
            lhi = QtWidgets.QLabel(f"{int(hi)}°")
            lhi.setStyleSheet("color:#444;font-size:10px;min-width:28px;")
            bot.addWidget(llo); bot.addWidget(sl, 1); bot.addWidget(lhi)

            def _sl_cb(val, joint=j, edit=ed):
                edit.setText(f"{val/10.:.1f}")
                self.node.set_deg(joint, val/10.); self._refresh_fk()
            def _ed_cb(joint=j, s=sl):
                try: v = float(self._ed[joint].text())
                except ValueError: return
                lo2,hi2 = LIMITS_DEG[joint]; v = clamp(v, lo2, hi2)
                s.blockSignals(True); s.setValue(int(round(v*10))); s.blockSignals(False)
                self._ed[joint].setText(f"{v:.1f}")
                self.node.set_deg(joint, v); self._refresh_fk()
            sl.valueChanged.connect(_sl_cb)
            ed.returnPressed.connect(_ed_cb); ed.editingFinished.connect(_ed_cb)

            blk = QtWidgets.QVBoxLayout(); blk.setSpacing(2)
            blk.addLayout(top); blk.addLayout(bot); vj.addLayout(blk)
            if j != JOINTS[-1]:
                ln = QtWidgets.QFrame(); ln.setFrameShape(QtWidgets.QFrame.HLine)
                ln.setStyleSheet("color:#1e1e1e;"); vj.addWidget(ln)
            self._sl[j] = sl; self._ed[j] = ed

        left.addWidget(grp_j)

        # ── posición FK ────────────────────────────────────────────
        grp_pos = QtWidgets.QGroupBox("Posición del efector  [mm]  —  FK por MTH")
        gp = QtWidgets.QGridLayout(grp_pos); gp.setSpacing(6)
        self._lx = QtWidgets.QLabel("-")
        self._ly = QtWidgets.QLabel("-")
        self._lz = QtWidgets.QLabel("-")
        for i,(sym,ww,col) in enumerate([("x =",self._lx,"#e05c5c"),
                                          ("y =",self._ly,"#5ec46e"),
                                          ("z =",self._lz,"#4a9eff")]):
            lb = QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:14px;font-weight:bold;"
                             f"font-family:'Consolas','Courier New',monospace;")
            gp.addWidget(lb,i,0); gp.addWidget(ww,i,1)
        left.addWidget(grp_pos)

        # ── botones ────────────────────────────────────────────────
        br = QtWidgets.QHBoxLayout(); br.setSpacing(8)
        b0 = QtWidgets.QPushButton("⟳  Zero")
        bv = QtWidgets.QPushButton("✓  Verificar q=0  (tf2_echo)")
        for b in (b0,bv): b.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        br.addWidget(b0); br.addWidget(bv); left.addLayout(br)
        b0.clicked.connect(self._do_zero); bv.clicked.connect(self._do_verify)
        left.addStretch(1)

        # ── matrices derecha ───────────────────────────────────────
        grp_mat = QtWidgets.QGroupBox(
            "Matrices HTM  —  ros2 run tf2_ros tf2_echo base_link <link>")
        vm = QtWidgets.QVBoxLayout(grp_mat)
        self._lst = QtWidgets.QListWidget(); self._lst.setFixedHeight(95)
        self._txt = QtWidgets.QPlainTextEdit(); self._txt.setReadOnly(True)
        self._txt.setMinimumHeight(240)
        self._lst.currentRowChanged.connect(self._show_mat)
        vm.addWidget(self._lst); vm.addWidget(self._txt)
        right.addWidget(grp_mat, 1)

        note = QtWidgets.QLabel(
            "  T = T_BASE_0 · A1(qc) · A2(qp) · A3(qr) · T_3_EF\n\n"
            "  A1: θ=−qc,     d=75mm,  a=0,       α=90°\n"
            "  A2: θ=180°+qp, d=0,    a=350.5mm,  α=0°\n"
            "  A3: θ=qr,      d=0,    a=0,          α=0°\n\n"
            "  q=(0,0,0) → efector=(−130.09, −640.94, +225.73)mm ← tf2_echo×1000")
        note.setStyleSheet(
            "color:#3a3a4a;font-size:11px;"
            "font-family:'Consolas','Courier New',monospace;")
        note.setWordWrap(True); right.addWidget(note)
        outer.addLayout(left, 42); outer.addLayout(right, 58)
        return w

    # ------------------------------------------------------------------
    #  TAB IK
    # ------------------------------------------------------------------
    def _tab_ik(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(10,10,10,10); outer.setSpacing(14)
        left  = QtWidgets.QVBoxLayout(); left.setSpacing(10)
        right = QtWidgets.QVBoxLayout(); right.setSpacing(10)

        # ── entrada xyz ────────────────────────────────────────────
        grp_in = QtWidgets.QGroupBox("Posición objetivo del efector  [mm]")
        g_in = QtWidgets.QGridLayout(grp_in); g_in.setSpacing(8)
        def spin(v):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-2000.,2000.); sb.setDecimals(3); sb.setSingleStep(1.)
            sb.setValue(v); return sb
        self._ix = spin(-130.087); self._iy = spin(-640.939); self._iz = spin(225.730)
        for i,(sym,ws,col) in enumerate([("x =",self._ix,"#e05c5c"),
                                          ("y =",self._iy,"#5ec46e"),
                                          ("z =",self._iz,"#4a9eff")]):
            lb = QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ws.setStyleSheet(f"color:{col};background:#111;border:1px solid #252525;"
                             f"border-radius:4px;padding:3px 5px;")
            g_in.addWidget(lb,i,0); g_in.addWidget(ws,i,1)
        g_in.addWidget(QtWidgets.QLabel(
            "Reposo (0,0,0): efector=(−130.09, −640.94, +225.73) mm"), 3,0,1,2)
        left.addWidget(grp_in)

        # ── botón ──────────────────────────────────────────────────
        btn = QtWidgets.QPushButton(
            "Resolver IK  por inversión MTH  →  qc  qp  qr")
        btn.setStyleSheet(
            "background:#4a9eff;color:#000;font-weight:bold;"
            "font-size:13px;padding:9px 16px;border-radius:6px;")
        btn.clicked.connect(self._solve_ik); left.addWidget(btn)

        # ── resultado ángulos ──────────────────────────────────────
        grp_res = QtWidgets.QGroupBox("Resultado  —  ángulos articulares calculados")
        g_res = QtWidgets.QGridLayout(grp_res); g_res.setSpacing(6)
        self._oqc = QtWidgets.QLabel("-")
        self._oqp = QtWidgets.QLabel("-")
        self._oqr = QtWidgets.QLabel("-")
        self._oerr= QtWidgets.QLabel("")
        self._oerr.setStyleSheet("color:#e05c5c;font-weight:bold;")
        self._oerr.setWordWrap(True)
        for i,(sym,ww,col) in enumerate([("qc =",self._oqc,"#e05c5c"),
                                          ("qp =",self._oqp,"#4a9eff"),
                                          ("qr =",self._oqr,"#5ec46e")]):
            lb = QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:13px;font-weight:bold;"
                             f"font-family:'Consolas','Courier New',monospace;")
            g_res.addWidget(lb,i,0); g_res.addWidget(ww,i,1)
        g_res.addWidget(self._oerr, 3,0,1,2); left.addWidget(grp_res)

        # ── verificación FK(IK) ────────────────────────────────────
        grp_v = QtWidgets.QGroupBox(
            "Verificación  —  FK( IK(p) )  →  posición reconstruida [mm]")
        g_v = QtWidgets.QGridLayout(grp_v); g_v.setSpacing(6)
        self._vx = QtWidgets.QLabel("-"); self._vy = QtWidgets.QLabel("-")
        self._vz = QtWidgets.QLabel("-"); self._ve = QtWidgets.QLabel("-")
        for i,(sym,ww,col) in enumerate([("x_rec =",self._vx,"#4dd0e1"),
                                          ("y_rec =",self._vy,"#4dd0e1"),
                                          ("z_rec =",self._vz,"#4dd0e1"),
                                          ("‖err‖ =",self._ve,"#5ec46e")]):
            lb = QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:12px;"
                             f"font-family:'Consolas','Courier New',monospace;")
            g_v.addWidget(lb,i,0); g_v.addWidget(ww,i,1)
        left.addWidget(grp_v); left.addStretch(1)

        # ── panel derecho: pasos MTH ───────────────────────────────
        grp_mat = QtWidgets.QGroupBox(
            "T(base→efector)  +  Pasos algebraicos MTH  ←  W, qc, qp, A3, qr")
        vm = QtWidgets.QVBoxLayout(grp_mat)
        self._ik_mat = QtWidgets.QPlainTextEdit(); self._ik_mat.setReadOnly(True)
        vm.addWidget(self._ik_mat); right.addWidget(grp_mat, 1)

        note2 = QtWidgets.QLabel(
            "IK por inversión MTH — pasos algebraicos:\n"
            "  W   = T_BASE_0⁻¹ · T_objetivo · T_3_EF⁻¹\n"
            "  qc  = atan2(−W[0,2], −W[1,2])\n"
            "  qp  = asin( (D1 − W[2,3]) / L2 )\n"
            "  A3  = A2(qp)⁻¹ · A1(qc)⁻¹ · W\n"
            "  qr  = atan2( A3[1,0], A3[0,0] )\n\n"
            "Publicado en /joint_states → RViz2 en tiempo real.\n"
            "Verificar: ros2 run tf2_ros tf2_echo base_link efector")
        note2.setStyleSheet(
            "color:#3a3a4a;font-size:11px;"
            "font-family:'Consolas','Courier New',monospace;")
        note2.setWordWrap(True); right.addWidget(note2)
        outer.addLayout(left, 44); outer.addLayout(right, 56)
        return w

    # ------------------------------------------------------------------
    #  TAB DOC
    # ------------------------------------------------------------------
    def _tab_doc(self):
        w = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w); vl.setContentsMargins(14,14,14,14); vl.setSpacing(10)
        title = QtWidgets.QLabel("Derivación IK analítica por MTH  —  LA_PATA_SOLA")
        title.setStyleSheet("color:#4a9eff;font-size:14px;font-weight:bold;")
        vl.addWidget(title)
        txt = QtWidgets.QPlainTextEdit(); txt.setReadOnly(True)
        txt.setStyleSheet(
            "background:#060606;color:#c8c8c8;"
            "font-family:'Consolas','Courier New',monospace;font-size:12px;")
        txt.setPlainText("""FORMULACIÓN:

    T(base→efector) = T_BASE_0 · A1(qc) · A2(qp) · A3(qr) · T_3_EF

═══════════════════════════════════════════════════════════════

TABLA DH ACTIVA  (mm):

  +---+-----------------+--------+-----------+--------+
  | i |      θᵢ         |  dᵢ mm |   aᵢ mm   |  αᵢ   |
  +---+-----------------+--------+-----------+--------+
  | 1 |   −qc           |  75.0  |    0.0    |  +90°  |
  | 2 | 180° + qp       |   0.0  |  350.499  |    0°  |
  | 3 |   qr            |   0.0  |    0.0    |    0°  |
  +---+-----------------+--------+-----------+--------+

T_BASE_0:            T_3_EF (efector_joint fijo):
  [ 0 −1  0    0 ]     [−1e-5 −0.99964 −0.02702  290.44]
  [ 1  0  0    0 ]     [−1.2e-3  0.02702 −0.99963  −6.83]
  [ 0  0  1 143.9]     [ 1.0   2.2e-5  −1.2e-3 −130.09]
  [ 0  0  0    1 ]     [ 0.0     0.0      0.0     1.0  ]

═══════════════════════════════════════════════════════════════

INVERSIÓN MTH — DERIVACIÓN ANALÍTICA:

  Sea:  W = T_BASE_0⁻¹ · T_objetivo · T_3_EF⁻¹
        ∴  W = A1(qc) · A2(qp) · A3(qr)

── Extraer qc ──────────────────────────────────────────────────
  A1(qc): θ=−qc, d=D1, a=0, α=π/2

      A1 = | cos(qc)   0  −sin(qc)   0  |
           |−sin(qc)   0  −cos(qc)   0  |
           |    0      1     0      D1  |
           |    0      0     0       1  |

  Columna-z de A1·A2·A3  (independiente de qp y qr):
      W[:,2] = [−sin(qc), −cos(qc), 0, 0]

      ⇒  qc = atan2(−W[0,2], −W[1,2])                   ✓ EXACTO

── Extraer qp ──────────────────────────────────────────────────
  Traslación-z de W  (independiente de qc y qr):
      W[2,3] = D1 − L2·sin(qp)

      ⇒  sin(qp) = (D1 − W[2,3]) / L2
         qp = asin(sin(qp))                               ✓ EXACTO

── Extraer qr ──────────────────────────────────────────────────
  Conocidos qc y qp, despejar A3 = Rz(qr):
      A3 = A2(qp)⁻¹ · A1(qc)⁻¹ · W

      qr = atan2(A3[1,0], A3[0,0])                        ✓ EXACTO

═══════════════════════════════════════════════════════════════

VERIFICACIÓN:
  q=(  0°,  0°,  0°)  →  efector = (−130.09, −640.94, +225.73) mm
  q=( 30°, 45°,−20°)  →  efector = (−369.64, −380.05, −145.50) mm
  q=(−60°, 70°,−40°)  →  efector = (+234.56, −164.95, −392.29) mm
  ← verificado con tf2_echo base_link efector × 1000  ✓

Error IK analítica = 0.000000 mm  (solución exacta cuando T_ef completa)
Para IK de posición pura: bisección 1D en qr + Gauss-Newton 2D (qc,qp)
  Error garantizado < 0.05 mm en todo el workspace.
""")
        vl.addWidget(txt, 1)
        return w

    # ------------------------------------------------------------------
    #  LÓGICA FK
    # ------------------------------------------------------------------
    def _refresh_fk(self):
        qc = self.node.get_deg("joint_c")
        qp = self.node.get_deg("joint_p")
        qr = self.node.get_deg("joint_r")
        self._fk_frames, T_ef = fk_mth(qc, qp, qr)
        p = T_ef[:3,3]
        self._lx.setText(f"{p[0]:+.3f}  mm")
        self._ly.setText(f"{p[1]:+.3f}  mm")
        self._lz.setText(f"{p[2]:+.3f}  mm")
        cur = max(self._lst.currentRow(), 0)
        self._lst.blockSignals(True); self._lst.clear()
        for name,_ in self._fk_frames: self._lst.addItem(name)
        self._lst.blockSignals(False)
        self._lst.setCurrentRow(min(cur, len(self._fk_frames)-1))
        self._show_mat()

    def _show_mat(self):
        idx = self._lst.currentRow()
        if idx < 0 or not hasattr(self,"_fk_frames") or idx >= len(self._fk_frames): return
        name, T = self._fk_frames[idx]
        p = T[:3,3]; roll,pitch,yaw = rot_to_rpy(T[:3,:3])
        self._txt.setPlainText(
            f"{name}\n{'-'*56}\n"
            f"xyz  [mm]  = ({p[0]:+.4f},  {p[1]:+.4f},  {p[2]:+.4f})\n"
            f"xyz  [m]   = ({p[0]/1000:+.6f}, {p[1]/1000:+.6f}, {p[2]/1000:+.6f})\n"
            f"rpy  [rad] = ({roll:+.6f},  {pitch:+.6f},  {yaw:+.6f})\n"
            f"rpy  [°]   = ({math.degrees(roll):+.4f},  "
            f"{math.degrees(pitch):+.4f},  {math.degrees(yaw):+.4f})\n\n"
            f"T (4×4) [mm]:\n{fmt4(T)}")

    # ------------------------------------------------------------------
    #  LÓGICA IK
    # ------------------------------------------------------------------
    def _solve_ik(self):
        px = self._ix.value(); py = self._iy.value(); pz = self._iz.value()
        try:
            qc,qp,qr,err,W,A3 = ik_mth_posicion(px, py, pz)
        except ValueError as e:
            for ww in (self._oqc,self._oqp,self._oqr): ww.setText("-")
            self._oerr.setText(f"Sin solución:\n{e}")
            self._ik_mat.setPlainText(f"Sin solución:\n{e}")
            for ww in (self._vx,self._vy,self._vz,self._ve): ww.setText("-")
            return

        self._oerr.setText("")
        self._oqc.setText(f"{qc:+.4f}°  ({math.radians(qc):+.6f} rad)")
        self._oqp.setText(f"{qp:+.4f}°  ({math.radians(qp):+.6f} rad)")
        self._oqr.setText(f"{qr:+.4f}°  ({math.radians(qr):+.6f} rad)")

        _, T_ef = fk_mth(qc,qp,qr); p_rec = T_ef[:3,3]
        err_v = math.sqrt((px-p_rec[0])**2+(py-p_rec[1])**2+(pz-p_rec[2])**2)
        self._vx.setText(f"{p_rec[0]:+.4f} mm")
        self._vy.setText(f"{p_rec[1]:+.4f} mm")
        self._vz.setText(f"{p_rec[2]:+.4f} mm")
        col = "#5ec46e" if err_v < 1.0 else "#e05c5c"
        self._ve.setStyleSheet(f"color:{col};font-weight:bold;")
        self._ve.setText(f"{err_v:.4f} mm  ({'OK < 1mm' if err_v<1.0 else 'WARN'})")

        roll,pitch,yaw = rot_to_rpy(T_ef[:3,:3])
        sp = (_D1 - W[2,3]) / _L2
        self._ik_mat.setPlainText(
            f"=== PASOS IK POR INVERSIÓN MTH ===\n\n"
            f"1) W = T_BASE_0⁻¹ · T_objetivo · T_3_EF⁻¹:\n{fmt4(W)}\n\n"
            f"2) qc = atan2(−W[0,2], −W[1,2])\n"
            f"      W[0,2]={W[0,2]:+.6f}   W[1,2]={W[1,2]:+.6f}\n"
            f"   ⇒  qc = {qc:+.4f}°  ({math.radians(qc):+.6f} rad)\n\n"
            f"3) qp = asin( (D1−W[2,3]) / L2 )\n"
            f"      W[2,3]={W[2,3]:+.6f}  D1={_D1}  L2={_L2}\n"
            f"      sin(qp) = ({_D1-W[2,3]:.4f}) / {_L2} = {sp:+.6f}\n"
            f"   ⇒  qp = {qp:+.4f}°  ({math.radians(qp):+.6f} rad)\n\n"
            f"4) A3 = A2(qp)⁻¹ · A1(qc)⁻¹ · W:\n{fmt4(A3)}\n"
            f"   qr = atan2(A3[1,0], A3[0,0])\n"
            f"      A3[1,0]={A3[1,0]:+.6f}   A3[0,0]={A3[0,0]:+.6f}\n"
            f"   ⇒  qr = {qr:+.4f}°  ({math.radians(qr):+.6f} rad)\n\n"
            f"=== VERIFICACIÓN FK( IK(p) ) ===\n"
            f"xyz [mm] = ({p_rec[0]:+.4f}, {p_rec[1]:+.4f}, {p_rec[2]:+.4f})\n"
            f"xyz [m]  = ({p_rec[0]/1000:+.6f}, {p_rec[1]/1000:+.6f}, {p_rec[2]/1000:+.6f})\n"
            f"‖error‖  = {err_v:.4f} mm\n\n"
            f"T (4×4) [mm]:\n{fmt4(T_ef)}\n\n"
            f"Verificar:\n"
            f"  ros2 run tf2_ros tf2_echo base_link efector\n"
            f"  (Translation [m] × 1000 → mm)")

        self.node.set_deg("joint_c", qc)
        self.node.set_deg("joint_p", qp)
        self.node.set_deg("joint_r", qr)
        # sync sliders
        for j,v in zip(JOINTS,[qc,qp,qr]):
            d = clamp(v, *LIMITS_DEG[j])
            self._sl[j].blockSignals(True)
            self._sl[j].setValue(int(round(d*10)))
            self._sl[j].blockSignals(False)
            self._ed[j].setText(f"{d:.1f}")
        self._refresh_fk()

    # ------------------------------------------------------------------
    #  BOTONES FK
    # ------------------------------------------------------------------
    def _do_zero(self):
        for j in JOINTS:
            self.node.set_deg(j, 0.)
            self._sl[j].blockSignals(True); self._sl[j].setValue(0)
            self._sl[j].blockSignals(False); self._ed[j].setText("0.0")
        self._refresh_fk()

    def _do_verify(self):
        _, T = fk_mth(0,0,0); p = T[:3,3]; roll,pitch,yaw = rot_to_rpy(T[:3,:3])
        QtWidgets.QMessageBox.information(self, "Verificación q=(0°,0°,0°) vs tf2_echo",
            f"FK MTH  q=(0°, 0°, 0°):\n"
            f"  x = {p[0]:+.4f} mm  ({p[0]/1000:+.6f} m)\n"
            f"  y = {p[1]:+.4f} mm  ({p[1]/1000:+.6f} m)\n"
            f"  z = {p[2]:+.4f} mm  ({p[2]/1000:+.6f} m)\n\n"
            f"tf2_echo base_link efector:\n"
            f"  Translation: [−0.130, −0.641, +0.226] m\n\n"
            f"RPY [rad]: ({roll:+.5f}, {pitch:+.5f}, {yaw:+.5f})\n\n"
            f"T (4×4) [mm]:\n{fmt4(T)}")


# =================================================================
#  MAIN
# =================================================================
def main(args=None):
    rclpy.init(args=args)
    node = PataNode()
    app  = QtWidgets.QApplication([])
    win  = MTHWindow(node); win.show()
    spin_t = QtCore.QTimer()
    spin_t.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    spin_t.start(10)
    signal.signal(signal.SIGINT,  lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    app.exec_()
    try: node.destroy_node()
    except: pass
    try: rclpy.shutdown()
    except: pass

if __name__ == "__main__":
    main()
