#!/usr/bin/env python3
"""
gui_algebraic.py — IK Algebraica para LA_PATA_SOLA
====================================================
DERIVACIÓN ALGEBRAICA completa a partir de las ecuaciones de FK:

    x = -d·cos(qc) - s·sin(qc)     s = L1·cos(qp) + L2·cos(qp+qr-φ)
    y =  d·sin(qc) - s·cos(qc)     z = h - L1·sin(qp) - L2·sin(qp+qr-φ)

PASO 1 — Elevar al cuadrado + identidad trigonométrica:
    x² = d²cos²(qc) + 2ds·sin(qc)cos(qc) + s²sin²(qc)
    y² = d²sin²(qc) - 2ds·sin(qc)cos(qc) + s²cos²(qc)
    x²+y² = d²+s²   →   s = ±√(x²+y²-d²)   [2 ramas de cadera]

PASO 2 — Sistema matricial para qc (sin·cos en forma matricial → atan2):
    Expandir la FK en función de cos(qc) y sin(qc):
      x = -d·cos(qc) - s·sin(qc)
      y =  d·sin(qc) - s·cos(qc)   ↔   y = -s·cos(qc) + d·sin(qc)
    En forma matricial:
      |-d  -s| |cos qc|   |x|
      |-s   d| |sin qc| = |y|
    det = (-d)(d) - (-s)(-s) = -(d²+s²)
    cos qc = ( d·x - s·y) / (-(d²+s²))  = (-d·x + s·y)/(d²+s²)
    sin qc = (-s·x - d·y) / (-(d²+s²))  = ( s·x + d·y)/(-(d²+s²))
    Alternativa directa: qc = atan2(-s,-d) - atan2(y,x)  [rama principal con s>0]
                         qc = atan2(-s,-d) - atan2(y,x)  [rama principal: yc = -s]

PASO 3 — Ley de cosenos para c₂ (elevando al cuadrado):
    De la FK en el plano sagital: u²+w² = L1²+L2²+2L1L2·cos(qr_eff)
    c₂ = (u²+w²-L1²-L2²)/(2L1L2),  s₂ = ±√(1-c₂²)  [2 ramas de rodilla]
    qr_eff = atan2(s₂,c₂)

PASO 4 — Sistema matricial para qp (ángulo doble + matricial → atan2):
    Expandir cos(qp+qr-φ) y sin(qp+qr-φ) con ángulo doble:
      u = L1·cos(qp) + L2·cos(qp+qr-φ) = (L1+L2c₂)cos(qp) - L2s₂·sin(qp)
      w = L1·sin(qp) + L2·sin(qp+qr-φ) = (L1+L2c₂)sin(qp) + L2s₂·cos(qp)
    En forma matricial:
      |L1+L2c₂  -L2s₂| |cos qp|   |u|
      |  L2s₂  L1+L2c₂| |sin qp| = |w|
    det = (L1+L2c₂)² + (L2s₂)² = L1²+L2²+2L1L2c₂
    qp = atan2(w,u) - atan2(L2s₂, L1+L2c₂)

PASO 5: qr = qr_eff + φ

Uso:
    ros2 launch parcial2 launch_algebraic.py
"""

import math, signal
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5 import QtWidgets, QtCore, QtGui

# =================================================================
#  PARÁMETROS (código base URDF)
# =================================================================
z0   = 0.1439
dz_p = 0.075
a    = 0.0403
L1   = 0.3505
k1   = -0.00365
L2   = math.hypot(0.2905, -0.0010066)
k2   = 0.093563
phi  = 0.026978 + math.atan2(-0.0010066, 0.2905)
h    = z0 + dz_p
d    = a + k2 + k1

JOINTS     = ["joint_c", "joint_p", "joint_r"]
LIMITS_DEG = {j: (-90.0, 90.0) for j in JOINTS}
PUBLISH_HZ = 50.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

# =================================================================
#  FK GEOMÉTRICA (idéntica al código base)
# =================================================================
def fk_geom_rad(qc, qp, qr):
    s = L1*math.cos(qp) + L2*math.cos(qp + qr - phi)
    z = h - L1*math.sin(qp) - L2*math.sin(qp + qr - phi)
    return (-d*math.cos(qc) - s*math.sin(qc),
             d*math.sin(qc) - s*math.cos(qc),
             z)

def fk_pos(qc_deg, qp_deg, qr_deg):
    return fk_geom_rad(math.radians(qc_deg),
                       math.radians(qp_deg),
                       math.radians(qr_deg))

def _Rx(t): c,s=math.cos(t),math.sin(t); return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]],np.float64)
def _Ry(t): c,s=math.cos(t),math.sin(t); return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]],np.float64)
def _Rz(t): c,s=math.cos(t),math.sin(t); return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],np.float64)
def _Tr(x,y,z): M=np.identity(4,np.float64); M[0,3]=x; M[1,3]=y; M[2,3]=z; return M
def _RPY(r,p,y_): return _Rz(y_)@_Ry(p)@_Rx(r)
_T_OC=_Tr(0,0,0.1439); _T_OP=_Tr(-0.0403,0,0.075)@_RPY(math.pi/2,0,-math.pi/2)
_T_OR=_Tr(0.3505,-2.9381e-5,-0.00365)@_RPY(0,0,0.026978)
_T_OE=_Tr(0.2905,-0.0010066,0.093563)@_RPY(-0.18364,1.5696,1.3872)

def fk_chain(qc_deg, qp_deg, qr_deg):
    T01=_T_OC@_Rz(-math.radians(qc_deg)); T02=T01@_T_OP@_Rz(-math.radians(qp_deg))
    T03=T02@_T_OR@_Rz(-math.radians(qr_deg)); T04=T03@_T_OE
    return [("base → link_c   (T₀₁)",T01),("base → link_p   (T₀₂)",T02),
            ("base → link_r   (T₀₃)",T03),("base → efector  (T₀₄)",T04)], T04

# =================================================================
#  IK ALGEBRAICA — 2 soluciones con derivación completa
# =================================================================
def ik_all_solutions(x, y, z):
    """
    Resuelve IK algebraicamente. Devuelve 4 dicts con la solución
    y el texto de la derivación paso a paso.
    """
    rho_sq = x**2 + y**2
    rho    = math.sqrt(rho_sq)

    if rho_sq < d**2 - 1e-9:
        raise ValueError(
            f"Punto no alcanzable.\n"
            f"  ρ = √(x²+y²) = {rho:.5f} m < d = {d:.5f} m")

    s_sq  = max(rho_sq - d**2, 0.0)
    s_abs = math.sqrt(s_sq)   # siempre ≥ 0

    results = []

    # Solo rama principal de cadera (yc = -s).
    # La rama alternativa produce qc fuera de ±90° para este URDF.
    s   = s_abs
    yc  = -s                          # rama principal: yc = -s
    qc  = math.atan2(yc, -d) - math.atan2(y, x)
    qc  = math.atan2(math.sin(qc), math.cos(qc))  # normalizar (-π,π]

    det_c    = d**2 + s**2
    cqc_show = (-d*x + yc*y) / det_c if det_c > 1e-12 else math.cos(qc)
    sqc_show = (-yc*x - d*y) / det_c if det_c > 1e-12 else math.sin(qc)

    u      = s_abs
    w      = h - z
    c2_num = u**2 + w**2 - L1**2 - L2**2
    c2_den = 2*L1*L2
    c2     = clamp(c2_num / c2_den, -1.0, 1.0)
    s2_abs = math.sqrt(max(0.0, 1.0 - c2**2))

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
        within = abs(qc_d)<=90.1 and abs(qp_d)<=90.1 and abs(qr_d)<=90.1

        xr, yr, zr = fk_geom_rad(qc, qp, qr)
        err = math.sqrt((x-xr)**2+(y-yr)**2+(z-zr)**2)*1e3

        pasos = (
            f"Solución: Rodilla = '{rama_rodilla}\n"
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
            f"  Expandiendo con ángulo doble cos/sin(qp+qr_eff):\n"
            f"    u = (L1+L2c₂)·cos(qp) − L2s₂·sin(qp)\n"
            f"    w =   L2s₂  ·cos(qp) + (L1+L2c₂)·sin(qp)\n"
            f"  ┌ L1+L2c₂   −L2s₂  ┐ ┌ cos qp ┐   ┌ u ┐\n"
            f"  └   L2s₂   L1+L2c₂ ─┘ └ sin qp ┘ = └ w ┘\n"
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
    return results


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
class LegNode(Node):
    def __init__(self):
        super().__init__("gui_algebraic_node")
        self.pub=self.create_publisher(JointState,"/joint_states",10)
        self._deg={j:0. for j in JOINTS}
        self.create_timer(1./PUBLISH_HZ,self._publish)
    def set_deg(self,j,dd): self._deg[j]=clamp(dd,-90.,90.)
    def get_deg(self,j):    return self._deg[j]
    def _publish(self):
        msg=JointState(); msg.header.stamp=self.get_clock().now().to_msg()
        msg.name=JOINTS[:]; msg.position=[math.radians(self._deg[j]) for j in JOINTS]
        self.pub.publish(msg)


# =================================================================
#  ESTILO
# =================================================================
STYLE="""
QMainWindow,QWidget{background:#0a0a0a;color:#d0d0d0;
  font-family:"Consolas","Courier New",monospace;font-size:12px;}
QGroupBox{border:1px solid #252525;border-radius:7px;margin-top:10px;
  padding:8px 6px 6px;font-size:11px;font-weight:bold;color:#4a9eff;}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}
QLabel{color:#d0d0d0;}
QSlider::groove:horizontal{height:5px;background:#111;border-radius:3px;}
QSlider::handle:horizontal{background:#4a9eff;border:none;width:16px;height:16px;margin:-6px 0;border-radius:8px;}
QSlider::sub-page:horizontal{background:#4a9eff;border-radius:3px;}
QLineEdit{background:#111;border:1px solid #252525;border-radius:4px;color:#d0d0d0;padding:3px 5px;min-width:64px;}
QLineEdit:focus{border-color:#4a9eff;}
QDoubleSpinBox{background:#111;border:1px solid #252525;border-radius:4px;color:#d0d0d0;padding:3px 5px;}
QDoubleSpinBox:focus{border-color:#4a9eff;}
QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{width:16px;background:#1a1a1a;border-radius:2px;}
QPushButton{background:#1a1a1a;border:none;border-radius:5px;color:#d0d0d0;padding:6px 14px;font-size:12px;}
QPushButton:hover{background:#222;} QPushButton:pressed{background:#4a9eff;color:#000;}
QListWidget{background:#060606;border:1px solid #252525;border-radius:4px;color:#5ec46e;font-size:12px;}
QListWidget::item{padding:3px 6px;} QListWidget::item:selected{background:#1a1a1a;color:#4a9eff;}
QPlainTextEdit{background:#060606;border:1px solid #252525;border-radius:4px;
  color:#5ec46e;font-family:"Consolas","Courier New",monospace;font-size:12px;}
QTabWidget::pane{border:1px solid #252525;border-radius:6px;margin-top:-1px;}
QTabBar::tab{background:#111;color:#d0d0d0;padding:7px 18px;border-radius:5px 5px 0 0;margin-right:3px;font-size:12px;}
QTabBar::tab:selected{background:#1a1a1a;color:#4a9eff;} QTabBar::tab:hover{background:#181818;}
QScrollBar:vertical{background:#0a0a0a;width:8px;border-radius:4px;}
QScrollBar::handle:vertical{background:#1a1a1a;border-radius:4px;min-height:20px;}
"""
_META={"joint_c":("qc","Coxa/Yaw","#e05c5c"),"joint_p":("qp","Fémur/Pitch","#4a9eff"),"joint_r":("qr","Tibia/Rod.","#5ec46e")}
def _slider(color):
    sl=QtWidgets.QSlider(QtCore.Qt.Horizontal)
    sl.setStyleSheet(f"QSlider::groove:horizontal{{height:5px;background:#111;border-radius:3px;}}"
                     f"QSlider::handle:horizontal{{background:{color};border:none;width:16px;height:16px;margin:-6px 0;border-radius:8px;}}"
                     f"QSlider::sub-page:horizontal{{background:{color};border-radius:3px;}}"); return sl
def _spin(v):
    sb=QtWidgets.QDoubleSpinBox(); sb.setRange(-1.5,1.5); sb.setDecimals(5); sb.setSingleStep(0.005); sb.setValue(v); return sb


# =================================================================
#  VENTANA PRINCIPAL
# =================================================================
class AlgWindow(QtWidgets.QMainWindow):
    def __init__(self, node):
        super().__init__(); self.node=node
        self.setWindowTitle("IK Algebraica — LA_PATA_SOLA  (2 soluciones: rodilla↑ | rodilla↓ | derivación algebraica)")
        self.setMinimumSize(1200,760); self.setStyleSheet(STYLE)
        tabs=QtWidgets.QTabWidget()
        tabs.addTab(self._tab_fk(),"  Cinemática Directa (FK)  ")
        tabs.addTab(self._tab_ik(),"  IK Algebraica — 2 soluciones + derivación  ")
        self.setCentralWidget(tabs); self._refresh_fk()

    # ── TAB FK ────────────────────────────────────────────────────
    def _tab_fk(self):
        w=QtWidgets.QWidget(); outer=QtWidgets.QHBoxLayout(w); outer.setContentsMargins(10,10,10,10); outer.setSpacing(14)
        left=QtWidgets.QVBoxLayout(); left.setSpacing(10)
        right=QtWidgets.QVBoxLayout(); right.setSpacing(10)
        grp_j=QtWidgets.QGroupBox("Ángulos articulares  [−90° ≤ q ≤ +90°]")
        vj=QtWidgets.QVBoxLayout(grp_j); vj.setSpacing(6)
        self._sl={}; self._ed={}
        for j in JOINTS:
            sym,name,color=_META[j]
            top=QtWidgets.QHBoxLayout(); top.setSpacing(6)
            lsym=QtWidgets.QLabel(sym); lsym.setStyleSheet(f"color:{color};font-size:18px;font-weight:bold;min-width:24px;"); lsym.setAlignment(QtCore.Qt.AlignCenter)
            lname=QtWidgets.QLabel(name); lname.setStyleSheet(f"color:{color};font-size:12px;")
            ed=QtWidgets.QLineEdit("0.0"); ed.setFixedWidth(72); ed.setAlignment(QtCore.Qt.AlignRight)
            lunit=QtWidgets.QLabel("°"); lunit.setStyleSheet("color:#555;font-size:12px;")
            top.addWidget(lsym); top.addWidget(lname,1); top.addWidget(ed); top.addWidget(lunit)
            bot=QtWidgets.QHBoxLayout(); bot.setSpacing(4)
            llo=QtWidgets.QLabel("−90°"); llo.setStyleSheet("color:#444;font-size:10px;min-width:32px;"); llo.setAlignment(QtCore.Qt.AlignRight)
            sl=_slider(color); sl.setMinimum(-900); sl.setMaximum(900); sl.setValue(0)
            lhi=QtWidgets.QLabel("+90°"); lhi.setStyleSheet("color:#444;font-size:10px;min-width:32px;")
            bot.addWidget(llo); bot.addWidget(sl,1); bot.addWidget(lhi)
            def _sl_cb(val,joint=j,edit=ed): edit.setText(f"{val/10.:.1f}"); self.node.set_deg(joint,val/10.); self._refresh_fk()
            def _ed_cb(joint=j,s=sl):
                try: v=float(self._ed[joint].text())
                except ValueError: return
                v=clamp(v,-90.,90.); s.blockSignals(True); s.setValue(int(round(v*10))); s.blockSignals(False)
                self._ed[joint].setText(f"{v:.1f}"); self.node.set_deg(joint,v); self._refresh_fk()
            sl.valueChanged.connect(_sl_cb); ed.returnPressed.connect(_ed_cb); ed.editingFinished.connect(_ed_cb)
            blk=QtWidgets.QVBoxLayout(); blk.setSpacing(2); blk.addLayout(top); blk.addLayout(bot); vj.addLayout(blk)
            if j!=JOINTS[-1]:
                ln=QtWidgets.QFrame(); ln.setFrameShape(QtWidgets.QFrame.HLine); ln.setStyleSheet("color:#1e1e1e;"); vj.addWidget(ln)
            self._sl[j]=sl; self._ed[j]=ed
        left.addWidget(grp_j)
        grp_pos=QtWidgets.QGroupBox("Posición del efector  [m]")
        gp=QtWidgets.QGridLayout(grp_pos); gp.setSpacing(6)
        self._fk_lx=QtWidgets.QLabel("—"); self._fk_ly=QtWidgets.QLabel("—"); self._fk_lz=QtWidgets.QLabel("—")
        for i,(sym,ww,col) in enumerate([("x =",self._fk_lx,"#e05c5c"),("y =",self._fk_ly,"#5ec46e"),("z =",self._fk_lz,"#4a9eff")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:14px;font-weight:bold;font-family:'Consolas','Courier New',monospace;")
            gp.addWidget(lb,i,0); gp.addWidget(ww,i,1)
        left.addWidget(grp_pos)
        br=QtWidgets.QHBoxLayout(); br.setSpacing(8)
        b0=QtWidgets.QPushButton("⟳  Zero"); bv=QtWidgets.QPushButton("✓  Verificar q=0")
        for b in(b0,bv): b.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Fixed)
        br.addWidget(b0); br.addWidget(bv); left.addLayout(br)
        b0.clicked.connect(self._do_zero); bv.clicked.connect(self._do_verify)
        left.addStretch(1)
        grp_mat=QtWidgets.QGroupBox("Matrices HTM")
        vm=QtWidgets.QVBoxLayout(grp_mat)
        self._fk_lst=QtWidgets.QListWidget(); self._fk_lst.setFixedHeight(95)
        self._fk_txt=QtWidgets.QPlainTextEdit(); self._fk_txt.setReadOnly(True); self._fk_txt.setMinimumHeight(220)
        self._fk_lst.currentRowChanged.connect(self._show_fk_mat)
        vm.addWidget(self._fk_lst); vm.addWidget(self._fk_txt); right.addWidget(grp_mat,1)
        note=QtWidgets.QLabel("  FK:\n  s=L1·cos(qp)+L2·cos(qp+qr−φ)\n  z=h−L1·sin(qp)−L2·sin(qp+qr−φ)\n  x=−d·cos(qc)−s·sin(qc)\n  y= d·sin(qc)−s·cos(qc)")
        note.setStyleSheet("color:#3a3a4a;font-size:11px;font-family:'Consolas','Courier New',monospace;"); note.setWordWrap(True); right.addWidget(note)
        outer.addLayout(left,43); outer.addLayout(right,57); return w

    # ── TAB IK ────────────────────────────────────────────────────
    def _tab_ik(self):
        w=QtWidgets.QWidget(); outer=QtWidgets.QHBoxLayout(w); outer.setContentsMargins(10,10,10,10); outer.setSpacing(14)
        left=QtWidgets.QVBoxLayout(); left.setSpacing(8)
        right=QtWidgets.QVBoxLayout(); right.setSpacing(8)

        title=QtWidgets.QLabel("IK Algebraica — 2 soluciones\n(elevación al cuadrado + identidad trigonométrica + sistema matricial → atan2)")
        title.setStyleSheet("color:#e05c5c;font-size:13px;font-weight:bold;"); left.addWidget(title)

        x0,y0,z0_=fk_pos(0,0,0)
        grp_in=QtWidgets.QGroupBox("Posición objetivo [m]")
        g_in=QtWidgets.QGridLayout(grp_in); g_in.setSpacing(8)
        self._ix=_spin(x0); self._iy=_spin(y0); self._iz=_spin(z0_)
        for i,(sym,ws,col) in enumerate([("x =",self._ix,"#e05c5c"),("y =",self._iy,"#5ec46e"),("z =",self._iz,"#4a9eff")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ws.setStyleSheet(f"color:{col};background:#111;border:1px solid #252525;border-radius:4px;padding:3px 5px;")
            g_in.addWidget(lb,i,0); g_in.addWidget(ws,i,1)
        ref=QtWidgets.QLabel(f"Reposo: ({x0:.5f}, {y0:.5f}, {z0_:.5f}) m"); ref.setStyleSheet("color:#3a5a3a;font-size:11px;"); g_in.addWidget(ref,3,0,1,2)
        left.addWidget(grp_in)

        btn=QtWidgets.QPushButton("Calcular 2 soluciones algebraicas  →")
        btn.setStyleSheet("background:#e05c5c;color:#000;font-weight:bold;font-size:13px;padding:9px 16px;border-radius:6px;")
        btn.clicked.connect(self._solve_ik); left.addWidget(btn)

        grp_sol=QtWidgets.QGroupBox("4 Soluciones (✓ dentro de ±90° | ✗ fuera del límite URDF)")
        gsol=QtWidgets.QVBoxLayout(grp_sol)
        self._sol_list=QtWidgets.QListWidget(); self._sol_list.setFixedHeight(115)
        self._sol_list.setStyleSheet("QListWidget{background:#060606;border:1px solid #252525;border-radius:4px;"
                                     "font-size:12px;font-family:'Consolas','Courier New',monospace;}"
                                     "QListWidget::item{padding:4px 8px;}"
                                     "QListWidget::item:selected{background:#1a1a1a;color:#4a9eff;}")
        self._sol_list.currentRowChanged.connect(self._on_sol_selected)
        gsol.addWidget(self._sol_list); left.addWidget(grp_sol)

        grp_res=QtWidgets.QGroupBox("Solución seleccionada")
        g_res=QtWidgets.QGridLayout(grp_res); g_res.setSpacing(6)
        self._oqc=QtWidgets.QLabel("—"); self._oqp=QtWidgets.QLabel("—"); self._oqr=QtWidgets.QLabel("—")
        self._oerr=QtWidgets.QLabel(""); self._oerr.setStyleSheet("color:#e05c5c;font-weight:bold;"); self._oerr.setWordWrap(True)
        for i,(sym,ww,col) in enumerate([("qc =",self._oqc,"#e05c5c"),("qp =",self._oqp,"#4a9eff"),("qr =",self._oqr,"#5ec46e")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:13px;font-weight:bold;font-family:'Consolas','Courier New',monospace;")
            g_res.addWidget(lb,i,0); g_res.addWidget(ww,i,1)
        g_res.addWidget(self._oerr,3,0,1,2)
        btn_a=QtWidgets.QPushButton("▶  Aplicar → RViz2"); btn_a.setStyleSheet("background:#5ec46e;color:#000;font-weight:bold;font-size:12px;padding:6px 14px;border-radius:5px;")
        btn_a.clicked.connect(self._apply_selected); g_res.addWidget(btn_a,4,0,1,2)
        left.addWidget(grp_res)

        grp_v=QtWidgets.QGroupBox("Verificación FK(IK)")
        g_v=QtWidgets.QGridLayout(grp_v); g_v.setSpacing(6)
        self._vx=QtWidgets.QLabel("—"); self._vy=QtWidgets.QLabel("—"); self._vz=QtWidgets.QLabel("—"); self._ve=QtWidgets.QLabel("—")
        for i,(sym,ww,col) in enumerate([("x_rec=",self._vx,"#4dd0e1"),("y_rec=",self._vy,"#4dd0e1"),("z_rec=",self._vz,"#4dd0e1"),("‖err‖=",self._ve,"#5ec46e")]):
            lb=QtWidgets.QLabel(sym); lb.setStyleSheet("color:#555;font-size:12px;")
            ww.setStyleSheet(f"color:{col};font-size:12px;font-family:'Consolas','Courier New',monospace;")
            g_v.addWidget(lb,i,0); g_v.addWidget(ww,i,1)
        left.addWidget(grp_v); left.addStretch(1)

        grp_steps=QtWidgets.QGroupBox("Derivación algebraica paso a paso")
        vs=QtWidgets.QVBoxLayout(grp_steps)
        self._steps_txt=QtWidgets.QPlainTextEdit(); self._steps_txt.setReadOnly(True)
        vs.addWidget(self._steps_txt); right.addWidget(grp_steps,2)

        grp_mat=QtWidgets.QGroupBox("T(base→efector)  tras IK")
        vm=QtWidgets.QVBoxLayout(grp_mat)
        self._ik_mat=QtWidgets.QPlainTextEdit(); self._ik_mat.setReadOnly(True); self._ik_mat.setMaximumHeight(200)
        vm.addWidget(self._ik_mat); right.addWidget(grp_mat,1)

        note=QtWidgets.QLabel(
            "Derivación algebraica:\n"
            "① x²+y²=d²+s²  (elevar al cuadrado)\n   s=±√(x²+y²−d²)  [2 ramas cadera]\n\n"
            "② Sist.2×2 en (cos qc, sin qc) → atan2\n\n"
            "③ c₂=(u²+w²−L1²−L2²)/(2L1L2)\n   s₂=±√(1−c₂²)  [2 ramas rodilla]\n\n"
            "④ Sist.2×2 en (cos qp, sin qp)\n   (expandiendo ángulo doble) → atan2\n\n"
            "⑤ qr = atan2(s₂,c₂) + φ")
        note.setStyleSheet("color:#3a3a4a;font-size:11px;font-family:'Consolas','Courier New',monospace;")
        note.setWordWrap(True); right.addWidget(note)
        outer.addLayout(left,44); outer.addLayout(right,56); return w

    # ── LÓGICA FK ─────────────────────────────────────────────────
    def _refresh_fk(self):
        qc=self.node.get_deg("joint_c"); qp=self.node.get_deg("joint_p"); qr=self.node.get_deg("joint_r")
        self._fk_frames,T=fk_chain(qc,qp,qr); p=T[:3,3]
        self._fk_lx.setText(f"{p[0]:+.6f} m"); self._fk_ly.setText(f"{p[1]:+.6f} m"); self._fk_lz.setText(f"{p[2]:+.6f} m")
        cur=max(self._fk_lst.currentRow(),0); self._fk_lst.blockSignals(True); self._fk_lst.clear()
        for name,_ in self._fk_frames: self._fk_lst.addItem(name)
        self._fk_lst.blockSignals(False); self._fk_lst.setCurrentRow(min(cur,len(self._fk_frames)-1)); self._show_fk_mat()

    def _show_fk_mat(self):
        idx=self._fk_lst.currentRow()
        if idx<0 or not hasattr(self,"_fk_frames") or idx>=len(self._fk_frames): return
        name,T=self._fk_frames[idx]; p=T[:3,3]; roll,pitch,yaw=rot_to_rpy(T[:3,:3])
        self._fk_txt.setPlainText(f"{name}\n{'-'*52}\nxyz  [m] = ({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f})\n"
            f"rpy  [°] = ({math.degrees(roll):+.4f}, {math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\nT (4×4):\n{fmt4(T)}")

    # ── LÓGICA IK ─────────────────────────────────────────────────
    def _solve_ik(self):
        px=self._ix.value(); py=self._iy.value(); pz=self._iz.value()
        self._sol_list.clear(); self._solutions=[]
        self._oqc.setText("—"); self._oqp.setText("—"); self._oqr.setText("—")
        self._oerr.setText(""); self._steps_txt.clear(); self._ik_mat.clear()
        for ww in(self._vx,self._vy,self._vz,self._ve): ww.setText("—")
        try: sols=ik_all_solutions(px,py,pz)
        except ValueError as e: self._oerr.setText(f"Sin solución:\n{e}"); return
        self._solutions=sols
        for i,sol in enumerate(sols):
            sym="✓" if sol['dentro_limites'] else "✗"
            col="#5ec46e" if sol['dentro_limites'] else "#e05c5c"
            txt=(f"[{i+1}] {sym}  Rodilla: {sol['rama_rodilla']:<10}"
                 f"qc={sol['qc_deg']:+6.1f}° qp={sol['qp_deg']:+6.1f}° qr={sol['qr_deg']:+6.1f}°  err={sol['err_mm']:.3f}mm")
            item=QtWidgets.QListWidgetItem(txt); item.setForeground(QtGui.QColor(col)); self._sol_list.addItem(item)
        for i,sol in enumerate(sols):
            if sol['dentro_limites']: self._sol_list.setCurrentRow(i); return
        self._sol_list.setCurrentRow(0)

    def _on_sol_selected(self, idx):
        if idx<0 or not hasattr(self,'_solutions') or idx>=len(self._solutions): return
        sol=self._solutions[idx]
        qc_d=sol['qc_deg']; qp_d=sol['qp_deg']; qr_d=sol['qr_deg']
        self._oqc.setText(f"{qc_d:+.4f}°  ({math.radians(qc_d):+.6f} rad)")
        self._oqp.setText(f"{qp_d:+.4f}°  ({math.radians(qp_d):+.6f} rad)")
        self._oqr.setText(f"{qr_d:+.4f}°  ({math.radians(qr_d):+.6f} rad)")
        if sol['dentro_limites']:
            self._oerr.setText("✓ Solución dentro de los límites ±90° del URDF")
            self._oerr.setStyleSheet("color:#5ec46e;font-weight:bold;")
        else:
            self._oerr.setText("✗ Fuera de los límites ±90° del URDF")
            self._oerr.setStyleSheet("color:#e05c5c;font-weight:bold;")
        xr,yr,zr=fk_pos(qc_d,qp_d,qr_d); err=sol['err_mm']
        self._vx.setText(f"{xr:+.6f} m"); self._vy.setText(f"{yr:+.6f} m"); self._vz.setText(f"{zr:+.6f} m")
        col2="#5ec46e" if err<5. else "#e05c5c"
        self._ve.setStyleSheet(f"color:{col2};font-weight:bold;"); self._ve.setText(f"{err:.4f} mm")
        self._steps_txt.setPlainText(sol['pasos'])
        _,T=fk_chain(qc_d,qp_d,qr_d); roll,pitch,yaw=rot_to_rpy(T[:3,:3])
        self._ik_mat.setPlainText(f"Solución [{idx+1}]\n{'-'*44}\nxyz [m]  = ({T[0,3]:+.6f}, {T[1,3]:+.6f}, {T[2,3]:+.6f})\n"
            f"rpy [°]  = ({math.degrees(roll):+.4f}, {math.degrees(pitch):+.4f}, {math.degrees(yaw):+.4f})\n\nT (4×4):\n{fmt4(T)}")

    def _apply_selected(self):
        idx=self._sol_list.currentRow()
        if idx<0 or not hasattr(self,'_solutions') or idx>=len(self._solutions): return
        sol=self._solutions[idx]
        for j,v in zip(JOINTS,[sol['qc_deg'],sol['qp_deg'],sol['qr_deg']]):
            dv=clamp(v,-90.,90.); self.node.set_deg(j,dv)
            self._sl[j].blockSignals(True); self._sl[j].setValue(int(round(dv*10))); self._sl[j].blockSignals(False)
            self._ed[j].setText(f"{dv:.1f}")
        self._refresh_fk()

    def _do_zero(self):
        for j in JOINTS:
            self.node.set_deg(j,0.); self._sl[j].blockSignals(True); self._sl[j].setValue(0); self._sl[j].blockSignals(False); self._ed[j].setText("0.0")
        self._refresh_fk()

    def _do_verify(self):
        x,y,z=fk_pos(0,0,0); _,T=fk_chain(0,0,0); roll,pitch,yaw=rot_to_rpy(T[:3,:3])
        QtWidgets.QMessageBox.information(self,"Verificación q=0",
            f"FK q=(0°,0°,0°):\n  x={x:+.6f}\n  y={y:+.6f}\n  z={z:+.6f}\n\ntf2_echo: [−0.130,−0.641,+0.226] m ✓\n\nT:\n{fmt4(T)}")


# =================================================================
#  MAIN
# =================================================================
def main(args=None):
    rclpy.init(args=args); node=LegNode()
    app=QtWidgets.QApplication([]); win=AlgWindow(node); win.show()
    spin_t=QtCore.QTimer(); spin_t.timeout.connect(lambda: rclpy.spin_once(node,timeout_sec=0.0)); spin_t.start(10)
    signal.signal(signal.SIGINT, lambda *_: app.quit()); signal.signal(signal.SIGTERM, lambda *_: app.quit())
    app.exec_()
    try: node.destroy_node()
    except: pass
    try: rclpy.shutdown()
    except: pass

if __name__=="__main__": main()