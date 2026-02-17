# app.py
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from qiskit.visualization import circuit_drawer

from ops import OPERATIONS
from simulator import (
    parse_complex,
    build_basis_state_for_AB,
    build_product_state_for_AB,
    format_statevector_like_screenshot,
    make_prob_figure_nonzero,
    bitstr_q0_to_qN_from_qiskit_str,
)

# =================================================
# Circuit drawing helpers (same behavior)
# =================================================
def add_wire_legend_in_window_corner(fig):
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=2, label="A (input)"),
        Line2D([0], [0], color="#d62728", lw=2, label="B (input)"),
        Line2D([0], [0], color="#2ca02c", lw=2, label="Outputs/ancilla"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        bbox_transform=fig.transFigure,
        fontsize=8,
        handlelength=1.4,
        borderpad=0.3,
        labelspacing=0.25,
        frameon=True,
        framealpha=0.92,
    )

def _find_stage_marker_x(ax, prefix="STAGE_"):
    xs = []
    for t in ax.texts:
        s = t.get_text()
        if isinstance(s, str) and s.startswith(prefix):
            x, _y = t.get_position()
            xs.append(float(x))

    xs = sorted(xs)

    out = []
    for x in xs:
        if not out or abs(x - out[-1]) > 0.2:
            out.append(x)
    return out


def add_stage_boxes_from_markers(ax, n, prefix="STAGE_"):
    xs = _find_stage_marker_x(ax, prefix=prefix)

    # expecting STAGE_0, STAGE_1..STAGE_n, STAGE_END  => n+2 markers
    if len(xs) < n + 2:
        return

    ymin, ymax = ax.get_ylim()
    y0 = min(ymin, ymax)
    y1 = max(ymin, ymax)

    starts = xs[1:1+n]   # STAGE_1..STAGE_n
    ends   = xs[2:2+n]   # STAGE_2..STAGE_END

    for i, (x0, x1) in enumerate(zip(starts, ends), start=1):
        if x1 <= x0:
            continue

        pad_x = 0.25
        rect = Rectangle(
            (x0 + pad_x, y0 + 0.15),
            (x1 - x0) - 2 * pad_x,
            (y1 - y0) - 0.30,
            fill=False,
            linewidth=1.8,
            linestyle=(0, (3, 3)),
            edgecolor="#444444",
        )
        ax.add_patch(rect)
        ax.text(
            (x0 + x1) / 2,
            y1 - 0.05,
            f"Stage {i}",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )

def make_grouped_circuit_figure(qc, n):
    style = {
        "backgroundcolor": "white",
        "linecolor": "#111111",
        "textcolor": "#111111",
        "fontsize": 11,
        "subfontsize": 9,
        "dpi": 160,
        "barrierfacecolor": "#f7f7f7",
        "barrieredgecolor": "#d0d0d0",
    }
    fig = plt.Figure(figsize=(20, 6), dpi=120)
    ax = fig.add_subplot(111)
    circuit_drawer(qc, output="mpl", ax=ax, fold=-1, plot_barriers=True, style=style)
    fig.subplots_adjust(left=0.04, right=0.83, top=0.96, bottom=0.06)
    add_wire_legend_in_window_corner(fig)
    add_stage_boxes_from_markers(ax, n)

    return fig

# =================================================
# Windows (same behavior)
# =================================================
class CircuitWindow(tk.Toplevel):
    def __init__(self, parent, qc, n, title):
        super().__init__(parent)
        self.title(title)
        self.geometry("1200x450")
        fig = make_grouped_circuit_figure(qc, n)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

class ResultsWindow(tk.Toplevel):
    def __init__(self, parent, summary_text, init_sv, final_sv, amp_tol):
        super().__init__(parent)
        self.title("Simulation Results (Statevectors + Decoding)")
        self.geometry("1200x850")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        tab1 = ttk.Frame(notebook, padding=8)
        notebook.add(tab1, text="Summary")
        txt1 = tk.Text(tab1, height=28, wrap="word")
        txt1.insert("1.0", summary_text)
        txt1.configure(state="disabled")
        txt1.pack(fill="both", expand=True)

        tab2 = ttk.Frame(notebook, padding=8)
        notebook.add(tab2, text="Initial Statevector")
        txt2 = tk.Text(tab2, wrap="none")
        txt2.insert("1.0", format_statevector_like_screenshot(init_sv, "Symbolic INITIAL statevector", amp_tol=amp_tol))
        txt2.configure(state="disabled")
        txt2.pack(fill="both", expand=True)

        tab3 = ttk.Frame(notebook, padding=8)
        notebook.add(tab3, text="Final Statevector")
        txt3 = tk.Text(tab3, wrap="none")
        txt3.insert("1.0", format_statevector_like_screenshot(final_sv, "Symbolic FINAL statevector", amp_tol=amp_tol))
        txt3.configure(state="disabled")
        txt3.pack(fill="both", expand=True)

class ProbabilityWindow(tk.Toplevel):
    def __init__(self, parent, fig, window_title):
        super().__init__(parent)
        self.title(window_title)
        self.geometry("1200x450")
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# =================================================
# App (keeps your amplitude UI exactly)
# =================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("n-bit Quantum Arithmetic Tool")

        self.n_var = tk.StringVar(value="4")
        self.mode_var = tk.StringVar(value="Decimal")
        self.op_key_var = tk.StringVar(value="sub")  # default operation
        self.amp_tol_var = tk.StringVar(value="1e-12")
        self.topk_var = tk.StringVar(value="64")

        self._open_windows = []

        self._build_ui()
        self._rebuild_mode_inputs()

    # ---- window tracking
    def _register_window(self, win):
        self._open_windows.append(win)
        win.bind("<Destroy>", lambda e, w=win: self._forget_window(w))

    def _forget_window(self, win):
        self._open_windows = [w for w in self._open_windows if w is not win]

    def _close_previous_results(self):
        for w in list(self._open_windows):
            try:
                if w.winfo_exists():
                    w.destroy()
            except Exception:
                pass
        self._open_windows = []

    # ---- settings getters
    def _get_n(self):
        n = int(self.n_var.get().strip())
        if n < 1:
            raise ValueError
        return n

    def _get_amp_tol(self):
        tol = float(self.amp_tol_var.get().strip())
        if tol <= 0:
            raise ValueError
        return tol

    def _get_topk(self):
        k = int(self.topk_var.get().strip())
        if k < 1:
            raise ValueError
        return k

    # ---- UI builders
    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        row0 = ttk.Frame(outer)
        row0.pack(fill="x")

        ttk.Label(row0, text="Operation:").pack(side="left")
        self._op_keys = list(OPERATIONS.keys())
        self._op_labels = [f"{k}: {OPERATIONS[k].name}" for k in self._op_keys]
        self.op_combo = ttk.Combobox(row0, state="readonly", values=self._op_labels, width=30)
        # set default
        try:
            self.op_combo.current(self._op_keys.index(self.op_key_var.get()))
        except Exception:
            self.op_combo.current(0)
            self.op_key_var.set(self._op_keys[0])

        self.op_combo.bind("<<ComboboxSelected>>", self._on_op_selected)
        self.op_combo.pack(side="left", padx=(6, 14))

        ttk.Label(row0, text="n (bits):").pack(side="left")
        ttk.Entry(row0, textvariable=self.n_var, width=10).pack(side="left", padx=(6, 12))
        ttk.Button(row0, text="Apply n", command=self._rebuild_mode_inputs).pack(side="left")

        rowcfg = ttk.Frame(outer)
        rowcfg.pack(fill="x", pady=(8, 0))
        ttk.Label(rowcfg, text="Amplitude tol (nonzero):").pack(side="left")
        ttk.Entry(rowcfg, textvariable=self.amp_tol_var, width=12).pack(side="left", padx=(6, 18))
        ttk.Label(rowcfg, text="Top-K nonzero (plots):").pack(side="left")
        ttk.Entry(rowcfg, textvariable=self.topk_var, width=8).pack(side="left", padx=(6, 0))

        row1 = ttk.LabelFrame(outer, text="Input Mode")
        row1.pack(fill="x", pady=(10, 0))
        for m in ["Decimal", "Probability", "Amplitude"]:
            ttk.Radiobutton(row1, text=m, value=m, variable=self.mode_var, command=self._rebuild_mode_inputs)\
                .pack(side="left", padx=10, pady=6)

        self.dynamic = ttk.Frame(outer)
        self.dynamic.pack(fill="both", expand=True, pady=(10, 0))

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_row, text="Run Simulation", command=self.run_simulation).pack(side="right")

    def _on_op_selected(self, _evt):
        idx = self.op_combo.current()
        if 0 <= idx < len(self._op_keys):
            self.op_key_var.set(self._op_keys[idx])

    def _clear_dynamic(self):
        for w in self.dynamic.winfo_children():
            w.destroy()

    # ---- amplitude prob labels (kept)
    def _update_amp_prob_labels(self):
        if not hasattr(self, "Aa"):
            return

        for i in range(len(self.Aa)):
            try:
                a_alpha = parse_complex(self.Aa[i].get())
                a_beta  = parse_complex(self.Ab[i].get())
                b_alpha = parse_complex(self.Ba[i].get())
                b_beta  = parse_complex(self.Bb[i].get())

                a0 = (a_alpha.real*a_alpha.real + a_alpha.imag*a_alpha.imag)
                a1 = (a_beta.real*a_beta.real + a_beta.imag*a_beta.imag)
                b0 = (b_alpha.real*b_alpha.real + b_alpha.imag*b_alpha.imag)
                b1 = (b_beta.real*b_beta.real + b_beta.imag*b_beta.imag)

                at = a0 + a1
                bt = b0 + b1
                if at > 0:
                    a0 /= at; a1 /= at
                if bt > 0:
                    b0 /= bt; b1 /= bt

                self.A_prob_lbl[i].config(text=f"P0={a0:.3f}, P1={a1:.3f}")
                self.B_prob_lbl[i].config(text=f"P0={b0:.3f}, P1={b1:.3f}")
            except Exception:
                self.A_prob_lbl[i].config(text="P0=?, P1=?")
                self.B_prob_lbl[i].config(text="P0=?, P1=?")

    # ---- randomizers (kept)
    def _rand_qubit_state_real(self):
        p = np.random.rand()
        a = np.sqrt(1.0 - p)
        b = np.sqrt(p)
        if np.random.rand() < 0.5:
            a = -a
        if np.random.rand() < 0.5:
            b = -b
        return (f"{a:.8f}", f"{b:.8f}")

    def _rand_qubit_state_complex(self):
        p = np.random.rand()
        ra = np.sqrt(1.0 - p)
        rb = np.sqrt(p)
        phi_a = 2.0 * np.pi * np.random.rand()
        phi_b = 2.0 * np.pi * np.random.rand()
        a = ra * (np.cos(phi_a) + 1j * np.sin(phi_a))
        b = rb * (np.cos(phi_b) + 1j * np.sin(phi_b))
        return (f"{a.real:.8f}{a.imag:+.8f}j", f"{b.real:.8f}{b.imag:+.8f}j")

    def randomize_amplitudes(self, use_complex=False):
        if not hasattr(self, "Aa"):
            messagebox.showinfo("Not in Amplitude mode", "Switch to Amplitude mode first.")
            return

        maker = self._rand_qubit_state_complex if use_complex else self._rand_qubit_state_real

        for i in range(len(self.Aa)):
            a_alpha, a_beta = maker()
            b_alpha, b_beta = maker()
            self.Aa[i].set(a_alpha)
            self.Ab[i].set(a_beta)
            self.Ba[i].set(b_alpha)
            self.Bb[i].set(b_beta)

        self._update_amp_prob_labels()

    # ---- rebuild mode inputs (kept + modular-safe)
    def _rebuild_mode_inputs(self):
        self._clear_dynamic()
        try:
            n = self._get_n()
        except Exception:
            messagebox.showerror("Invalid n", "Please enter a valid integer n >= 1.")
            return

        mode = self.mode_var.get()

        if mode == "Decimal":
            frm = ttk.LabelFrame(self.dynamic, text="Decimal inputs (A, B)")
            frm.pack(fill="x", padx=2, pady=2)

            self.A_dec = tk.StringVar(value="6")
            self.B_dec = tk.StringVar(value="3")

            grid = ttk.Frame(frm)
            grid.pack(fill="x", padx=8, pady=8)

            ttk.Label(grid, text="A (decimal):").grid(row=0, column=0, sticky="w")
            ttk.Entry(grid, textvariable=self.A_dec, width=18).grid(row=0, column=1, padx=8)

            ttk.Label(grid, text="B (decimal):").grid(row=1, column=0, sticky="w")
            ttk.Entry(grid, textvariable=self.B_dec, width=18).grid(row=1, column=1, padx=8)

            ttk.Label(frm, text="Valid range: 0 .. %d" % ((1 << n) - 1), foreground="#444")\
                .pack(anchor="w", padx=8, pady=(0, 8))

        elif mode == "Probability":
            frm = ttk.LabelFrame(self.dynamic, text="Per-bit probabilities: sqrt(1-p)|0> + sqrt(p)|1>")
            frm.pack(fill="both", expand=True, padx=2, pady=2)

            canvas = tk.Canvas(frm, height=240)
            scrollbar = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview)
            inner = ttk.Frame(canvas)

            inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            ttk.Label(inner, text="bit i (LSB=i)").grid(row=0, column=0, padx=6, pady=4)
            ttk.Label(inner, text="p(Ai=1)").grid(row=0, column=1, padx=6, pady=4)
            ttk.Label(inner, text="p(Bi=1)").grid(row=0, column=2, padx=6, pady=4)

            self.pA, self.pB = [], []
            for i in range(n):
                pA_var = tk.StringVar(value="0")
                pB_var = tk.StringVar(value="0")
                self.pA.append(pA_var); self.pB.append(pB_var)
                ttk.Label(inner, text=str(i)).grid(row=i + 1, column=0, padx=6, pady=2, sticky="w")
                ttk.Entry(inner, textvariable=pA_var, width=10).grid(row=i + 1, column=1, padx=6, pady=2)
                ttk.Entry(inner, textvariable=pB_var, width=10).grid(row=i + 1, column=2, padx=6, pady=2)

        else:  # Amplitude (EXACT UI you have)
            frm = ttk.LabelFrame(self.dynamic, text="Per-bit amplitudes (complex): α,β for each Ai and Bi")
            frm.pack(fill="both", expand=True, padx=2, pady=2)

            canvas = tk.Canvas(frm, height=240)
            scrollbar = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview)
            inner = ttk.Frame(canvas)

            inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            headers = ["bit", "A α", "A β", "A probs", "B α", "B β", "B probs"]
            for j, h in enumerate(headers):
                ttk.Label(inner, text=h).grid(row=0, column=j, padx=6, pady=4)

            self.Aa, self.Ab, self.Ba, self.Bb = [], [], [], []
            self.A_prob_lbl = []
            self.B_prob_lbl = []

            for i in range(n):
                Aa = tk.StringVar(value="1")
                Ab = tk.StringVar(value="0")
                Ba = tk.StringVar(value="1")
                Bb = tk.StringVar(value="0")
                self.Aa.append(Aa); self.Ab.append(Ab); self.Ba.append(Ba); self.Bb.append(Bb)

                ttk.Label(inner, text=str(i)).grid(row=i + 1, column=0, padx=6, pady=2, sticky="w")

                ttk.Entry(inner, textvariable=Aa, width=14).grid(row=i + 1, column=1, padx=6, pady=2)
                ttk.Entry(inner, textvariable=Ab, width=14).grid(row=i + 1, column=2, padx=6, pady=2)

                Apl = ttk.Label(inner, text="P0=1.000, P1=0.000", width=16)
                Apl.grid(row=i + 1, column=3, padx=6, pady=2, sticky="w")
                self.A_prob_lbl.append(Apl)

                ttk.Entry(inner, textvariable=Ba, width=14).grid(row=i + 1, column=4, padx=6, pady=2)
                ttk.Entry(inner, textvariable=Bb, width=14).grid(row=i + 1, column=5, padx=6, pady=2)

                Bpl = ttk.Label(inner, text="P0=1.000, P1=0.000", width=16)
                Bpl.grid(row=i + 1, column=6, padx=6, pady=2, sticky="w")
                self.B_prob_lbl.append(Bpl)

                # auto-update prob labels while typing
                Aa.trace_add("write", lambda *_: self._update_amp_prob_labels())
                Ab.trace_add("write", lambda *_: self._update_amp_prob_labels())
                Ba.trace_add("write", lambda *_: self._update_amp_prob_labels())
                Bb.trace_add("write", lambda *_: self._update_amp_prob_labels())

            btns = ttk.Frame(frm)
            btns.pack(fill="x", padx=8, pady=(6, 8))

            ttk.Button(btns, text="Randomize α/β (real)", command=lambda: self.randomize_amplitudes(use_complex=False))\
                .pack(side="left")

            ttk.Button(btns, text="Randomize α/β (complex phase)", command=lambda: self.randomize_amplitudes(use_complex=True))\
                .pack(side="left", padx=(8, 0))

            self._update_amp_prob_labels()

    # ---- decimal input validation (kept)
    def _validate_decimal_fit(self, n, A, B):
        if A < 0 or B < 0:
            messagebox.showerror("Invalid input", "A and B must be >= 0.")
            return False
        max_val = (1 << n) - 1
        if A > max_val or B > max_val:
            messagebox.showerror(
                "Invalid input",
                "n=%d can represent only 0..%d.\nYou entered A=%d, B=%d.\n\n"
                "Increase n to at least %d bits, or lower A/B."
                % (n, max_val, A, B, max(A, B).bit_length())
            )
            return False
        return True

    # ---- Run simulation (now operation-driven)
    def run_simulation(self):
        self._close_previous_results()

        try:
            n = self._get_n()
            amp_tol = self._get_amp_tol()
            top_k = self._get_topk()
        except Exception:
            messagebox.showerror("Invalid settings", "Check n, tolerance, and Top-K fields.")
            return

        op_key = self.op_key_var.get()
        spec = OPERATIONS.get(op_key)
        if spec is None:
            messagebox.showerror("Operation error", "Unknown operation selected.")
            return

        # Build circuit (mul might not be implemented yet)
        try:
            qc = spec.build_circuit(n)
        except NotImplementedError as e:
            messagebox.showinfo("Not implemented", str(e))
            return
        except Exception as e:
            messagebox.showerror("Circuit build error", str(e))
            return

        mode = self.mode_var.get()

        try:
            if mode == "Decimal":
                A = int(self.A_dec.get().strip())
                B = int(self.B_dec.get().strip())
                if not self._validate_decimal_fit(n, A, B):
                    return
                init_sv = build_basis_state_for_AB(n, A, B, qc)
                expected = spec.expected_decimal(A, B, n) if spec.expected_decimal else None

            elif mode == "Probability":
                amps_A, amps_B = [], []
                for i in range(n):
                    pAi = float(self.pA[i].get().strip())
                    pBi = float(self.pB[i].get().strip())
                    if not (0.0 <= pAi <= 1.0) or not (0.0 <= pBi <= 1.0):
                        raise ValueError("Probabilities must be between 0 and 1.")
                    amps_A.append((np.sqrt(1 - pAi), np.sqrt(pAi)))
                    amps_B.append((np.sqrt(1 - pBi), np.sqrt(pBi)))
                init_sv = build_product_state_for_AB(n, amps_A, amps_B, qc)
                expected = None

            else:  # Amplitude
                amps_A, amps_B = [], []
                for i in range(n):
                    amps_A.append((parse_complex(self.Aa[i].get()), parse_complex(self.Ab[i].get())))
                    amps_B.append((parse_complex(self.Ba[i].get()), parse_complex(self.Bb[i].get())))
                init_sv = build_product_state_for_AB(n, amps_A, amps_B, qc)
                expected = None

        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return

        final_sv = init_sv.evolve(qc)

        probs_final = final_sv.probabilities_dict()
        top_state_qiskit = max(probs_final, key=probs_final.get)  # q(N-1)..q0
        top_state_disp = bitstr_q0_to_qN_from_qiskit_str(top_state_qiskit)  # q0..qN

        decoded = spec.decode_q0_basis(n, qc, top_state_disp)

        Nq = qc.num_qubits
        lines = [
            "Operation: %s" % spec.name,
            "Mode: %s" % mode,
            "n = %d" % n,
            "Total qubits N = %d" % Nq,
            "State dimension = 2^N = %d" % (2 ** Nq),
            "",
            "Amplitude tol (nonzero) = %g" % amp_tol,
            "Top-K nonzero (plots) = %d" % top_k,
            "",
            "Most probable FINAL basis state |q0...qN> = %s" % top_state_disp,
            "Decoded from that state:",
        ]

        for k in sorted(decoded.keys()):
            val = decoded[k]
            if k in ("A_out", "B_out"):
                lines.append("  %s = %d (bin %s)" % (k, val, format(val, "0%db" % n)))
            else:
                # outputs may have varying widths; keep simple integer print
                lines.append("  %s = %d" % (k, val))

        if mode == "Decimal" and spec.expected_decimal and expected is not None:
            lines.append("")
            lines.append("Expected:")
            for k in sorted(expected.keys()):
                val = expected[k]
                if k in ("Diff_out", "Sum_out"):
                    lines.append("  %s = %d (bin %s)" % (k, val, format(val, "0%db" % n)))
                else:
                    lines.append("  %s = %d" % (k, val))

            passed = True
            # check outputs match expected
            for k, v in expected.items():
                if decoded.get(k) != v:
                    passed = False
            # ensure inputs survived (your circuits are reversible in that sense)
            if decoded.get("A_out") != A or decoded.get("B_out") != B:
                passed = False

            lines.append("")
            lines.append("PASS (deterministic check): %s" % passed)

        summary_text = "\n".join(lines)

        w_circ = CircuitWindow(self, qc, n, "%s (grouped circuit)" % spec.name); self._register_window(w_circ)
        w_res = ResultsWindow(self, summary_text, init_sv, final_sv, amp_tol=amp_tol); self._register_window(w_res)

        fig_init = make_prob_figure_nonzero(init_sv, "Initial state probabilities (nonzero)", amp_tol, top_k, plt)
        w_p1 = ProbabilityWindow(self, fig_init, "Initial State Probabilities (Nonzero)"); self._register_window(w_p1)

        fig_final = make_prob_figure_nonzero(final_sv, "Final state probabilities (nonzero)", amp_tol, top_k, plt)
        w_p2 = ProbabilityWindow(self, fig_final, "Final State Probabilities (Nonzero)"); self._register_window(w_p2)

if __name__ == "__main__":
    app = App()
    app.mainloop()
