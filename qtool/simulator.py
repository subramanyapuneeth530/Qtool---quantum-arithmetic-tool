# simulator.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit import Gate, QuantumCircuit

import numpy as np

def add_stage_marker(qc, qubit, label: str):
    """Insert a visible identity marker gate for plotting stage boxes."""
    qc.unitary(np.eye(2), [qubit], label=label)


def bitstr_q0_to_qN(idx, n_qubits):
    # LEFTMOST is q0
    return format(idx, "0%db" % n_qubits)[::-1]

def bitstr_q0_to_qN_from_qiskit_str(bitstr_qN_to_q0):
    # Qiskit dict keys are usually q(N-1)..q0
    return bitstr_qN_to_q0[::-1]

def normalize(vec, tol=1e-12):
    norm = np.linalg.norm(vec)
    if norm < tol:
        raise ValueError("Vector norm is ~0; cannot normalize.")
    return vec / norm

def parse_complex(s):
    s = s.strip().replace(" ", "")
    s = s.replace("i", "j")
    return complex(s)

def build_basis_state_for_AB(n, A_val, B_val, qc):
    # assumes qc qubits are [A(0..n-1), B(n..2n-1), ...outputs...]
    prep = QuantumCircuit(qc.num_qubits)
    for i in range(n):
        if (A_val >> i) & 1:
            prep.x(i)
        if (B_val >> i) & 1:
            prep.x(n + i)
    return Statevector.from_instruction(prep)

def build_product_state_for_AB(n, amps_A, amps_B, qc):
    # same idea as your current build_product_state_for_registers, but generic for any op
    if len(amps_A) != n or len(amps_B) != n:
        raise ValueError("amps_A and amps_B must have length n")

    total_qubits = qc.num_qubits
    single_states = []

    # A qubits
    for (alpha, beta) in amps_A:
        v = normalize(np.array([complex(alpha), complex(beta)], dtype=complex))
        single_states.append(v)

    # B qubits
    for (alpha, beta) in amps_B:
        v = normalize(np.array([complex(alpha), complex(beta)], dtype=complex))
        single_states.append(v)

    # remaining qubits fixed at |0>
    rest = total_qubits - 2 * n
    if rest < 0:
        raise ValueError("Circuit has fewer than 2n qubits; cannot map A,B.")
    for _ in range(rest):
        single_states.append(np.array([1.0 + 0j, 0.0 + 0j], dtype=complex))

    # Build full vector using kron in forward order (NOT reversed) as in your latest code
    full = np.array([1.0 + 0j])
    for v in single_states:
        full = np.kron(v, full)

    return Statevector(normalize(full))

def format_statevector_like_screenshot(sv, header, amp_tol=1e-12, max_rows=5000, sort_by_prob=True):
    data = np.asarray(sv.data, dtype=complex)
    n_qubits = int(np.log2(data.size))

    lines = []
    lines.append(header)
    lines.append("(basis = |q0 q1 ... q%d>)" % (n_qubits - 1))
    lines.append("-" * 78)

    entries = []
    for idx, amp in enumerate(data):
        if abs(amp) > amp_tol:
            p = (amp.real * amp.real) + (amp.imag * amp.imag)
            entries.append((p, idx, amp))

    if not entries:
        lines.append("(no entries above tolerance)")
        return "\n".join(lines)

    if sort_by_prob:
        entries.sort(key=lambda t: t[0], reverse=True)

    count = 0
    for p, idx, amp in entries:
        bitstr = bitstr_q0_to_qN(idx, n_qubits)
        lines.append("|%s> : %s    (prob=%s)" % (bitstr, amp, p))
        count += 1
        if count >= max_rows:
            lines.append("-" * 78)
            lines.append("(truncated: showing first %d entries)" % max_rows)
            break

    return "\n".join(lines)

def nice_ymax(max_val):
    if max_val <= 0:
        return 1.0
    if max_val < 0.01:
        step = 0.001
    elif max_val < 0.1:
        step = 0.01
    else:
        step = 0.1
    ymax = step * np.ceil(max_val / step)
    ymax = ymax + step * 0.05
    return min(max(ymax, step), 1.0)

def make_stage_marker(label: str) -> Gate:
    """
    A 1-qubit identity gate with a visible label.
    Does NOT change the quantum state; only helps visualization.
    """
    g = Gate(name=label, num_qubits=1, params=[])
    qc_def = QuantumCircuit(1, name=label)
    qc_def.id(0)
    g.definition = qc_def
    return g

def make_prob_figure_nonzero(sv, title, amp_tol, top_k, plt):
    data = np.asarray(sv.data, dtype=complex)
    probs = (data.real * data.real) + (data.imag * data.imag)
    n_qubits = int(np.log2(probs.size))

    nz = np.where(probs > (amp_tol * amp_tol))[0]
    if nz.size == 0:
        fig = plt.Figure(figsize=(12, 3.8), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No nonzero states above tolerance.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    if nz.size > top_k:
        nz = nz[np.argsort(probs[nz])[::-1][:top_k]]

    nz = np.sort(nz)
    labels = [bitstr_q0_to_qN(i, n_qubits) for i in nz]
    vals = probs[nz]

    fig = plt.Figure(figsize=(14, 4.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, nice_ymax(float(np.max(vals))))
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig
