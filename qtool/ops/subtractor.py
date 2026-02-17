# quantum_tool/ops/subtractor.py

from qiskit import QuantumCircuit, QuantumRegister
from .base import OperationSpec

# IMPORTANT: use the correct package name (lowercase)
from simulator import add_stage_marker


def full_adder_1bit_core():
    qc = QuantumCircuit(4, name="FA_1bit_core")
    A, B, Cin, Cout = 0, 1, 2, 3
    qc.ccx(A, B, Cout)
    qc.cx(A, B)
    qc.ccx(B, Cin, Cout)
    qc.cx(B, Cin)
    qc.cx(A, B)
    return qc


def subtractor_nbit_same_logic(n: int):
    """
    Registers:
      A[0..n-1], B[0..n-1], C[0..n]

    Output:
      Difference bits -> C[0..n-1]
      Borrow bit      -> C[n]  (after final X)
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    A = QuantumRegister(n, "A")
    B = QuantumRegister(n, "B")
    C = QuantumRegister(n + 1, "C")
    qc = QuantumCircuit(A, B, C, name="SUB_%dbit_same_core" % n)

    # Two's complement preparation
    for i in range(n):
        qc.x(B[i])      # B <- ~B
    qc.barrier()

    qc.x(C[0])          # Cin <- 1
    qc.barrier()

    fa = full_adder_1bit_core()

    # Use a fixed qubit to host visible stage markers.
    # (These do NOT change the state; they only help drawing.)
    marker_qubit = A[0]

    add_stage_marker(qc, marker_qubit, "STAGE_0")  # optional "start" marker

    # Ripple stages using the SAME 1-bit core
    for i in range(n):
        qc.barrier()  # optional: stage boundary for readability
        add_stage_marker(qc, marker_qubit, f"STAGE_{i+1}")  # visible label
        qc.compose(fa, qubits=[A[i], B[i], C[i], C[i + 1]], inplace=True)

    add_stage_marker(qc, marker_qubit, "STAGE_END")
    qc.barrier()

    # Restore B, convert "no-borrow" -> "borrow"
    for i in range(n):
        qc.x(B[i])
    qc.barrier()
    qc.x(C[n])

    return qc


def _bits_to_int_lsb(bits):
    out = 0
    for i, b in enumerate(bits):
        out |= (b & 1) << i
    return out


def decode_subtractor_q0_basis(n: int, qc, bitstr_q0_to_qN: str):
    """
    Expects bitstr in q0..qN order (LEFTMOST is q0).

    Layout in qubit index order:
      A: q0..q(n-1)
      B: qn..q(2n-1)
      C: q(2n)..q(3n)
    """
    bits = list(map(int, bitstr_q0_to_qN))

    A_bits = bits[0:n]
    B_bits = bits[n:2 * n]
    C_bits = bits[2 * n:2 * n + (n + 1)]

    diff_bits = C_bits[0:n]
    borrow_bit = C_bits[n]

    return {
        "A_out": _bits_to_int_lsb(A_bits),
        "B_out": _bits_to_int_lsb(B_bits),
        "Diff_out": _bits_to_int_lsb(diff_bits),
        "Borrow_out": borrow_bit
    }


def expected_subtractor_decimal(A: int, B: int, n: int):
    return {
        "Diff_out": (A - B) % (1 << n),
        "Borrow_out": 1 if A < B else 0
    }


SUBTRACTOR_SPEC = OperationSpec(
    key="sub",
    name="Subtraction (A - B)",
    build_circuit=subtractor_nbit_same_logic,
    decode_q0_basis=decode_subtractor_q0_basis,
    expected_decimal=expected_subtractor_decimal
)
