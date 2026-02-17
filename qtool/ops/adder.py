# ops/adder.py
from qiskit import QuantumCircuit, QuantumRegister
from .base import OperationSpec

def full_adder_1bit_core():
    qc = QuantumCircuit(4, name="FA_1bit_core")
    A, B, Cin, Cout = 0, 1, 2, 3
    qc.ccx(A, B, Cout)
    qc.cx(A, B)
    qc.ccx(B, Cin, Cout)
    qc.cx(B, Cin)
    qc.cx(A, B)
    return qc

def adder_nbit_same_logic(n: int):
    if n < 1:
        raise ValueError("n must be >= 1")

    A = QuantumRegister(n, "A")
    B = QuantumRegister(n, "B")
    S = QuantumRegister(n, "S")     # holds carry-in + sums
    Cout = QuantumRegister(1, "Cout")

    qc = QuantumCircuit(A, B, S, Cout, name="ADD_%dbit" % n)

    fa = full_adder_1bit_core()
    for i in range(n):
        cin = S[i]
        cout = S[i + 1] if i < n - 1 else Cout[0]
        qc.barrier(A[i], B[i], cin, cout)
        qc.compose(fa, qubits=[A[i], B[i], cin, cout], inplace=True)

    return qc

def _bits_to_int_lsb(bits):
    out = 0
    for i, b in enumerate(bits):
        out |= (b & 1) << i
    return out

def decode_adder_q0_basis(n: int, qc, bitstr_q0_to_qN: str):
    # Layout:
    # A: q0..q(n-1), B: qn..q(2n-1), S: q(2n)..q(3n-1), Cout: q(3n)
    bits = list(map(int, bitstr_q0_to_qN))
    A_bits = bits[0:n]
    B_bits = bits[n:2*n]
    S_bits = bits[2*n:3*n]
    cout_bit = bits[3*n]

    return {
        "A_out": _bits_to_int_lsb(A_bits),
        "B_out": _bits_to_int_lsb(B_bits),
        "Sum_out": _bits_to_int_lsb(S_bits),
        "Cout_out": cout_bit
    }

def expected_adder_decimal(A: int, B: int, n: int):
    total = A + B
    return {
        "Sum_out": total % (1 << n),
        "Cout_out": 1 if total >= (1 << n) else 0
    }

ADDER_SPEC = OperationSpec(
    key="add",
    name="Addition (A + B)",
    build_circuit=adder_nbit_same_logic,
    decode_q0_basis=decode_adder_q0_basis,
    expected_decimal=expected_adder_decimal
)
