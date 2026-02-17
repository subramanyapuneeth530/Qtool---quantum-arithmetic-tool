# qtool/ops/karatsuba_gf2_multiplier.py
# GF(2) carryless (polynomial) multiplication using SMALL fixed circuits for n=2,3,4.
# This avoids recursion/ancilla blow-ups that make Statevector impossible.
#
# Layout (q0..qN) for decode in your tool:
#   A[0..n-1], B[0..n-1], P[0..2n-2], (ancillas...)
#
# Output Prod_out is the carryless product (2n-1 bits).

from qiskit import QuantumCircuit, QuantumRegister
from .base import OperationSpec


def _bits_to_int_lsb(bits):
    out = 0
    for i, b in enumerate(bits):
        out |= (b & 1) << i
    return out


def _carryless_mul_expected(A: int, B: int, n: int) -> int:
    """Schoolbook carryless (GF(2)) product for checking expected output."""
    prod = 0
    for i in range(n):
        if (A >> i) & 1:
            for j in range(n):
                if (B >> j) & 1:
                    prod ^= (1 << (i + j))
    return prod  # fits in 2n-1 bits


def _mul2_xor_into(qc: QuantumCircuit, x0, x1, y0, y1, p0, p1, p2):
    """
    2-bit carryless multiply: (x0 + x1 x) * (y0 + y1 x) = p0 + p1 x + p2 x^2
    XOR into (p0,p1,p2), assumes nothing about existing p bits.
    Uses 4 Toffolis (schoolbook).
    """
    qc.ccx(x0, y0, p0)
    qc.ccx(x0, y1, p1)
    qc.ccx(x1, y0, p1)
    qc.ccx(x1, y1, p2)


def _karatsuba_2bit(qc: QuantumCircuit, A, B, P):
    """
    2-bit Karatsuba carryless multiplier (no ancillas).
    Produces:
      P0 = A0B0
      P2 = A1B1
      P1 = (A0^A1)(B0^B1) ^ P0 ^ P2
    """
    a0, a1 = A[0], A[1]
    b0, b1 = B[0], B[1]
    p0, p1, p2 = P[0], P[1], P[2]

    # p0 = a0*b0
    qc.ccx(a0, b0, p0)
    # p2 = a1*b1
    qc.ccx(a1, b1, p2)

    # a0 <- a0 xor a1 ; b0 <- b0 xor b1
    qc.cx(a1, a0)
    qc.cx(b1, b0)

    # p1 = (a0*b0)
    qc.ccx(a0, b0, p1)

    # p1 <- p1 xor p0 xor p2
    qc.cx(p0, p1)
    qc.cx(p2, p1)

    # uncompute a0,b0
    qc.cx(b1, b0)
    qc.cx(a1, a0)


def _build_gf2_2bit() -> QuantumCircuit:
    A = QuantumRegister(2, "A")
    B = QuantumRegister(2, "B")
    P = QuantumRegister(3, "P")
    qc = QuantumCircuit(A, B, P, name="GF2_KARATSUBA_2bit")

    _karatsuba_2bit(qc, A, B, P)
    return qc


def _build_gf2_3bit() -> QuantumCircuit:
    """
    3-bit Karatsuba from the paper equation (degree-2 polys -> degree-4 product).
    Coefficients in GF(2) (XOR arithmetic):
      c0 = X0Y0
      c4 = X2Y2
      c1 = (X1^X0)(Y1^Y0) ^ c0 ^ (X1Y1)
      c3 = (X2^X1)(Y2^Y1) ^ c4 ^ (X1Y1)
      c2 = (X2^X0)(Y2^Y0) ^ c0 ^ c4 ^ (X1Y1)
    We implement this with 4 ancillas: a11, sx, sy, t.
    All ancillas are uncomputed back to |0>.
    """
    A = QuantumRegister(3, "A")
    B = QuantumRegister(3, "B")
    P = QuantumRegister(5, "P")  # product bits c0..c4

    a11 = QuantumRegister(1, "a11")  # X1Y1
    sx  = QuantumRegister(1, "sx")
    sy  = QuantumRegister(1, "sy")
    t   = QuantumRegister(1, "t")    # temp product of sx*sy

    qc = QuantumCircuit(A, B, P, a11, sx, sy, t, name="GF2_KARATSUBA_3bit")

    x0, x1, x2 = A[0], A[1], A[2]
    y0, y1, y2 = B[0], B[1], B[2]
    c0, c1, c2, c3, c4 = P[0], P[1], P[2], P[3], P[4]

    # c0 = x0*y0 ; c4 = x2*y2
    qc.ccx(x0, y0, c0)
    qc.ccx(x2, y2, c4)

    # a11 = x1*y1 (kept until end)
    qc.ccx(x1, y1, a11[0])

    # ---- c1: (x1^x0)(y1^y0) ^ c0 ^ a11
    qc.cx(x0, sx[0]); qc.cx(x1, sx[0])          # sx = x0^x1
    qc.cx(y0, sy[0]); qc.cx(y1, sy[0])          # sy = y0^y1
    qc.ccx(sx[0], sy[0], t[0])                   # t = sx*sy
    qc.cx(t[0], c1); qc.cx(c0, c1); qc.cx(a11[0], c1)
    qc.ccx(sx[0], sy[0], t[0])                   # uncompute t
    qc.cx(y1, sy[0]); qc.cx(y0, sy[0])          # uncompute sy
    qc.cx(x1, sx[0]); qc.cx(x0, sx[0])          # uncompute sx

    # ---- c3: (x2^x1)(y2^y1) ^ c4 ^ a11
    qc.cx(x1, sx[0]); qc.cx(x2, sx[0])          # sx = x1^x2
    qc.cx(y1, sy[0]); qc.cx(y2, sy[0])          # sy = y1^y2
    qc.ccx(sx[0], sy[0], t[0])
    qc.cx(t[0], c3); qc.cx(c4, c3); qc.cx(a11[0], c3)
    qc.ccx(sx[0], sy[0], t[0])
    qc.cx(y2, sy[0]); qc.cx(y1, sy[0])
    qc.cx(x2, sx[0]); qc.cx(x1, sx[0])

    # ---- c2: (x2^x0)(y2^y0) ^ c0 ^ c4 ^ a11
    qc.cx(x0, sx[0]); qc.cx(x2, sx[0])          # sx = x0^x2
    qc.cx(y0, sy[0]); qc.cx(y2, sy[0])          # sy = y0^y2
    qc.ccx(sx[0], sy[0], t[0])
    qc.cx(t[0], c2); qc.cx(c0, c2); qc.cx(c4, c2); qc.cx(a11[0], c2)
    qc.ccx(sx[0], sy[0], t[0])
    qc.cx(y2, sy[0]); qc.cx(y0, sy[0])
    qc.cx(x2, sx[0]); qc.cx(x0, sx[0])

    # uncompute a11
    qc.ccx(x1, y1, a11[0])

    return qc


def _build_gf2_4bit() -> QuantumCircuit:
    """
    4-bit Karatsuba split (2+2):
      X = XH*x^2 + XL
      Y = YH*x^2 + YL
      z0 = XL*YL (2-bit carryless => 3 bits)
      z2 = XH*YH (2-bit carryless => 3 bits)
      m  = (XL^XH)*(YL^YH) (2-bit carryless => 3 bits)
      out = z0 ^ ((m ^ z0 ^ z2) << 2) ^ (z2 << 4)
    We compute directly into P with ONLY 4 ancillas: SX(2), SY(2).
    """
    A = QuantumRegister(4, "A")
    B = QuantumRegister(4, "B")
    P = QuantumRegister(7, "P")      # 2n-1 = 7

    SX = QuantumRegister(2, "SX")    # XL^XH
    SY = QuantumRegister(2, "SY")    # YL^YH

    qc = QuantumCircuit(A, B, P, SX, SY, name="GF2_KARATSUBA_4bit")

    # aliases
    a0, a1, a2, a3 = A[0], A[1], A[2], A[3]
    b0, b1, b2, b3 = B[0], B[1], B[2], B[3]

    # z0 into P[0..2]
    _mul2_xor_into(qc, a0, a1, b0, b1, P[0], P[1], P[2])

    # z2 into P[4..6]
    _mul2_xor_into(qc, a2, a3, b2, b3, P[4], P[5], P[6])

    # SX = XL ^ XH, SY = YL ^ YH
    qc.cx(a0, SX[0]); qc.cx(a2, SX[0])
    qc.cx(a1, SX[1]); qc.cx(a3, SX[1])

    qc.cx(b0, SY[0]); qc.cx(b2, SY[0])
    qc.cx(b1, SY[1]); qc.cx(b3, SY[1])

    # m into P[2..4]
    _mul2_xor_into(qc, SX[0], SX[1], SY[0], SY[1], P[2], P[3], P[4])

    # add z0 shifted by 2 into P[2..4]
    _mul2_xor_into(qc, a0, a1, b0, b1, P[2], P[3], P[4])

    # add z2 shifted by 2 into P[2..4]
    _mul2_xor_into(qc, a2, a3, b2, b3, P[2], P[3], P[4])

    # uncompute SX,SY back to |0>
    qc.cx(b2, SY[0]); qc.cx(b0, SY[0])
    qc.cx(b3, SY[1]); qc.cx(b1, SY[1])

    qc.cx(a2, SX[0]); qc.cx(a0, SX[0])
    qc.cx(a3, SX[1]); qc.cx(a1, SX[1])

    return qc


def karatsuba_gf2_multiplier_nbit(n: int) -> QuantumCircuit:
    """
    Dispatch to fixed small circuits for n=2,3,4 (paper-style).
    Avoids recursion ancilla explosion that breaks Statevector.
    """
    if n == 2:
        return _build_gf2_2bit()
    if n == 3:
        return _build_gf2_3bit()
    if n == 4:
        return _build_gf2_4bit()

    raise ValueError(
        "This GF(2) Karatsuba module supports only n=2,3,4 (fixed circuits). "
        "Larger n with Statevector will explode in memory. "
        "If you want larger n, add a non-statevector backend or a schoolbook multiplier."
    )


def decode_karatsuba_q0_basis(n: int, qc, bitstr_q0_to_qN: str):
    bits = list(map(int, bitstr_q0_to_qN))

    A_bits = bits[0:n]
    B_bits = bits[n:2 * n]
    P_bits = bits[2 * n:2 * n + (2 * n - 1)]

    return {
        "A_out": _bits_to_int_lsb(A_bits),
        "B_out": _bits_to_int_lsb(B_bits),
        "Prod_out": _bits_to_int_lsb(P_bits),  # carryless product (2n-1 bits)
    }


def expected_karatsuba_decimal(A: int, B: int, n: int):
    return {"Prod_out": _carryless_mul_expected(A, B, n)}


KARATSUBA_GF2_SPEC = OperationSpec(
    key="mul_gf2_karatsuba",
    name="Multiply (GF(2) carryless, Karatsuba n=2..4)",
    build_circuit=karatsuba_gf2_multiplier_nbit,
    decode_q0_basis=decode_karatsuba_q0_basis,
    expected_decimal=expected_karatsuba_decimal,
)
