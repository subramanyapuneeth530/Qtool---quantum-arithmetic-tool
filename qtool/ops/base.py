# ops/base.py
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from qiskit import QuantumCircuit

@dataclass
class OperationSpec:
    key: str
    name: str

    # Build circuit for n
    build_circuit: Callable[[int], QuantumCircuit]

    # Decode from a basis bitstring shown in q0..qN order (LEFTMOST is q0)
    decode_q0_basis: Callable[[int, QuantumCircuit, str], Dict[str, int]]

    # Expected outputs for Decimal mode (optional)
    expected_decimal: Optional[Callable[[int, int, int], Dict[str, int]]] = None
