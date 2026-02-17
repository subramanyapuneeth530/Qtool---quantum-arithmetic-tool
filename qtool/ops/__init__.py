# ops/__init__.py
from .subtractor import SUBTRACTOR_SPEC
from .adder import ADDER_SPEC
from .karatsuba_gf2_multiplier import KARATSUBA_GF2_SPEC

OPERATIONS = {
    SUBTRACTOR_SPEC.key: SUBTRACTOR_SPEC,
    ADDER_SPEC.key: ADDER_SPEC,
    KARATSUBA_GF2_SPEC.key: KARATSUBA_GF2_SPEC,
}
