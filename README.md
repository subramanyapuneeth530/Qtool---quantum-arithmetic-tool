# QTool — n-bit Quantum Arithmetic Tool (Qiskit + Tkinter)

QTool is a small desktop GUI for experimenting with **n-bit quantum arithmetic circuits** using **Qiskit Statevector simulation**. It lets you choose an operation (adder / subtractor / GF(2) Karatsuba multiplier), pick the bit-width `n`, provide inputs in multiple ways (decimal, probability, full complex amplitudes), and then inspect:

- a grouped circuit diagram (with stage markers when available),
- formatted initial/final statevectors (nonzero amplitudes),
- probability bar charts for the most likely basis states.

> This project is aimed at learning, debugging, and visualizing how arithmetic circuits map quantum states to quantum states.

---

## Features

### Operations included
- **Addition (A + B)**: ripple-carry style using a 1-bit full-adder core.
- **Subtraction (A − B)**: two’s complement approach with stage markers for clearer circuit visualization.
- **GF(2) carryless multiplication (Karatsuba)**: fixed small circuits for **n = 2, 3, 4** only.

### Input modes
- **Decimal**: enter `A` and `B` as integers.
- **Probability**: specify per-qubit probabilities for |0⟩ / |1⟩ style product states.
- **Amplitude**: enter per-qubit complex amplitudes (supports real or complex; tool normalizes).

### Outputs
- Circuit diagram window (Qiskit drawer)
- Initial + final nonzero-state probability charts
- Initial + final statevector text (nonzero amplitudes only, tolerance-controlled)

---

## Project layout

```
qtool/
  app.py                  # Tkinter GUI
  simulator.py             # state prep + formatting + plotting helpers
  ops/
    base.py                # OperationSpec dataclass
    adder.py               # n-bit adder circuit + decode/expected
    subtractor.py          # n-bit subtractor circuit + decode/expected
    karatsuba_gf2_multiplier.py  # GF(2) carryless multiplier for n=2..4
    __init__.py            # operation registry
```

---

## Requirements

- Python **3.9+** recommended
- Qiskit
- NumPy
- Matplotlib
- Tkinter (usually included with Python on Windows/macOS; may require a package on Linux)

If you're on Debian/Ubuntu and Tkinter is missing:
```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

---

## Installation

```bash
git clone <your-repo-url>
cd qtool
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install qiskit numpy matplotlib
```

*(Optional)* If you prefer a requirements file, create `requirements.txt`:
```txt
qiskit
numpy
matplotlib
```
Then:
```bash
pip install -r requirements.txt
```

---

## Running the app

Because the current imports are written for running from inside the `qtool/` directory, run it like this:

```bash
cd qtool
python app.py
```

A window titled **“n-bit Quantum Arithmetic Tool”** should open.

---

## Using QTool

1. Pick an **Operation** (add / sub / GF(2) karatsuba multiply).
2. Set `n` (bit-width), click **Apply n**.
3. Choose an **Input Mode**:
   - **Decimal**: enter A and B
   - **Probability** / **Amplitude**: fill per-qubit fields (A and B registers)
4. Click **Run Simulation**.
5. Inspect results windows:
   - grouped circuit diagram
   - text summary + statevectors
   - initial/final probability bar charts

### Tuning output
- **Amplitude tol (nonzero):** hides tiny amplitudes in the text/plots
- **Top-K nonzero (plots):** limits how many basis states are shown in charts

---

## Notes & limitations

- This tool uses **Statevector simulation**, so memory grows exponentially with qubits.
- The **GF(2) Karatsuba multiplier** is intentionally limited to **n = 2..4** to avoid statevector blow-ups.
- Subtractor stage markers are *visual-only* identity markers used to make circuit drawings easier to read.

---

## Troubleshooting

### “No module named …”
Run the app from the `qtool/` directory:
```bash
cd qtool
python app.py
```

### Tkinter errors on Linux
Install Tkinter:
```bash
sudo apt-get install -y python3-tk
```

### Circuit is too large / simulation is slow
Reduce `n`, increase amplitude tolerance, or use smaller operations. Statevector size doubles with every qubit.

---

## License

Choose a license that matches your intent (MIT is common for small tools).
Add a `LICENSE` file if you plan to publish publicly.
