# Presentation

LaTeX Beamer presentation explaining the neural network inverse kinematics project.

## Building the Presentation

From this directory, run:

```bash
make
```

This generates `main.pdf` containing the full presentation.

## Cleaning

To remove generated files:

```bash
make clean
```

## Viewing

To view the PDF after building:

```bash
make view
```

## Contents

The presentation covers:

1. Universal Approximation Theorem - theoretical background
2. Kinematics and DH Parameters - robot geometry fundamentals
3. Classical geometric solutions and DLS iterative solver
4. First attempt with full joint range (±180°) - demonstrates the problem
5. Solution: constraining joint range to ±90°
6. 3-DOF RRR robot - simple case with both FC and CNN networks
7. 6-DOF RRRRRR robot - complex case with performance comparison
8. Results and conclusions

## Requirements

- LaTeX with Beamer package
- pdflatex (standard with most LaTeX distributions)
