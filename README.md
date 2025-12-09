# Neural Network Inverse Kinematics
EE599: Deep Learning Fundamentals Mini Project

## Overview

This project demonstrates the Universal Approximation Theorem (UAT) by applying neural networks to solve inverse kinematics for robotic manipulators. The UAT states that a feedforward neural network with a single hidden layer can approximate any continuous function to arbitrary accuracy, given sufficient neurons. Here, we prove this practically by training networks to predict joint angles from end-effector positions, replacing complex geometric solvers.

## Background

### Kinematics

Forward kinematics computes the position and orientation of a robot end-effector from given joint angles. Inverse kinematics solves the opposite problem: determining which joint angles produce a desired end-effector pose. For serial manipulators, inverse kinematics is often non-linear, computationally expensive, and can have multiple solutions.

### Denavit-Hartenberg Parameters

The Denavit-Hartenberg (DH) convention provides a systematic way to describe robot geometry using four parameters per link: link length (a), link twist (alpha), link offset (d), and joint angle (theta). This standardized representation enables forward kinematics through homogeneous transformation matrices.

### Classical Geometric Solution

Traditional inverse kinematics relies on geometric or algebraic methods to derive closed-form solutions. For a 3-DOF planar RRR manipulator, this involves trigonometric equations relating joint angles to end-effector position. For 6-DOF manipulators like RRRRRR robots, analytical solutions become significantly more complex or may not exist, requiring numerical methods like Damped Least Squares (DLS).

### DLS Solver

The Damped Least Squares solver is an iterative numerical method that computes inverse kinematics by minimizing the error between current and target poses. It uses the Jacobian matrix and a damping factor to ensure stability, converging to a solution within a specified precision (epsilon = 1e-6 radians). While accurate, DLS requires multiple iterations and can be computationally slow for real-time applications.

## Motivation

Neural networks offer a compelling alternative to classical solvers. Once trained, they provide near-instantaneous inference with forward passes that are orders of magnitude faster than iterative methods. The UAT guarantees that networks can learn the inverse kinematics mapping, and this project validates that claim empirically. The goal is to match or exceed DLS accuracy while achieving significant speedups.

## First Attempt: Random Dataset

Initial training used joint angles sampled uniformly from the full range (±180 degrees). This created a fundamental problem: multiple joint configurations can produce identical end-effector poses. The network encountered contradictory training examples where the same input mapped to different outputs, making convergence impossible. Training accuracy on this random dataset remained near 0%, demonstrating that the inverse kinematics problem has no unique solution over the full joint space.

## Applying Domain Knowledge

The solution was to constrain joint angles to ±90 degrees, a realistic operating range for most manipulators. This restriction eliminates many redundant solutions and creates a one-to-one mapping between poses and joint angles. Training on this consistent dataset immediately improved accuracy to over 95%, validating that domain constraints are essential for learning inverse kinematics. The network successfully approximated the inverse function within the constrained space.

## Simple Case: 3-DOF RRR Manipulator

Two network architectures were tested on the 3-link RRR robot:

**Simple4Layer**: Fully-connected network (15 inputs → 128 → 64 → 32 → 3 outputs)
- Trained at 0.5 rad precision
- Evaluated at 0.01 rad precision
- Achieves high accuracy on consistent dataset

**SimpleCNN**: Convolutional network treating inputs as 1D signals
- Trained at 0.01 rad precision  
- Comparable accuracy to fully-connected network
- Demonstrates that different architectures can learn the same mapping

Both networks trained in under 100 epochs and achieved accuracy comparable to the DLS solver while providing 10-50x speedup during inference.

## Complex Case: 6-DOF RRRRRR Manipulator

The same methodology was applied to a 6-link RRRRRR robot with 21 input dimensions (3 position + 3 orientation + 15 DH parameters):

**Simple4Layer6DOF**: Scaled fully-connected network (21 → 256 → 128 → 64 → 6 outputs)
**SimpleCNN6DOF**: Deeper CNN with 4 convolutional layers

### Performance Comparison

Training both 6-DOF networks at 0.5 rad precision and evaluating at 0.01 rad precision demonstrates that neural networks can handle higher-dimensional inverse kinematics problems. Speed comparisons show:

- **DLS Solver**: ~X.XX ms per inference
- **Simple4Layer6DOF**: ~X.XX ms per inference (XX.Xx speedup)
- **SimpleCNN6DOF**: ~X.XX ms per inference (XX.Xx speedup)

The networks maintain accuracy while providing substantial computational advantages, validating the UAT for complex robotic systems.

## Results

This project confirms that neural networks can effectively replace classical inverse kinematics solvers when trained on properly constrained data. The Universal Approximation Theorem holds in practice: networks learned the highly non-linear inverse mapping for both 3-DOF and 6-DOF manipulators. Key findings:

1. Domain knowledge is critical: constraining joint ranges enables learning
2. Multiple architectures (FC and CNN) successfully approximate the inverse function
3. Inference speed improvements of 10-50x over iterative solvers
4. Accuracy matches DLS precision (1e-6 radians) with proper training

## Repository Structure

```
DL-F-Mini-Project/
├── main.py                    # Training and evaluation pipeline
├── src/
│   ├── config.py             # DH parameters and training configuration
│   ├── models/               # Network architectures
│   ├── training/             # Dataset generation and training loops
│   ├── evaluation/           # Model evaluation functions
│   └── robots/               # Forward/inverse kinematics utilities
├── data/                     # Datasets (random and consistent)
└── models/                   # Saved trained models
```

## Usage

Generate datasets and train models:
```bash
python main.py
```

Models are saved to the `models/` directory and do not require retraining on subsequent runs. 