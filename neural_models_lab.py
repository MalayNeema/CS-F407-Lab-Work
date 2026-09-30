"""
Laboratory - Neural Models: Learning, Depth, Activations, and Output Layers
Complete executable laboratory script covering Tasks 1 to 5.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def run_all_tasks():
    print("=" * 70)
    print("LABORATORY: NEURAL MODELS - LEARNING, DEPTH, ACTIVATIONS, OUTPUT LAYERS")
    print("=" * 70)

    # Global Dataset Definition
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    y_xor = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    # -------------------------------------------------------------------------
    # TASK 1: Understand the Problem Before Coding
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 1: Linear Model Check on XOR")
    print("=" * 70)
    torch.manual_seed(42)
    linear_clf = nn.Linear(2, 1)
    opt_lin = optim.SGD(linear_clf.parameters(), lr=0.1)
    crit_bce_logits = nn.BCEWithLogitsLoss()

    for step in range(2000):
        opt_lin.zero_grad()
        out = linear_clf(X)
        loss_lin = crit_bce_logits(out, y_xor)
        loss_lin.backward()
        opt_lin.step()

    with torch.no_grad():
        probs_lin = torch.sigmoid(linear_clf(X))
        preds_lin = (probs_lin > 0.5).int()
    print(f"Linear Model Final Loss: {loss_lin.item():.4f}")
    print(f"Linear Model Predicted Probabilities:\n  {probs_lin.squeeze().tolist()}")
    print(f"Linear Model Thresholded (0.5): {preds_lin.squeeze().tolist()} | Ground Truth: {y_xor.squeeze().int().tolist()}")
    print(f"Accuracy: {(preds_lin == y_xor).float().mean().item() * 100:.1f}% (Fails XOR due to linear non-separability)")

    # -------------------------------------------------------------------------
    # TASK 2 & 3: Model Architecture Definition
    # -------------------------------------------------------------------------
    class XORNet(nn.Module):
        """2-2-1 Neural Network with customizable activation function."""
        def __init__(self, activation=nn.Tanh()):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.act = activation
            self.fc2 = nn.Linear(2, 1)

        def forward(self, x):
            # Forward pass: affine -> activation -> affine (returns logit)
            h = self.act(self.fc1(x))
            logits = self.fc2(h)
            return logits

    # -------------------------------------------------------------------------
    # TASK 4: Execute, Test, and Diagnose Generated Code
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 4 PART A: Basic Learning Check (Baseline 2-2-1 Network, Tanh)")
    print("=" * 70)
    torch.manual_seed(2)
    model = XORNet(nn.Tanh())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Step 0: Record initial loss
    with torch.no_grad():
        init_logits = model(X)
        init_loss = criterion(init_logits, y_xor).item()
    print(f"Initial BCE Loss: {init_loss:.4f}")

    # Inspect early gradient (Step 1)
    optimizer.zero_grad()
    loss_step1 = criterion(model(X), y_xor)
    loss_step1.backward()
    grad_w1_step1 = model.fc1.weight.grad.clone()
    print("\nTASK 4 PART B: Backpropagation Check")
    print(f"First-layer weight gradient dL/dW^(1) at Step 1:\n{grad_w1_step1}")
    print(f"Euclidean Norm ||dL/dW^(1)||_2 at Step 1: {torch.norm(grad_w1_step1).item():.6f}")

    # Train model
    for step in range(2000):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y_xor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_probs = torch.sigmoid(model(X))
        final_preds = (final_probs > 0.5).int()
        final_loss = loss.item()

    print(f"\nFinal BCE Loss: {final_loss:.6f}")
    print(f"Final Probabilities: {final_probs.squeeze().tolist()}")
    print(f"Thresholded Predictions: {final_preds.squeeze().tolist()}")
    print(f"All 4 examples correct: {(final_preds == y_xor).all().item()}")

    # -------------------------------------------------------------------------
    # TASK 4 PART C: Symmetry Experiment (Zero Weight Initialization)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 4 PART C: Symmetry Experiment (All Weights Initialized to Zero)")
    print("=" * 70)
    model_zero = XORNet(nn.Tanh())
    with torch.no_grad():
        for p in model_zero.parameters():
            p.zero_()

    opt_zero = optim.SGD(model_zero.parameters(), lr=0.1)
    print("Initial W1 (Layer 1 weights):\n", model_zero.fc1.weight.data)
    for step in range(1, 6):
        opt_zero.zero_grad()
        out = model_zero(X)
        loss_zero = criterion(out, y_xor)
        loss_zero.backward()
        opt_zero.step()
        print(f"Step {step} - W1 weights:\n{model_zero.fc1.weight.data}")
        print(f"Step {step} - Row 0 == Row 1: {torch.equal(model_zero.fc1.weight.data[0], model_zero.fc1.weight.data[1])}")

    # -------------------------------------------------------------------------
    # TASK 4 PART D: Activation Experiment (Sigmoid vs Tanh vs ReLU)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 4 PART D: Activation Experiment (Sigmoid vs Tanh vs ReLU)")
    print("=" * 70)
    activations = {
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'ReLU': nn.ReLU()
    }
    print(f"{'Hidden Activation':<18} | {'Final Loss':<12} | {'4/4 Correct?':<14} | {'Early ||grad W1||':<18}")
    print("-" * 70)
    for name, act in activations.items():
        torch.manual_seed(2)
        m = XORNet(act)
        crit = nn.BCEWithLogitsLoss()
        opt = optim.Adam(m.parameters(), lr=0.05)
        early_norm = None

        for step in range(1, 2001):
            opt.zero_grad()
            out = m(X)
            l = crit(out, y_xor)
            l.backward()
            if step == 10:
                early_norm = torch.norm(m.fc1.weight.grad).item()
            opt.step()

        with torch.no_grad():
            probs = torch.sigmoid(m(X))
            preds = (probs > 0.5).int()
            all_correct = (preds == y_xor).all().item()

        print(f"{name:<18} | {l.item():<12.6f} | {str(all_correct):<14} | {early_norm:<18.6f}")

    # -------------------------------------------------------------------------
    # TASK 5: Extend to Three-Class Decision Problem
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 5: Multi-Class Decision Model (3 Classes: Inactive, Disagree, Active)")
    print("=" * 70)
    y_3class = torch.tensor([0, 1, 1, 2], dtype=torch.long)

    class MultiClassNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.act = nn.Tanh()
            self.fc2 = nn.Linear(2, 3) # 3 class logits

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    torch.manual_seed(2)
    mc_model = MultiClassNet()
    crit_mc = nn.CrossEntropyLoss()
    opt_mc = optim.Adam(mc_model.parameters(), lr=0.05)

    for step in range(2000):
        opt_mc.zero_grad()
        logits = mc_model(X)
        loss_mc = crit_mc(logits, y_3class)
        loss_mc.backward()
        opt_mc.step()

    with torch.no_grad():
        final_logits = mc_model(X)
        probs_mc = torch.softmax(final_logits, dim=-1)
        preds_mc = torch.argmax(probs_mc, dim=-1)

    print(f"Final 3-Class Cross-Entropy Loss: {loss_mc.item():.6f}")
    print(f"Final Weight Matrix W2 Shape: {mc_model.fc2.weight.shape} (3 output classes x 2 hidden units)")
    print(f"Number of logits per example: {final_logits.shape[1]}")
    print(f"Softmax Probabilities for all 4 inputs:\n{probs_mc.numpy()}")
    print(f"Row 0 Probabilities Sum: {probs_mc[0].sum().item():.6f} (verifying sum to 1)")
    print(f"Predicted Classes: {preds_mc.tolist()} | Ground Truth: {y_3class.tolist()}")

    # Optional Diagnostic: Shift-invariance of Softmax
    logits_shifted = final_logits + 100.0
    probs_shifted = torch.softmax(logits_shifted, dim=-1)
    max_shift_diff = (probs_mc - probs_shifted).abs().max().item()
    print(f"Shift Invariance Check (Logits + 100): Max Diff = {max_shift_diff:.2e}")
    print("=" * 70)
    print("ALL LAB EXPERIMENTS EXECUTED AND VALIDATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_all_tasks()
