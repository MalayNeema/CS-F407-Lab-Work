import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set styling for clear, readable academic plots
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

save_dir = r"C:\Users\nikhi\.gemini\antigravity\scratch\neural_models_lab"
os.makedirs(save_dir, exist_ok=True)

# 1. Dataset
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=torch.float32)
y_xor = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
y_3class = torch.tensor([0, 1, 1, 2], dtype=torch.long)

# -------------------------------------------------------------
# PLOT 1: XOR Geometric Representation & Linear Inseparability
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
x_pts = X.numpy()
y_pts = y_xor.numpy().squeeze()

ax.scatter(x_pts[y_pts == 0, 0], x_pts[y_pts == 0, 1], color='#1f77b4', s=160, marker='o', edgecolors='k', linewidth=1.5, label='Class 0 (Normal / Agreement)')
ax.scatter(x_pts[y_pts == 1, 0], x_pts[y_pts == 1, 1], color='#d62728', s=160, marker='s', edgecolors='k', linewidth=1.5, label='Class 1 (Warning / Disagreement)')

# Label points
for (xi, yi), label in zip(x_pts, y_pts):
    ax.annotate(f"({int(xi)}, {int(yi)}) -> {int(label)}", 
                (xi, yi), textcoords="offset points", xytext=(10, 10),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Draw attempted separating lines to illustrate impossibility
line_x = np.linspace(-0.2, 1.2, 100)
ax.plot(line_x, 0.5 * np.ones_like(line_x), '--', color='gray', alpha=0.7, label='Candidate linear boundary A')
ax.plot(line_x, 1.1 - line_x, ':', color='purple', alpha=0.7, label='Candidate linear boundary B')

ax.set_xlim(-0.3, 1.4)
ax.set_ylim(-0.3, 1.4)
ax.set_xlabel('Sensor $x_1$', fontsize=12)
ax.set_ylabel('Sensor $x_2$', fontsize=12)
ax.set_title('Redundant Safety Sensors (XOR Decision Problem)\nGeometric Linear Non-Separability', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(save_dir, "fig1_xor_linear_inseparability.png"))
plt.close()
print("Saved fig1_xor_linear_inseparability.png")

# -------------------------------------------------------------
# TASK 4: Network Architecture definition
# -------------------------------------------------------------
class XORNet(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.act = activation_fn
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        h = self.act(self.fc1(x))
        logits = self.fc2(h)
        return logits

# -------------------------------------------------------------
# TASK 4 Part A & B: Baseline Run (Seed 2, Adam lr=0.05)
# -------------------------------------------------------------
torch.manual_seed(2)
model_baseline = XORNet(nn.Tanh())
crit_bce = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_baseline.parameters(), lr=0.05)

# Step 0 initial loss and gradient
init_logits = model_baseline(X)
init_loss = crit_bce(init_logits, y_xor)
optimizer.zero_grad()
init_loss.backward()
grad_step1 = model_baseline.fc1.weight.grad.clone()
init_loss_val = init_loss.item()

loss_history = []
for epoch in range(2000):
    optimizer.zero_grad()
    out = model_baseline(X)
    loss = crit_bce(out, y_xor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

final_loss_val = loss.item()
with torch.no_grad():
    final_probs = torch.sigmoid(model_baseline(X))
    final_preds = (final_probs > 0.5).int()

print(f"Baseline (Tanh, Seed 2): Initial Loss = {init_loss_val:.4f}, Final Loss = {final_loss_val:.6f}")
print("Baseline final probabilities:", [round(p, 4) for p in final_probs.squeeze().tolist()])
print("Baseline final predictions:", final_preds.squeeze().tolist())
print("Baseline early grad W1 (step 1):\n", grad_step1.numpy())

# -------------------------------------------------------------
# TASK 4 Part C: Symmetry Experiment (Zero Initialization)
# -------------------------------------------------------------
model_zero = XORNet(nn.Tanh())
with torch.no_grad():
    for p in model_zero.parameters():
        p.zero_()

opt_zero = optim.SGD(model_zero.parameters(), lr=0.1)
weights_history_row0 = []
weights_history_row1 = []
step_records = []

for step in range(1, 11):
    opt_zero.zero_grad()
    out = model_zero(X)
    loss = crit_bce(out, y_xor)
    loss.backward()
    
    w1_val = model_zero.fc1.weight.data.clone().numpy()
    grad_val = model_zero.fc1.weight.grad.clone().numpy()
    weights_history_row0.append(w1_val[0, :].copy())
    weights_history_row1.append(w1_val[1, :].copy())
    step_records.append((step, w1_val, grad_val, loss.item()))
    
    opt_zero.step()

# -------------------------------------------------------------
# TASK 4 Part D: Activation Comparison (Sigmoid, Tanh, ReLU)
# -------------------------------------------------------------
activations = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU()
}

activation_stats = {}
activation_curves = {}

for name, act in activations.items():
    torch.manual_seed(2)
    m = XORNet(act)
    opt = optim.Adam(m.parameters(), lr=0.05)
    crit = nn.BCEWithLogitsLoss()
    
    early_grad_norm = None
    curve = []
    for step in range(1, 2001):
        opt.zero_grad()
        out = m(X)
        l = crit(out, y_xor)
        l.backward()
        if step == 10:
            early_grad_norm = torch.norm(m.fc1.weight.grad).item()
        opt.step()
        curve.append(l.item())
        
    with torch.no_grad():
        probs = torch.sigmoid(m(X))
        preds = (probs > 0.5).int()
        all_correct = (preds == y_xor).all().item()
        
    activation_stats[name] = {
        'Final loss': l.item(),
        '4/4 correct?': all_correct,
        'Early ||grad||': early_grad_norm,
        'probs': probs.squeeze().tolist()
    }
    activation_curves[name] = curve
    print(f"Activation {name}: Final Loss = {l.item():.6f}, Correct = {all_correct}, Early Grad Norm = {early_grad_norm:.6f}")

# Plot 2: Activation Training Loss Curves
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, curve in activation_curves.items():
    ax.plot(curve, label=f"{name} (Final Loss: {activation_stats[name]['Final loss']:.4f})", linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Training Step (Epoch)', fontsize=11)
ax.set_ylabel('Binary Cross-Entropy Loss (Log Scale)', fontsize=11)
ax.set_title('Task 4 Part D: Optimization Dynamics Across Hidden Activations', fontsize=12, fontweight='bold')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(save_dir, "fig2_activation_loss_curves.png"))
plt.close()
print("Saved fig2_activation_loss_curves.png")

# Plot 3: Decision Surfaces
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x_grid = np.linspace(-0.2, 1.2, 200)
y_grid = np.linspace(-0.2, 1.2, 200)
XX, YY = np.meshgrid(x_grid, y_grid)
grid_tensor = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)

for ax, (name, act) in zip(axes, activations.items()):
    torch.manual_seed(2)
    m = XORNet(act)
    opt = optim.Adam(m.parameters(), lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        crit_bce(m(X), y_xor).backward()
        opt.step()
    with torch.no_grad():
        Z = torch.sigmoid(m(grid_tensor)).reshape(XX.shape).numpy()
    
    contour = ax.contourf(XX, YY, Z, levels=50, cmap='RdBu_r', alpha=0.8, vmin=0, vmax=1)
    ax.contour(XX, YY, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    ax.scatter(x_pts[y_pts == 0, 0], x_pts[y_pts == 0, 1], color='blue', s=100, edgecolors='white', linewidth=1.5, label='Class 0')
    ax.scatter(x_pts[y_pts == 1, 0], x_pts[y_pts == 1, 1], color='red', s=100, edgecolors='white', linewidth=1.5, label='Class 1')
    ax.set_title(f'Decision Surface: {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig.savefig(os.path.join(save_dir, "fig3_decision_surfaces.png"))
plt.close()
print("Saved fig3_decision_surfaces.png")

# -------------------------------------------------------------
# TASK 5: Three-Class Decision Model
# -------------------------------------------------------------
class MultiClassXORNet(nn.Module):
    def __init__(self, hidden_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 3) # 3 class logits

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

torch.manual_seed(2)
mc_model = MultiClassXORNet(hidden_dim=2)
crit_ce = nn.CrossEntropyLoss()
opt_mc = optim.Adam(mc_model.parameters(), lr=0.05)

for step in range(2000):
    opt_mc.zero_grad()
    logits = mc_model(X)
    loss_mc = crit_ce(logits, y_3class)
    loss_mc.backward()
    opt_mc.step()

with torch.no_grad():
    logits_mc = mc_model(X)
    probs_mc = torch.softmax(logits_mc, dim=-1)
    preds_mc = torch.argmax(probs_mc, dim=-1)
    
    # Diagnostic: Add +100 to all logits
    logits_shifted = logits_mc + 100.0
    probs_shifted = torch.softmax(logits_shifted, dim=-1)
    max_shift_diff = (probs_mc - probs_shifted).abs().max().item()

print("\n--- TASK 5 MULTICLASS RESULTS ---")
print("W2 shape:", mc_model.fc2.weight.shape)
print("Logits shape:", logits_mc.shape)
print("Softmax Probabilities:\n", probs_mc.numpy())
print("Row 0 sum:", probs_mc[0].sum().item())
print("Predictions:", preds_mc.tolist(), "Ground truth:", y_3class.tolist())
print("Max diff when adding +100:", max_shift_diff)

print("\nAll experiments and plots completed successfully.")
