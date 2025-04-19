#%%
import torch as t
from tqdm import tqdm
class Mess3Process:
    def __init__(self):
        self.T_A = t.tensor([
            [0.76500, 0.00375, 0.00375],
            [0.04250, 0.06750, 0.00375],
            [0.04250, 0.00375, 0.06750]
        ], dtype=t.float32)

        self.T_B = t.tensor([
            [0.06750, 0.04250, 0.00375],
            [0.00375, 0.76500, 0.00375],
            [0.00375, 0.04250, 0.06750]
        ], dtype=t.float32)

        self.T_C = t.tensor([
            [0.06750, 0.00375, 0.04250],
            [0.00375, 0.06750, 0.04250],
            [0.00375, 0.00375, 0.76500]
        ], dtype=t.float32)
        # self.T_A = t.tensor(
        #     [[0.4, 0.3, 0.3],
        #     [0.3, 0.4, 0.3],
        #     [0.3, 0.3, 0.4]],
        #     dtype=t.float32
        # )
        # self.T_B = t.tensor([[0.45, 0.45, 0.1],
        #                      [0.1, 0.8, 0.1],
        #                      [0.1, 0.1, 0.8]])
        # self.T_C = t.tensor([[0.45, 0.1, 0.45],
        #                      [0.1, 0.45, 0.45],
        #                      [0.1, 0.1, 0.8]])
        # self.T_B = self.T_A.clone()
        # self.T_C = self.T_A.clone()
        self.tokens = ['A', 'B', 'C']
        self.num_states = 3
        self.T_A = self.T_A / self.T_A.sum(dim=1, keepdim=True)
        self.T_B = self.T_B / self.T_B.sum(dim=1, keepdim=True)
        self.T_C = self.T_C / self.T_C.sum(dim=1, keepdim=True)

    def generate_sequence(self, length, use_tqdm=True):
        states = t.zeros(length, dtype=t.long)
        observations = []
        current_state = t.randint(0, self.num_states, (1,)).item()
        for t_idx in tqdm(range(length), disable=not use_tqdm):
            states[t_idx] = current_state
            T_choice = t.randint(0, 3, (1,)).item()
            T = self.T_A if T_choice == 0 else self.T_B if T_choice == 1 else self.T_C
            probs = T[current_state]
            token_idx = t.multinomial(probs, 1).item()
            token = self.tokens[token_idx]
            observations.append(token)
            current_state = t.multinomial(probs, 1).item()
        return states, observations
#%%
# states, observations = proc.generate_sequence(1000)

#%% The optimal belief state
proc = Mess3Process()
T = proc.T_A + proc.T_B + proc.T_C
T = T / T.sum(dim=1, keepdim=True)

# Find the left eigenvector of the transition matrix T
eigvals, eigvecs = t.linalg.eig(T.T)  # Transpose T since we want left eigenvector
# Get index of largest eigenvalue
max_eigval_idx = t.argmax(eigvals.real)
# Get corresponding eigenvector and normalize it
stationary_dist = eigvecs[:, max_eigval_idx].real
stationary_dist = stationary_dist / stationary_dist.sum()

print("Stationary distribution:", stationary_dist)
print("Largest eigenvalue:", eigvals[max_eigval_idx].real)

#%% compute the optimal belief state visualization

burn_in = 100
sts, obs = proc.generate_sequence(100000)

#%%
from tqdm import tqdm

dst = stationary_dist.clone()
T_dict = {
    'A': proc.T_A,
    'B': proc.T_B,
    'C': proc.T_C
}
belief_states = []
for i in tqdm(range(len(obs))):
    # propagate
    dst = dst @ T_dict[obs[i]]
    dst = dst / dst.sum()
    # if i >= burn_in:
    #     belief_states.append(dst)
    belief_states.append(dst)
belief_states = t.stack(belief_states, dim=0)

# plot the belief states
import matplotlib.pyplot as plt
import numpy as np

# Convert from barycentric to cartesian coordinates
def barycentric_to_cartesian(points):
    # Constants for equilateral triangle
    a = 1.0  # Side length
    
    # Vertices of equilateral triangle
    v1 = np.array([0, 0])  # Bottom left
    v2 = np.array([a, 0])  # Bottom right 
    v3 = np.array([a/2, a*np.sqrt(3)/2])  # Top
    
    # Convert each point
    cartesian = points[:, 0:1] * v2 + points[:, 1:2] * v3
    return cartesian

# Convert belief states to numpy for plotting
belief_states_np = belief_states.numpy()

def plot_belief_states_simplex(belief_states_np):
    # Convert to cartesian coordinates
    cart_coords = barycentric_to_cartesian(belief_states_np)

    # Create figure
    plt.figure(figsize=(10, 10))

    # Plot triangle outline
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    plt.plot(triangle[:, 0], triangle[:, 1], 'k-', alpha=0.5)

    # Plot points with RGB colors matching their barycentric coordinates
    plt.scatter(cart_coords[:, 0], cart_coords[:, 1], c=belief_states_np, alpha=0.5, s=1)

    # Add vertex labels
    plt.text(-0.05, -0.05, 'State 0', fontsize=12)
    plt.text(1.05, -0.05, 'State 1', fontsize=12)
    plt.text(0.45, 0.9, 'State 2', fontsize=12)

    plt.axis('equal')
    plt.title('Belief States on Probability Simplex')
    plt.axis('off')
    plt.show()

plot_belief_states_simplex(belief_states_np)


#%% Sample code for probing the transformer model
from transformer_lens import HookedTransformer, HookedTransformerConfig
cfg = HookedTransformerConfig(
    n_layers=4,
    d_model=64,
    n_ctx=10,
    d_vocab=3,
    n_heads=1,
    d_head=8,
    d_mlp=256,
    act_fn="relu",
)

model = HookedTransformer(cfg)

#%% testing the model and probing the residue stream

# reuse the same dataset
# Convert observations to indices 0,1,2
obs_indices = t.tensor([0 if c == 'A' else 1 if c == 'B' else 2 for c in tqdm(obs)], dtype=t.long)

#%% get the residue stream
trunc = 100000
# Reshape to process sequences of length n_ctx
n_sequences = trunc // cfg.n_ctx
obs_trunc = obs_indices[:n_sequences * cfg.n_ctx].reshape(n_sequences, cfg.n_ctx)  # (n_sequences, n_ctx)

logits, cache = model.run_with_cache(
    obs_trunc,
    names_filter=[f"blocks.{i}.hook_resid_post" for i in range(4)]
)

#%% Train linear probes from the model
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

resid = cache["blocks.3.hook_resid_post"]

# the new shape of the resid
flat_resid = resid.flatten(0, 1).cpu().numpy()


# Apparently this is an unsupervides method
def visualize_belief_state_geometry(residual_stream, belief_states):
    pca = PCA(n_components=3)
    residual_pca = pca.fit_transform(residual_stream)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(residual_pca[:, 0], residual_pca[:, 1], residual_pca[:, 2], c=belief_states, cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Belief State Geometry in Residual Stream')
    plt.colorbar(scatter, label='Belief State')
    plt.show()
    return residual_pca


residual_pca = visualize_belief_state_geometry(flat_resid, belief_states)

plot_belief_states_simplex(residual_pca)



# %% using linear probes 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Prepare data
X = flat_resid  # residual vectors
y = belief_states.numpy()  # true belief states

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_tensor = t.FloatTensor(X_train)
y_train_tensor = t.FloatTensor(y_train)
X_test_tensor = t.FloatTensor(X_test)
y_test_tensor = t.FloatTensor(y_test)

# Solve least squares problem directly
outcome = t.linalg.lstsq(X_train_tensor, y_train_tensor)
W = outcome[0]
print(f"Linear probe shape: {W.shape}")

# Evaluate on test set
test_predictions = X_test_tensor @ W
test_mse = t.mean((test_predictions - y_test_tensor) ** 2)
test_r2 = r2_score(y_test, test_predictions.numpy())

print(f'\nTest MSE: {test_mse.item():.4f}')
print(f'Test R²: {test_r2:.4f}')

# Shuffle control
shuffled_y = y[np.random.permutation(len(y))]
shuffled_r2 = r2_score(y_test, shuffled_y[:len(y_test)])
print(f'Shuffle Control R²: {shuffled_r2:.4f}')

# Visual check: Project decoded beliefs into 2D
def plot_decoded_beliefs(residual_stream, W, belief_states):
    # Get predictions for all data
    decoded_beliefs = (t.FloatTensor(residual_stream) @ W).numpy()
    
    # Convert to cartesian coordinates
    cart_coords = barycentric_to_cartesian(decoded_beliefs)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot triangle outline
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    plt.plot(triangle[:, 0], triangle[:, 1], 'k-', alpha=0.5)
    
    # Scale decoded beliefs to 0-1 range for RGB
    min_vals = decoded_beliefs.min(axis=0)
    max_vals = decoded_beliefs.max(axis=0)
    scaled_beliefs = (decoded_beliefs - min_vals) / (max_vals - min_vals)
    
    # Plot points with RGB colors matching their barycentric coordinates
    scatter = plt.scatter(cart_coords[:, 0], cart_coords[:, 1], 
                         c=scaled_beliefs, 
                         alpha=0.5, s=1,
                         cmap='viridis')  # Use viridis colormap
    
    # Add vertex labels
    plt.text(-0.05, -0.05, 'State 0', fontsize=12)
    plt.text(1.05, -0.05, 'State 1', fontsize=12)
    plt.text(0.45, 0.9, 'State 2', fontsize=12)
    
    plt.axis('equal')
    plt.title('Decoded Belief States on Probability Simplex')
    
    # Customize colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Decoded Belief Value')
    cbar.ax.tick_params(labelsize=10)
    
    plt.axis('off')
    plt.show()

# Plot decoded beliefs
plot_decoded_beliefs(flat_resid, W, belief_states)


