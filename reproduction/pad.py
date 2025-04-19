import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm
import wandb
from dataclasses import dataclass
import os

# Data generation from hidden markov model
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
            [0.00375, 0.06750, 0.00375],
            [0.00375, 0.00375, 0.76500]
        ], dtype=t.float32)
        self.tokens = ['A', 'B', 'C']
        self.num_states = 3
        self.T_A = self.T_A / self.T_A.sum(dim=1, keepdim=True)
        self.T_B = self.T_B / self.T_B.sum(dim=1, keepdim=True)
        self.T_C = self.T_C / self.T_C.sum(dim=1, keepdim=True)

    def generate_sequence(self, length):
        states = t.zeros(length, dtype=t.long)
        observations = []
        current_state = t.randint(0, self.num_states, (1,)).item()
        for t_idx in range(length):
            states[t_idx] = current_state
            T_choice = t.randint(0, 3, (1,)).item()
            T = self.T_A if T_choice == 0 else self.T_B if T_choice == 1 else self.T_C
            probs = T[current_state]
            token_idx = t.multinomial(probs, 1).item()
            token = self.tokens[token_idx]
            observations.append(token)
            current_state = t.multinomial(probs, 1).item()
        return states, observations

# Training configurations
@dataclass
class TrainConfig:
    learning_rate: float = 1e-2
    batch_size: int = 64
    n_epoch = 10_000
    weight_decay: float | None = None
    checkpoint_every: int = 100
    device: str = "cuda" if t.cuda.is_available() else "cpu"


def generate_data(proc, batch_size, n_ctx, use_tqdm=False):
    src = []
    tgt = []
    
    for _ in tqdm(range(batch_size), desc="Generating data", disable=not use_tqdm):
        sts, obs = proc.generate_sequence(n_ctx + 1)
        if isinstance(obs[0], str):
            token_to_int = {token: i for i, token in enumerate(proc.tokens)}
            obs = t.tensor([token_to_int[token] for token in obs])
        src.append(obs[:-1])
        tgt.append(obs[1:])
    
    src = t.stack(src)
    tgt = t.stack(tgt)
    return src, tgt

def train_model(
    cfg: HookedTransformerConfig,
    tcfg: TrainConfig,
    save_weights_dir: str | None = None,
    proc: Mess3Process | None = None,
) -> HookedTransformer:
    if proc is None:
        proc = Mess3Process()
    # initialize model
    model = HookedTransformer(cfg)
    model.to(tcfg.device)
    
    # Optimizer
    optimizer = t.optim.Adam(
        model.parameters(),
        lr=tcfg.learning_rate,
    )
    run = wandb.init(
        project="toy-transformer-markov-chain",
        name=f"transformer-epochs-{tcfg.n_epoch}",
    )
    pbar = tqdm(range(tcfg.n_epoch))
    # training loop begins
    for epoch in pbar:
        model.train()
        src, tgt = generate_data(proc, tcfg.batch_size, cfg.n_ctx)
        src = src.to(tcfg.device)
        tgt = tgt.to(tcfg.device)
        optimizer.zero_grad()
        logits = model(src)
        loss = t.nn.functional.cross_entropy(logits.view(-1, cfg.d_vocab), tgt.view(-1))
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        
        
        # after each epoch of updates (total number of updates = n_epoch)
        

        if epoch % tcfg.checkpoint_every == 0 or epoch == tcfg.n_epoch - 1:
            model.eval()
            with t.no_grad():
                src, tgt = generate_data(proc, tcfg.batch_size, cfg.n_ctx)
                src = src.to(tcfg.device)
                tgt = tgt.to(tcfg.device)
                loss = t.nn.functional.cross_entropy(model(src).view(-1, cfg.d_vocab), tgt.view(-1))
                val_loss = loss.item()
            
            run.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, step=epoch)
            
            # save the model
            if save_weights_dir is not None:
                os.makedirs(save_weights_dir, exist_ok=True)
                t.save(model.state_dict(), f"{save_weights_dir}/model_epoch_{epoch}.pth")
            
            pbar.set_description(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
    run.finish()
    return model



if __name__ == "__main__":
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
    
    tcfg = TrainConfig()
    
    model = HookedTransformer(cfg)
    model.to(tcfg.device)
    
    train_model(cfg, tcfg, save_weights_dir="weights")
    











