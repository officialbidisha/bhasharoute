"""Train approach C (multilingual Transformer) with an explicit PyTorch loop.

    uv run python -m bhasharoute.models.train_transformer --mode finetune [--augment]
    uv run python -m bhasharoute.models.train_transformer --mode frozen

finetune: AdamW with separate learning rates for encoder (3e-5) and freshly initialised
heads (1e-3), linear warm-up (6%) then linear decay, gradient clipping at 1.0, early
stopping on dev macro-F1, checkpoint of the best epoch only.

frozen: the encoder runs once (no grad) to cache pooled features; an MLP head is then
trained on those features. Reported separately - this is NOT fine-tuning.

Class imbalance: scenario frequencies differ ~6x (calendar vs audio), so cross-entropy
uses inverse-sqrt-frequency class weights (full inverse frequency over-corrects and
hurts precision on the large classes). The side-effect BCE uses pos_weight =
sqrt(n_neg / n_pos) for the same reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bhasharoute.config import ARTIFACTS_DIR, SEED
from bhasharoute.data.splits import dev_rows, load_rows, training_rows
from bhasharoute.labels import SCENARIO_TO_ID
from bhasharoute.models.encoder import Collator, RequestDataset, RouterNet, load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Best effort: some MPS kernels have no deterministic implementation, so warn only.
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weights(labels: list[int], k: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels), minlength=k).astype(float)
    w = np.sqrt(counts.sum() / (k * np.maximum(counts, 1)))
    return torch.tensor(w / w.mean(), dtype=torch.float)


def side_pos_weight(sides: list[float]) -> torch.Tensor:
    pos = sum(sides)
    return torch.tensor(((len(sides) - pos) / max(pos, 1)) ** 0.5)


def warmup_linear(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict_logits(model: RouterNet, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scen, side = [], []
    for b in loader:
        s, e = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        scen.append(s.float().cpu())
        side.append(e.float().cpu())
    return torch.cat(scen).numpy(), torch.cat(side).numpy()


def dev_metrics(scen_logits: np.ndarray, side_logits: np.ndarray, ds: RequestDataset) -> dict:
    y, s = np.array(ds.scenarios), np.array(ds.side)
    return {"dev_macro_f1": float(f1_score(y, scen_logits.argmax(1), average="macro")),
            "dev_acc": float((scen_logits.argmax(1) == y).mean()),
            "dev_side_auroc": float(roc_auc_score(s, side_logits))}


def _param_groups(model: RouterNet, lr: float, head_lr: float, wd: float) -> list[dict]:
    groups = {k: [] for k in ("enc_decay", "enc_nodecay", "head_decay", "head_nodecay")}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if name.startswith("heads.") else "enc"
        nodecay = p.ndim == 1  # biases and LayerNorm weights are not decayed
        groups[f"{part}_{'nodecay' if nodecay else 'decay'}"].append(p)
    return [
        {"params": groups["enc_decay"], "lr": lr, "weight_decay": wd},
        {"params": groups["enc_nodecay"], "lr": lr, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": head_lr, "weight_decay": wd},
        {"params": groups["head_nodecay"], "lr": head_lr, "weight_decay": 0.0},
    ]


def _save(model: RouterNet, out_dir, cfg: dict) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.trainable_state_dict(), out_dir / "model.pt")
    digest = hashlib.sha256((out_dir / "model.pt").read_bytes()).hexdigest()[:12]
    (out_dir / "config.json").write_text(json.dumps({**cfg, "weights_sha256_12": digest}, indent=2))
    return digest


def train_finetune(args, model, tok, tr, dv, device, out_dir, cfg) -> dict:
    tr_ds, dv_ds = RequestDataset(tr), RequestDataset(dv)
    collate = Collator(tok, args.max_len)
    gen = torch.Generator().manual_seed(args.seed)
    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, collate_fn=collate, generator=gen)
    dv_loader = DataLoader(dv_ds, batch_size=128, shuffle=False, collate_fn=collate)

    ce = nn.CrossEntropyLoss(weight=class_weights(tr_ds.scenarios, len(SCENARIO_TO_ID)).to(device))
    bce = nn.BCEWithLogitsLoss(pos_weight=side_pos_weight(tr_ds.side).to(device))
    opt = torch.optim.AdamW(_param_groups(model, args.lr, args.head_lr, 0.01))
    total = args.epochs * len(tr_loader)
    sched = warmup_linear(opt, int(0.06 * total), total)
    trainable = [p for p in model.parameters() if p.requires_grad]

    best, bad, step, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for b in tr_loader:
            scen_logits, side_logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = ce(scen_logits, b["scenario"].to(device)) + args.side_weight * bce(side_logits, b["side"].to(device))
            loss.backward()                                   # gradients for every trainable weight
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()                                        # AdamW update
            sched.step()                                      # warm-up / decay schedule
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {run_loss / 100:.4f} "
                      f"grad_norm {gnorm:.2f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{100 * args.bs / (time.time() - t0):.0f} ex/s", flush=True)
                run_loss, t0 = 0.0, time.time()
        m = dev_metrics(*predict_logits(model, dv_loader, device), dv_ds)
        m["epoch"] = epoch
        history.append(m)
        print("dev", json.dumps(m), flush=True)
        if m["dev_macro_f1"] > best:
            best, bad = m["dev_macro_f1"], 0
            cfg["best_epoch"] = epoch
            _save(model, out_dir, cfg)
        else:
            bad += 1
            if bad > args.patience:
                print(f"early stop after epoch {epoch}", flush=True)
                break
    return {"history": history, "best_dev_macro_f1": best}


@torch.no_grad()
def _features(model: RouterNet, ds: RequestDataset, tok, device, max_len: int) -> torch.Tensor:
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=Collator(tok, max_len))
    return torch.cat([model.encode(b["input_ids"].to(device), b["attention_mask"].to(device)).cpu()
                      for b in loader])


def train_frozen(args, model, tok, tr, dv, device, out_dir, cfg) -> dict:
    tr_ds, dv_ds = RequestDataset(tr), RequestDataset(dv)
    t0 = time.time()
    f_tr, f_dv = _features(model, tr_ds, tok, device, args.max_len), _features(model, dv_ds, tok, device, args.max_len)
    print(f"cached features {tuple(f_tr.shape)} in {time.time() - t0:.0f}s", flush=True)
    heads = model.heads
    y_tr, s_tr = torch.tensor(tr_ds.scenarios), torch.tensor(tr_ds.side)
    loader = DataLoader(TensorDataset(f_tr, y_tr, s_tr), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    ce = nn.CrossEntropyLoss(weight=class_weights(tr_ds.scenarios, len(SCENARIO_TO_ID)).to(device))
    bce = nn.BCEWithLogitsLoss(pos_weight=side_pos_weight(tr_ds.side).to(device))
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.01)
    best, bad, history = -1.0, 0, []
    for epoch in range(1, 61):
        heads.train()
        for x, y, s in loader:
            scen_logits, side_logits = heads(x.to(device))
            loss = ce(scen_logits, y.to(device)) + args.side_weight * bce(side_logits, s.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        heads.eval()
        with torch.no_grad():
            sl, el = heads(f_dv.to(device))
        m = dev_metrics(sl.cpu().numpy(), el.cpu().numpy(), dv_ds)
        m["epoch"] = epoch
        history.append(m)
        if epoch % 5 == 0:
            print("dev", json.dumps(m), flush=True)
        if m["dev_macro_f1"] > best:
            best, bad = m["dev_macro_f1"], 0
            cfg["best_epoch"] = epoch
            _save(model, out_dir, cfg)
        else:
            bad += 1
            if bad > 5:
                print(f"early stop after epoch {epoch}", flush=True)
                break
    return {"history": history, "best_dev_macro_f1": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "finetune"], required=True)
    ap.add_argument("--encoder", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--side-weight", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train/dev rows")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    rows = load_rows()
    tr, dv = training_rows(rows, args.augment), dev_rows(rows)
    if args.limit:
        tr, dv = random.Random(args.seed).sample(tr, args.limit), dv[: args.limit // 4]
    name = f"C_xlmr_{args.mode}" + ("_aug" if args.augment else "") + ("_debug" if args.limit else "")
    out_dir = ARTIFACTS_DIR / name

    tok = load_tokenizer(args.encoder)
    model = RouterNet(args.encoder, len(SCENARIO_TO_ID), args.mode,
                      head_hidden=256 if args.mode == "frozen" else 0).to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = {"name": name, "encoder": args.encoder,
           "revision": getattr(model.encoder.config, "_commit_hash", None), "mode": args.mode,
           "augment": args.augment, "head_hidden": 256 if args.mode == "frozen" else 0,
           "n_scenarios": len(SCENARIO_TO_ID), "labels": list(SCENARIO_TO_ID), "max_len": args.max_len,
           "params_total": n_all, "params_trainable": n_train, "n_train": len(tr), "n_dev": len(dv),
           "hparams": {k: v for k, v in vars(args).items() if k not in ("mode", "encoder", "augment")},
           "device": str(device), "torch": torch.__version__}
    print(f"{name}: {n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M params trainable; "
          f"{len(tr)} train / {len(dv)} dev rows on {device}", flush=True)

    t0 = time.time()
    fit = train_frozen if args.mode == "frozen" else train_finetune
    result = fit(args, model, tok, tr, dv, device, out_dir, cfg)
    result["train_minutes"] = round((time.time() - t0) / 60, 1)
    (out_dir / "train_log.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
