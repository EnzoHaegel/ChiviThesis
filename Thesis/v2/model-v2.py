"""
Bond Direction Prediction Model V2 - Multi-Horizon Classification
OPTIMIZED VERSION - Preloads all texts to RAM for GPU efficiency.
Predicts UP/DOWN/NEUTRAL direction for J+1, J+2, J+5, J+10.
"""

import os
import glob
import re
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent.parent
CSV_DIR = BASE_DIR / "csv_raw"
SEC_DIR = BASE_DIR / "sec_10q" / "sec-edgar-filings"
V1_DIR = BASE_DIR / "v1"
V2_DIR = BASE_DIR / "v2"
MODEL_SAVE_PATH = V2_DIR / "best_model.pt"
FILINGS_CACHE_PATH = V1_DIR / "filings_cache.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LENGTH = 256
BATCH_SIZE = 48
LEARNING_RATE = 3e-5
EPOCHS = 15
DROPOUT = 0.2
TRAIN_SPLIT = 0.85
EARLY_STOPPING_PATIENCE = 4

DIRECTION_THRESHOLD = 0.005
NUM_CLASSES = 3
CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]

TARGET_HORIZONS = {
    "J+1": "bond_return_plus_day_1",
    "J+2": "bond_return_plus_day_2",
    "J+5": "bond_return_plus_day_5",
    "J+10": "bond_return_plus_day_10"
}


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def return_to_class(return_value: float, threshold: float = DIRECTION_THRESHOLD) -> int:
    if pd.isna(return_value):
        return -1
    if return_value < -threshold:
        return 0
    elif return_value > threshold:
        return 2
    else:
        return 1


def load_csv_data() -> pd.DataFrame:
    csv_files = sorted(glob.glob(str(CSV_DIR / "*.csv")))
    print(f"Found {len(csv_files)} CSV files")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, low_memory=False)
        year = re.search(r'(\d{4})', os.path.basename(csv_file))
        if year:
            df['source_year'] = int(year.group(1))
        dfs.append(df)
        print(f"  Loaded {os.path.basename(csv_file)}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(combined)}")
    return combined


def load_filings_cache() -> Dict[str, List[Tuple[datetime, str]]]:
    if not FILINGS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Filings cache not found at {FILINGS_CACHE_PATH}. "
            "Please run v1/model-v1.py first to generate the cache."
        )

    print(f"Loading SEC filings from cache: {FILINGS_CACHE_PATH}")
    with open(FILINGS_CACHE_PATH, 'r') as f:
        cached = json.load(f)

    filings_map = {}
    for ticker, filings in cached.items():
        filings_map[ticker] = [
            (datetime.fromisoformat(date_str), path)
            for date_str, path in filings
        ]
    print(f"Loaded {len(filings_map)} tickers from cache")
    return filings_map


def find_matching_filing(
    ticker: str,
    rdq_date: datetime,
    filings_map: Dict[str, List[Tuple[datetime, str]]]
) -> Optional[str]:
    if ticker not in filings_map:
        return None

    filings = filings_map[ticker]
    matching_path = None

    for filing_date, file_path in filings:
        if filing_date < rdq_date:
            matching_path = file_path
        else:
            break

    return matching_path


def load_risk_factors_text(file_path: str, max_chars: int = 10000) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        text = ' '.join(lines[1:]).strip()
        return text[:max_chars]


def prepare_dataset(df: pd.DataFrame, filings_map: Dict) -> pd.DataFrame:
    print("Matching bond observations with SEC filings...")
    print(f"Target horizons: {list(TARGET_HORIZONS.keys())}")

    matched_data = []
    target_cols = list(TARGET_HORIZONS.values())

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching data"):
        ticker = str(row.get('tic', '')).strip()
        rdq = row.get('rdq')

        if pd.isna(rdq) or not ticker:
            continue

        has_any_target = any(pd.notna(row.get(col)) for col in target_cols)
        if not has_any_target:
            continue

        try:
            rdq_date = pd.to_datetime(rdq)
        except Exception:
            continue

        file_path = find_matching_filing(ticker, rdq_date, filings_map)
        if file_path is None:
            continue

        record = {
            'ticker': ticker,
            'rdq': rdq_date,
            'file_path': file_path,
        }

        for horizon_name, col_name in TARGET_HORIZONS.items():
            return_val = row.get(col_name)
            record[f'return_{horizon_name}'] = return_val
            record[f'class_{horizon_name}'] = return_to_class(return_val)

        matched_data.append(record)

    result_df = pd.DataFrame(matched_data)
    print(f"Matched {len(result_df)} observations")

    print(f"\nClass distribution per horizon:")
    for horizon_name in TARGET_HORIZONS.keys():
        class_col = f'class_{horizon_name}'
        valid_mask = result_df[class_col] >= 0
        class_counts = result_df.loc[valid_mask,
                                     class_col].value_counts().sort_index()
        valid_count = valid_mask.sum()
        print(f"\n  {horizon_name} ({valid_count} valid samples):")
        for i, name in enumerate(CLASS_NAMES):
            count = class_counts.get(i, 0)
            pct = count / valid_count * 100 if valid_count > 0 else 0
            print(f"    {name}: {count} ({pct:.1f}%)")

    return result_df


TOKENIZED_CACHE_PATH = V2_DIR / "tokenized_cache.pt"


def preload_and_tokenize_all(
    data: pd.DataFrame,
    tokenizer,
    max_length: int = MAX_SEQ_LENGTH
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Preload all texts and tokenize them - uses disk cache for speed."""

    if TOKENIZED_CACHE_PATH.exists():
        print(f"Loading tokenized cache from {TOKENIZED_CACHE_PATH}...")
        cache_data = torch.load(TOKENIZED_CACHE_PATH, weights_only=False)
        print(f"Loaded {len(cache_data)} tokenized files from cache")
        return cache_data

    unique_paths = list(data['file_path'].unique())
    print(f"Tokenizing {len(unique_paths)} unique files...")

    tokenized_cache = {}
    batch_size = 16

    for i in tqdm(range(0, len(unique_paths), batch_size), desc="Tokenizing"):
        batch_paths = unique_paths[i:i+batch_size]
        batch_texts = []

        for file_path in batch_paths:
            try:
                text = load_risk_factors_text(file_path)
                batch_texts.append(text[:50000])
            except Exception:
                batch_texts.append("")

        encodings = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        for j, path in enumerate(batch_paths):
            tokenized_cache[path] = {
                'input_ids': encodings['input_ids'][j].clone(),
                'attention_mask': encodings['attention_mask'][j].clone()
            }

        del encodings, batch_texts
        if i % 500 == 0:
            gc.collect()

    print(f"Saving tokenized cache to {TOKENIZED_CACHE_PATH}...")
    torch.save(tokenized_cache, TOKENIZED_CACHE_PATH)

    print(f"Tokenized {len(tokenized_cache)} files")
    return tokenized_cache


class PreloadedBondDataset(Dataset):
    """Dataset with pre-tokenized data for maximum GPU throughput."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenized_cache: Dict[str, Dict[str, torch.Tensor]]
    ):
        self.data = data.reset_index(drop=True)
        self.tokenized_cache = tokenized_cache
        self.horizon_names = list(TARGET_HORIZONS.keys())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        file_path = row['file_path']

        cached = self.tokenized_cache[file_path]

        labels = []
        masks = []
        for horizon_name in self.horizon_names:
            class_val = row[f'class_{horizon_name}']
            if class_val >= 0:
                labels.append(class_val)
                masks.append(1.0)
            else:
                labels.append(0)
                masks.append(0.0)

        return {
            'input_ids': cached['input_ids'],
            'attention_mask': cached['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'label_masks': torch.tensor(masks, dtype=torch.float32)
        }


class MultiHorizonClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        num_horizons: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_horizons = num_horizons

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.encoder.layer[-2:].parameters():
            param.requires_grad = True

        self.shared_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )
            for _ in range(num_horizons)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        shared_features = self.shared_layer(cls_output)

        all_logits = []
        for head in self.horizon_heads:
            logits = head(shared_features)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_size, num_horizons, num_classes = logits.shape

    total_loss = 0.0
    valid_count = 0

    for h in range(num_horizons):
        h_logits = logits[:, h, :]
        h_labels = labels[:, h]
        h_mask = masks[:, h]

        if h_mask.sum() > 0:
            if class_weights is not None:
                ce_loss = F.cross_entropy(
                    h_logits, h_labels, weight=class_weights, reduction='none')
            else:
                ce_loss = F.cross_entropy(h_logits, h_labels, reduction='none')

            masked_loss = (ce_loss * h_mask).sum()
            total_loss += masked_loss
            valid_count += h_mask.sum()

    if valid_count > 0:
        return total_loss / valid_count
    return torch.tensor(0.0, device=logits.device)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    class_weights: torch.Tensor
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    horizon_correct = {h: 0 for h in TARGET_HORIZONS.keys()}
    horizon_total = {h: 0 for h in TARGET_HORIZONS.keys()}
    horizon_names = list(TARGET_HORIZONS.keys())

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        label_masks = batch['label_masks'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            logits = model(input_ids, attention_mask)
            loss = masked_cross_entropy(
                logits, labels, label_masks, class_weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=-1)
        for h_idx, h_name in enumerate(horizon_names):
            mask = label_masks[:, h_idx] > 0
            if mask.sum() > 0:
                horizon_correct[h_name] += (preds[mask, h_idx]
                                            == labels[mask, h_idx]).sum().item()
                horizon_total[h_name] += mask.sum().item()

    accuracies = {}
    for h_name in horizon_names:
        if horizon_total[h_name] > 0:
            accuracies[h_name] = horizon_correct[h_name] / \
                horizon_total[h_name]
        else:
            accuracies[h_name] = 0.0

    return total_loss / max(num_batches, 1), accuracies


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor
) -> Tuple[float, Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    horizon_names = list(TARGET_HORIZONS.keys())
    horizon_preds = {h: [] for h in horizon_names}
    horizon_labels = {h: [] for h in horizon_names}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(
                device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            label_masks = batch['label_masks'].to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(input_ids, attention_mask)
                loss = masked_cross_entropy(
                    logits, labels, label_masks, class_weights)

            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1)
            for h_idx, h_name in enumerate(horizon_names):
                mask = label_masks[:, h_idx] > 0
                if mask.sum() > 0:
                    horizon_preds[h_name].extend(
                        preds[mask, h_idx].cpu().numpy())
                    horizon_labels[h_name].extend(
                        labels[mask, h_idx].cpu().numpy())

    accuracies = {}
    results = {}
    for h_name in horizon_names:
        preds_arr = np.array(horizon_preds[h_name])
        labels_arr = np.array(horizon_labels[h_name])
        if len(preds_arr) > 0:
            accuracies[h_name] = (preds_arr == labels_arr).mean()
            results[h_name] = (preds_arr, labels_arr)
        else:
            accuracies[h_name] = 0.0
            results[h_name] = (np.array([]), np.array([]))

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, accuracies, results


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Dict[str, List[float]],
    val_accs: Dict[str, List[float]],
    save_path: str
):
    num_horizons = len(TARGET_HORIZONS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(train_losses, label='Train Loss', marker='o')
    axes[0, 0].plot(val_losses, label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for idx, (h_name, color) in enumerate(zip(TARGET_HORIZONS.keys(), colors)):
        if h_name in train_accs:
            axes[0, 1].plot(
                train_accs[h_name], label=f'{h_name}', color=color, marker='o', linestyle='-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy by Horizon')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    for idx, (h_name, color) in enumerate(zip(TARGET_HORIZONS.keys(), colors)):
        if h_name in val_accs:
            axes[1, 0].plot(
                val_accs[h_name], label=f'{h_name}', color=color, marker='s', linestyle='--')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Validation Accuracy by Horizon')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    final_accs = [val_accs[h][-1] if h in val_accs and len(val_accs[h]) > 0 else 0
                  for h in TARGET_HORIZONS.keys()]
    bars = axes[1, 1].bar(list(TARGET_HORIZONS.keys()),
                          final_accs, color=colors)
    axes[1, 1].axhline(y=0.33, color='red', linestyle='--',
                       label='Random (33%)')
    axes[1, 1].set_xlabel('Horizon')
    axes[1, 1].set_ylabel('Final Accuracy')
    axes[1, 1].set_title('Final Validation Accuracy by Horizon')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    for bar, acc in zip(bars, final_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{acc:.1%}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_confusion_matrices(results: Dict[str, Tuple[np.ndarray, np.ndarray]], save_path: str):
    num_horizons = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (h_name, (preds, labels)) in enumerate(results.items()):
        if len(preds) == 0:
            continue

        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
        cm_normalized = cm.astype(
            'float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

        im = axes[idx].imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)

        axes[idx].set_xticks(range(NUM_CLASSES))
        axes[idx].set_yticks(range(NUM_CLASSES))
        axes[idx].set_xticklabels(CLASS_NAMES)
        axes[idx].set_yticklabels(CLASS_NAMES)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].set_title(f'{h_name} (n={len(preds)})')

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                text = f'{cm[i, j]}\n({cm_normalized[i, j]:.0%})'
                color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                axes[idx].text(j, i, text, ha='center',
                               va='center', color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices to {save_path}")


def main():
    print("=" * 60)
    print("Bond Direction Prediction V2 - OPTIMIZED Multi-Horizon")
    print("=" * 60)
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

    print(f"\nConfiguration:")
    print(f"  Direction threshold: ±{DIRECTION_THRESHOLD*100:.1f}%")
    print(f"  Target horizons: {list(TARGET_HORIZONS.keys())}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")

    print("\n[1/5] Loading CSV data...")
    csv_data = load_csv_data()

    print("\n[2/5] Loading SEC filings cache...")
    try:
        filings_map = load_filings_cache()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print("\n[3/5] Preparing dataset...")
    matched_data = prepare_dataset(csv_data, filings_map)

    if len(matched_data) == 0:
        print("ERROR: No matched data found.")
        return

    print("\n[4/5] Initializing model and tokenizer...")
    clear_gpu_memory()

    try:
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = MultiHorizonClassifier(model_name="yiyanghkust/finbert-tone")
    except Exception as e:
        print(f"FinBERT not available ({e}), falling back to RoBERTa...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = MultiHorizonClassifier(model_name="roberta-base")

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n[5/5] Pre-loading and tokenizing all texts...")
    tokenized_cache = preload_and_tokenize_all(matched_data, tokenizer)

    train_data, val_data = train_test_split(
        matched_data,
        test_size=1 - TRAIN_SPLIT,
        random_state=42
    )
    print(
        f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    train_dataset = PreloadedBondDataset(train_data, tokenized_cache)
    val_dataset = PreloadedBondDataset(val_data, tokenized_cache)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    class_weights = torch.tensor(
        [2.0, 0.5, 1.2], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-7
    )
    scaler = GradScaler('cuda')

    train_losses, val_losses = [], []
    train_accs = {h: [] for h in TARGET_HORIZONS.keys()}
    val_accs = {h: [] for h in TARGET_HORIZONS.keys()}
    best_avg_acc = 0.0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Batches per epoch: {len(train_loader)}")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, DEVICE, class_weights
        )
        train_losses.append(train_loss)
        for h_name, acc in train_acc.items():
            train_accs[h_name].append(acc)

        val_loss, val_acc, val_results = validate(
            model, val_loader, DEVICE, class_weights
        )
        val_losses.append(val_loss)
        for h_name, acc in val_acc.items():
            val_accs[h_name].append(acc)

        scheduler.step()

        avg_val_acc = np.mean(list(val_acc.values()))

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Validation Accuracy by Horizon:")
        for h_name in TARGET_HORIZONS.keys():
            t_acc = train_acc.get(h_name, 0)
            v_acc = val_acc.get(h_name, 0)
            print(f"  {h_name}: Train {t_acc:.1%} | Val {v_acc:.1%}")
        print(
            f"Average Val Acc: {avg_val_acc:.2%} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'avg_val_acc': avg_val_acc,
                'val_loss': val_loss,
                'class_names': CLASS_NAMES,
                'horizon_names': list(TARGET_HORIZONS.keys()),
                'threshold': DIRECTION_THRESHOLD
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (avg_acc: {avg_val_acc:.2%})")
        else:
            patience_counter += 1
            print(
                f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        str(V2_DIR / "training_curves.png")
    )

    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    checkpoint = torch.load(
        MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, final_accs, final_results = validate(
        model, val_loader, DEVICE, class_weights)

    print(f"\nBest Average Validation Accuracy: {best_avg_acc:.2%}")
    print(f"\nFinal Accuracy by Horizon:")
    for h_name, acc in final_accs.items():
        print(f"  {h_name}: {acc:.2%}")

    print("\nClassification Reports:")
    for h_name, (preds, labels) in final_results.items():
        if len(preds) > 0:
            print(f"\n=== {h_name} ===")
            print(classification_report(labels, preds,
                  target_names=CLASS_NAMES, zero_division=0))

    plot_confusion_matrices(final_results, str(
        V2_DIR / "confusion_matrices.png"))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best average validation accuracy: {best_avg_acc:.2%}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
