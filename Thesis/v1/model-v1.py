"""
Bond Price Prediction Model from SEC 10-Q Risk Factors
OPTIMIZED VERSION - Preloads all texts to RAM for GPU efficiency.
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
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent.parent
CSV_DIR = BASE_DIR / "csv_raw"
SEC_DIR = BASE_DIR / "sec_10q" / "sec-edgar-filings"
MODEL_SAVE_PATH = BASE_DIR / "v1" / "best_model.pt"
FILINGS_CACHE_PATH = BASE_DIR / "v1" / "filings_cache.json"
TOKENIZED_CACHE_PATH = BASE_DIR / "v1" / "tokenized_cache.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 20
DROPOUT = 0.3
TRAIN_SPLIT = 0.85
EARLY_STOPPING_PATIENCE = 5

TARGET_COLUMNS = [
    "bond_return",
    "bond_return_plus_day_1",
    "bond_return_plus_day_2",
    "bond_return_plus_day_5",
    "bond_return_plus_day_10"
]


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


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


def scan_sec_filings(use_cache: bool = True) -> Dict[str, List[Tuple[datetime, str]]]:
    if use_cache and FILINGS_CACHE_PATH.exists():
        print("Loading SEC filings from cache...")
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

    filings_map = {}
    ticker_dirs = [d for d in SEC_DIR.iterdir() if d.is_dir()]
    print(f"Scanning {len(ticker_dirs)} ticker directories...")

    for ticker_dir in tqdm(ticker_dirs, desc="Scanning SEC filings"):
        ticker = ticker_dir.name
        filings = []

        form_dir = ticker_dir / "10-Q"
        if not form_dir.exists():
            continue

        for filing_dir in form_dir.iterdir():
            if not filing_dir.is_dir():
                continue

            rf_file = filing_dir / "full-submission_rf.txt"
            if not rf_file.exists():
                continue

            try:
                with open(rf_file, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()

                if len(first_line) == 8 and first_line.isdigit():
                    filing_date = datetime.strptime(first_line, "%Y%m%d")
                    filings.append((filing_date, str(rf_file)))
            except Exception:
                continue

        if filings:
            filings.sort(key=lambda x: x[0])
            filings_map[ticker] = filings

    print(f"Found filings for {len(filings_map)} tickers")

    print("Saving cache for future runs...")
    cache_data = {}
    for ticker, filings in filings_map.items():
        cache_data[ticker] = [
            (date.isoformat(), path) for date, path in filings
        ]
    with open(FILINGS_CACHE_PATH, 'w') as f:
        json.dump(cache_data, f)

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


def load_risk_factors_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        return ' '.join(lines[1:]).strip()


def prepare_dataset(df: pd.DataFrame, filings_map: Dict) -> pd.DataFrame:
    print("Matching bond observations with SEC filings...")

    matched_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching data"):
        ticker = str(row.get('tic', '')).strip()
        rdq = row.get('rdq')

        if pd.isna(rdq) or not ticker:
            continue

        try:
            rdq_date = pd.to_datetime(rdq)
        except Exception:
            continue

        has_targets = any(pd.notna(row.get(col)) for col in TARGET_COLUMNS)
        if not has_targets:
            continue

        file_path = find_matching_filing(ticker, rdq_date, filings_map)
        if file_path is None:
            continue

        targets = [row.get(col, np.nan) for col in TARGET_COLUMNS]

        matched_data.append({
            'ticker': ticker,
            'rdq': rdq_date,
            'file_path': file_path,
            **{col: targets[i] for i, col in enumerate(TARGET_COLUMNS)}
        })

    result_df = pd.DataFrame(matched_data)
    print(f"Matched {len(result_df)} observations")
    return result_df


def preload_and_tokenize_all(
    data: pd.DataFrame,
    tokenizer,
    max_length: int = MAX_SEQ_LENGTH
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Preload all texts and tokenize them - uses incremental disk cache."""

    if TOKENIZED_CACHE_PATH.exists():
        print(f"Loading existing cache from {TOKENIZED_CACHE_PATH}...")
        tokenized_cache = torch.load(TOKENIZED_CACHE_PATH, weights_only=False)
        print(f"Loaded {len(tokenized_cache)} cached files")
    else:
        tokenized_cache = {}

    unique_paths = list(data['file_path'].unique())
    remaining_paths = [p for p in unique_paths if p not in tokenized_cache]

    if len(remaining_paths) == 0:
        print(f"All {len(tokenized_cache)} files already cached!")
        return tokenized_cache

    print(
        f"Tokenizing {len(remaining_paths)} remaining files (of {len(unique_paths)} total)...")

    batch_size = 32
    save_every = 50

    for batch_idx, i in enumerate(tqdm(range(0, len(remaining_paths), batch_size), desc="Tokenizing")):
        batch_paths = remaining_paths[i:i+batch_size]
        batch_texts = []
        batch_lengths = []

        for file_path in batch_paths:
            try:
                text = load_risk_factors_text(file_path)
                batch_texts.append(text[:30000])
                batch_lengths.append(min(len(text.split()), 5000))
            except Exception:
                batch_texts.append("")
                batch_lengths.append(0)

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
                'attention_mask': encodings['attention_mask'][j].clone(),
                'text_length': batch_lengths[j]
            }

        del encodings, batch_texts

        if (batch_idx + 1) % save_every == 0:
            torch.save(tokenized_cache, TOKENIZED_CACHE_PATH)
            tqdm.write(f"  Checkpoint: {len(tokenized_cache)} files saved")
            gc.collect()

    print(f"Saving final cache to {TOKENIZED_CACHE_PATH}...")
    torch.save(tokenized_cache, TOKENIZED_CACHE_PATH)

    print(f"Tokenized {len(tokenized_cache)} files total")
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        file_path = row['file_path']

        cached = self.tokenized_cache[file_path]

        targets = torch.tensor(
            [row[col] if pd.notna(row[col])
             else 0.0 for col in TARGET_COLUMNS],
            dtype=torch.float32
        )

        mask = torch.tensor(
            [1.0 if pd.notna(row[col]) else 0.0 for col in TARGET_COLUMNS],
            dtype=torch.float32
        )

        return {
            'input_ids': cached['input_ids'],
            'attention_mask': cached['attention_mask'],
            'targets': targets,
            'target_mask': mask,
            'text_length': cached['text_length']
        }


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = self.attention(hidden_states)

        mask = attention_mask.unsqueeze(-1).float()
        attention_weights = attention_weights * mask
        attention_weights = attention_weights / \
            (attention_weights.sum(dim=1, keepdim=True) + 1e-10)

        pooled = torch.sum(hidden_states * attention_weights, dim=1)

        return pooled, attention_weights.squeeze(-1)


class BondReturnPredictor(nn.Module):
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        num_targets: int = 5,
        dropout: float = DROPOUT,
        freeze_encoder: bool = True
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encoder.encoder.layer[-2:].parameters():
                param.requires_grad = True

        self.attention_pool = AttentionPooling(self.hidden_size)

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_size + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.regression_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_targets)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        pooled, attention_weights = self.attention_pool(
            hidden_states, attention_mask)

        if text_length is not None:
            text_length_norm = (text_length.float() / 5000.0).unsqueeze(-1)
        else:
            text_length_norm = torch.zeros(
                pooled.size(0), 1, device=pooled.device)

        features = torch.cat([pooled, text_length_norm], dim=-1)

        features = self.feature_extractor(features)

        predictions = torch.cat(
            [head(features) for head in self.regression_heads],
            dim=-1
        )

        return predictions, attention_weights


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (predictions - targets) ** 2
    masked_diff = diff * mask

    if mask.sum() > 0:
        return masked_diff.sum() / mask.sum()
    return torch.tensor(0.0, device=predictions.device)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        target_mask = batch['target_mask'].to(device, non_blocking=True)
        text_length = batch['text_length'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            predictions, _ = model(input_ids, attention_mask, text_length)
            loss = masked_mse_loss(predictions, targets, target_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(
                device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            target_mask = batch['target_mask'].to(device, non_blocking=True)
            text_length = batch['text_length'].to(device, non_blocking=True)

            with autocast('cuda'):
                predictions, _ = model(input_ids, attention_mask, text_length)
                loss = masked_mse_loss(predictions, targets, target_mask)

            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(target_mask.cpu())

    avg_loss = total_loss / max(num_batches, 1)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    metrics = {}
    for i, col in enumerate(TARGET_COLUMNS):
        mask = all_masks[:, i] > 0
        if mask.sum() > 0:
            pred = all_predictions[mask, i].numpy()
            targ = all_targets[mask, i].numpy()

            mse = np.mean((pred - targ) ** 2)
            metrics[f'{col}_mse'] = mse

            correct_direction = np.mean((pred > 0) == (targ > 0))
            metrics[f'{col}_direction_acc'] = correct_direction

    return avg_loss, metrics


def extract_important_words(
    model: nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    top_k: int = 20
) -> List[Tuple[str, float]]:
    model.eval()

    encoding = tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        _, attention_weights = model(input_ids, attention_mask)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    weights = attention_weights[0].cpu().numpy()

    mask = attention_mask[0].cpu().numpy()
    valid_indices = np.where(mask > 0)[0]

    token_weights = []
    for idx in valid_indices:
        token = tokens[idx]
        if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            token_weights.append((token, weights[idx]))

    token_weights.sort(key=lambda x: x[1], reverse=True)

    return token_weights[:top_k]


def plot_training_curves(train_losses: List[float], val_losses: List[float], save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    print("=" * 60)
    print("Bond Return Prediction - OPTIMIZED VERSION")
    print("=" * 60)
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

    print("\n[1/6] Loading CSV data...")
    csv_data = load_csv_data()

    print("\n[2/6] Scanning SEC filings...")
    filings_map = scan_sec_filings(use_cache=True)

    print("\n[3/6] Preparing dataset...")
    matched_data = prepare_dataset(csv_data, filings_map)

    if len(matched_data) == 0:
        print("ERROR: No matched data found. Check data paths and formats.")
        return

    print("\n[4/6] Initializing model and tokenizer...")
    clear_gpu_memory()

    try:
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = BondReturnPredictor(model_name="yiyanghkust/finbert-tone")
    except Exception as e:
        print(f"FinBERT not available ({e}), falling back to RoBERTa...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = BondReturnPredictor(model_name="roberta-base")

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n[5/6] Pre-loading and tokenizing all texts...")
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

    print(f"\n[6/6] Starting training...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Batches per epoch: {len(train_loader)}")

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

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, DEVICE)
        train_losses.append(train_loss)

        val_loss, metrics = validate(model, val_loader, DEVICE)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")

        for col in TARGET_COLUMNS[:3]:
            if f'{col}_direction_acc' in metrics:
                print(
                    f"  {col} Direction Acc: {metrics[f'{col}_direction_acc']:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(
                f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    plot_training_curves(
        train_losses,
        val_losses,
        str(BASE_DIR / "v1" / "training_curves.png")
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    print("\n[Bonus] Extracting important words from sample...")
    checkpoint = torch.load(
        MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    sample_path = matched_data.iloc[0]['file_path']
    sample_text = load_risk_factors_text(sample_path)
    important_words = extract_important_words(
        model, tokenizer, sample_text, DEVICE)

    print("\nTop 20 Most Important Words (by attention weight):")
    for i, (word, weight) in enumerate(important_words):
        print(f"  {i+1:2d}. {word:20s} (weight: {weight:.4f})")


if __name__ == "__main__":
    main()
