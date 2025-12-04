# train_model.py
# PyTorch training script for U-Net on STFT-magnitude masks.
#
# loss_type:
#   - "mse"       : amplitude MSE only
#   - "mse_perc"  : amplitude MSE + perceptual STFT-L1 (50% + 50%)

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_unet import UNet


# =========================
# Dataset
# =========================

class PairWaveDataset(Dataset):
    """
    清音（clean）と雑音付き音声（noisy）のペアを返す Dataset です。
    前提：
        data/processed/{split}/clean/*.wav
        data/processed/{split}/noisy/*.wav
    で、clean/noisy は同じファイル名で揃えてあるとします。
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        sample_rate: int = 16000,
        manifest_path: Path | None = None,
        limit: int | None = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate

        self.proc_root = self.root_dir / "data" / "processed"
        if manifest_path is None:
            manifest_path = self.proc_root / "manifest.csv"

        if not manifest_path.is_file():
            raise RuntimeError(f"manifest not found: {manifest_path}")

        # manifest から (noisy, clean) のパスを読む
        self.pairs = []
        import csv
        with manifest_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                if not row["clean_path"]:
                    # test など clean が無い行はスキップ
                    continue
                noisy_path = Path(row["noisy_path"])
                clean_path = Path(row["clean_path"])
                if noisy_path.is_file() and clean_path.is_file():
                    self.pairs.append((noisy_path, clean_path))
                else:
                    print(f"[WARN] missing file for row: {row}")

                if limit is not None and len(self.pairs) >= limit:
                    break

        if len(self.pairs) == 0:
            raise RuntimeError(f"No wav pairs found for split={split} in manifest.")

        print(f"[INFO] {split}: {len(self.pairs)} pairs found.")

    def __len__(self):
        return len(self.pairs)

    def _load_wav(self, path: Path) -> np.ndarray:
        wav, sr = sf.read(path)
        # shape (T,) or (T, C)
        if wav.ndim == 2:
            wav = wav[:, 0]  # take first channel
        wav = wav.astype(np.float32)

        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        return wav

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]

        noisy = self._load_wav(noisy_path)
        clean = self._load_wav(clean_path)

        # 長さが違えば短い方に合わせて切る
        L = min(len(noisy), len(clean))
        noisy = noisy[:L]
        clean = clean[:L]

        noisy = torch.from_numpy(noisy)  # (T,)
        clean = torch.from_numpy(clean)  # (T,)
        return noisy, clean


# =========================
# Losses
# =========================

def amplitude_mse_loss(
    noisy_wave: torch.Tensor,
    clean_wave: torch.Tensor,
    mask_pred: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    振幅スペクトル MSE 損失を計算します。
    ついでに ISTFT による復元波形も返します。
    """
    window = torch.hann_window(n_fft, device=device)

    # STFT: complex tensor (B, F, T)
    X = torch.stft(
        noisy_wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    S = torch.stft(
        clean_wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )

    mag_X = X.abs()          # (B, F, T)
    mag_S = S.abs()

    # model input: (B, 1, F, T)
    mag_X_in = mag_X.unsqueeze(1)

    # mask_pred も (B, 1, F, T) を想定
    M = mask_pred

    # 推定振幅
    mag_hat = M.squeeze(1) * mag_X  # (B, F, T)

    # MSE 損失
    mse = torch.mean((mag_hat - mag_S) ** 2)

    # 位相は観測 X の位相を使用
    phase_X = torch.angle(X)
    complex_hat = mag_hat * torch.exp(1j * phase_X)

    # ISTFT で復元波形
    # noisy_wave.shape: (B, T)
    T = noisy_wave.shape[1]
    y_hat = torch.istft(
        complex_hat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        length=T,
    )

    return mse, y_hat, mag_X_in, mag_S


def perceptual_stft_l1(
    y_hat: torch.Tensor,
    y_ref: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    device: str = "cuda",
) -> torch.Tensor:
    """
    知覚スペクトル損失（周波数重み付き STFT-L1）を簡易実装します。
    """
    window = torch.hann_window(n_fft, device=device)

    Y_hat = torch.stft(
        y_hat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    Y_ref = torch.stft(
        y_ref,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )

    mag_hat = Y_hat.abs()
    mag_ref = Y_ref.abs()

    # 周波数重み（低域 < 高域）を単調増加で付与
    F = mag_hat.shape[-2]
    w = torch.linspace(1.0, 2.0, F, device=device).view(1, F, 1)  # (1, F, 1)

    diff = (mag_hat - mag_ref).abs()
    loss = (w * diff).mean()
    return loss


# =========================
# Training Loop
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str = "mse",
):
    model.train()
    total_loss = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)  # (B, T)
        clean = clean.to(device)

        optimizer.zero_grad()

        # STFT → magnitude → model
        # amplitude MSE（主損失）＋必要なら知覚損失
        mse_loss, y_hat, mag_X_in, mag_S = amplitude_mse_loss(
            noisy_wave=noisy,
            clean_wave=clean,
            mask_pred=model(
                mag_X_in := torch.stft(
                    noisy,
                    n_fft=1024,
                    hop_length=256,
                    win_length=1024,
                    window=torch.hann_window(1024, device=device),
                    return_complex=True,
                )
                .abs()
                .unsqueeze(1)
            ),
            n_fft=1024,
            hop_length=256,
            device=device,
        )

        if loss_type == "mse":
            loss = mse_loss
        elif loss_type == "mse_perc":
            # 0.5 * MSE + 0.5 * 知覚損失
            perc_loss = perceptual_stft_l1(
                y_hat=y_hat,
                y_ref=clean,
                n_fft=512,
                hop_length=160,
                device=device,
            )
            loss = 0.5 * mse_loss + 0.5 * perc_loss
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * noisy.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str = "mse",
):
    model.eval()
    total_loss = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        mse_loss, y_hat, mag_X_in, mag_S = amplitude_mse_loss(
            noisy_wave=noisy,
            clean_wave=clean,
            mask_pred=model(
                mag_X_in := torch.stft(
                    noisy,
                    n_fft=1024,
                    hop_length=256,
                    win_length=1024,
                    window=torch.hann_window(1024, device=device),
                    return_complex=True,
                )
                .abs()
                .unsqueeze(1)
            ),
            n_fft=1024,
            hop_length=256,
            device=device,
        )

        if loss_type == "mse":
            loss = mse_loss
        else:
            perc_loss = perceptual_stft_l1(
                y_hat=y_hat,
                y_ref=clean,
                n_fft=512,
                hop_length=160,
                device=device,
            )
            loss = 0.5 * mse_loss + 0.5 * perc_loss

        total_loss += loss.item() * noisy.size(0)

    return total_loss / len(loader.dataset)


# =========================
# Main
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".", help="project root dir")
    p.add_argument("--loss_type", type=str, default="mse",
                  choices=["mse", "mse_perc"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--train_limit", type=int, default=-1,
                  help="use first N train pairs (<=0 for all)")
    p.add_argument("--valid_limit", type=int, default=-1,
                  help="use first N valid pairs (<=0 for all)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] root={root}")
    print(f"[INFO] device={device}")
    print(f"[INFO] loss_type={args.loss_type}")

    # Dataset
    train_limit = args.train_limit if args.train_limit > 0 else None
    valid_limit = args.valid_limit if args.valid_limit > 0 else None

    train_ds = PairWaveDataset(root, split="train", sample_rate=args.sr, limit=train_limit)
    valid_ds = PairWaveDataset(root, split="valid", sample_rate=args.sr, limit=valid_limit)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Model
    model = UNet(in_channels=1, out_channels=1, base_ch=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    weights_dir = root / "weights"
    weights_dir.mkdir(exist_ok=True, parents=True)

    if args.loss_type == "mse":
        out_path = weights_dir / "unet_mse.pt"
    else:
        out_path = weights_dir / "unet_mse_perc.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_type=args.loss_type
        )
        val_loss = eval_one_epoch(
            model, valid_loader, device, loss_type=args.loss_type
        )

        print(f"  train_loss = {train_loss:.6f}")
        print(f"  valid_loss = {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": vars(args),
                },
                out_path,
            )
            print(f"  [BEST] saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
