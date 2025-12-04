
import os
import glob
import argparse

import numpy as np
import librosa

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from model_unet import UNet  # vbelz/Speech-enhancement の PyTorch UNet


# =========================
# 基本ハイパーパラメータ
# =========================

SAMPLE_RATE = 16000          # 16 kHz に統一
N_FFT       = 1024           # STFT FFT 長
HOP_LENGTH  = 256            # STFT ホップ長
WIN_LENGTH  = 1024           # 窓長
WINDOW      = "hann"         # Hann 窓

SEED = 3407
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# データ読み込み関連
# =========================

def _stft_mag(y,
              n_fft=N_FFT,
              hop_length=HOP_LENGTH,
              win_length=WIN_LENGTH,
              window=WINDOW):
    """波形 y から STFT 振幅スペクトログラムを計算して返します。"""
    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    mag = np.abs(S)  # (F, T)
    return mag


def _load_wav_mono(path, sr=SAMPLE_RATE):
    """モノラル + sr にリサンプリングして読み込み。"""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def _list_pairs(split):
    """
    data/processed/{split}/clean と noisy のペア一覧を作ります。

    前提:
        data/processed/{split}/clean/*.wav
        data/processed/{split}/noisy/*.wav
    """
    clean_dir = os.path.join("data", "processed", split, "clean")
    noisy_dir = os.path.join("data", "processed", split, "noisy")

    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))

    if len(clean_files) == 0 or len(noisy_files) == 0:
        raise RuntimeError(
            f"{split}: clean/noisy の wav が見つかりません。"
            f" clean={clean_dir}, noisy={noisy_dir}"
        )

    if len(clean_files) != len(noisy_files):
        print(
            f"[WARN] {split}: mismatch {len(clean_files)} vs {len(noisy_files)}"
        )

    n = min(len(clean_files), len(noisy_files))
    pairs = list(zip(clean_files[:n], noisy_files[:n]))
    return pairs


def _make_dataset(split):
    """
    指定 split(train/valid) のすべての (clean, noisy) ペアから
    STFT 振幅スペクトログラムを作り，
      X: noisy, Y: clean
    を torch.Tensor (N, 1, F, T) で返します。
    """
    pairs = _list_pairs(split)
    print(f"[INFO] {split}: {len(pairs)} pairs")

    mags_noisy = []
    mags_clean = []

    # 1パス目: STFT をすべて計算（numpy）
    for clean_path, noisy_path in pairs:
        y_clean = _load_wav_mono(clean_path)
        y_noisy = _load_wav_mono(noisy_path)

        mag_clean = _stft_mag(y_clean)   # (F, T_c)
        mag_noisy = _stft_mag(y_noisy)   # (F, T_n)

        # 時間フレーム数を短い方に合わせる
        T = min(mag_clean.shape[1], mag_noisy.shape[1])
        mag_clean = mag_clean[:, :T]
        mag_noisy = mag_noisy[:, :T]

        mags_clean.append(mag_clean)
        mags_noisy.append(mag_noisy)

    # 全サンプルの時間長を最小値にそろえる
    min_T = min(m.shape[1] for m in mags_noisy)
    F_bins = mags_noisy[0].shape[0]
    print(f"[INFO] {split}: X/Y shape -> F x T = {F_bins} x {min_T}")

    X_list = []
    Y_list = []
    for mag_clean, mag_noisy in zip(mags_clean, mags_noisy):
        X_list.append(mag_noisy[:, :min_T])  # noisy
        Y_list.append(mag_clean[:, :min_T])  # clean

    X = np.stack(X_list, axis=0)  # (N, F, T)
    Y = np.stack(Y_list, axis=0)  # (N, F, T)

    # PyTorch の Conv2d 形式 (N, C=1, F, T) に変形
    X = X[:, np.newaxis, :, :]  # (N, 1, F, T)
    Y = Y[:, np.newaxis, :, :]  # (N, 1, F, T)

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    print(f"[INFO] {split}: X{tuple(X_t.shape)}, Y{tuple(Y_t.shape)}")
    return X_t, Y_t


# =========================
# 損失関数
# =========================

def loss_mse(y_pred, y_true):
    """振幅スペクトルの MSE."""
    return F.mse_loss(y_pred, y_true)


def loss_perceptual_stft_l1(y_pred, y_true, eps=1e-6):
    """
    知覚 STFT-L1 もどき:
        log 振幅スペクトル同士の L1 距離。
    """
    y_true_clamped = torch.clamp(y_true, min=eps)
    y_pred_clamped = torch.clamp(y_pred, min=eps)

    true_log = torch.log(y_true_clamped)
    pred_log = torch.log(y_pred_clamped)

    return torch.mean(torch.abs(pred_log - true_log))


def loss_mse_plus_perc(y_pred, y_true):
    """
    ケースB:
        L = 0.5 * MSE + 0.5 * perceptual_L1
    """
    mse = loss_mse(y_pred, y_true)
    perc = loss_perceptual_stft_l1(y_pred, y_true)
    return 0.5 * mse + 0.5 * perc


# =========================
# 学習本体
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["mse", "mse_perc"],
        default="mse",
        help="mse: 振幅MSEのみ, mse_perc: MSE+知覚L1(0.5:0.5)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    print("[INFO] Loading dataset")
    X_train, Y_train = _make_dataset("train")
    X_valid, Y_valid = _make_dataset("valid")

    train_ds = TensorDataset(X_train, Y_train)
    valid_ds = TensorDataset(X_valid, Y_valid)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    _, C, F_bins, T_bins = X_train.shape
    input_shape = (C, F_bins, T_bins)
    print(f"[INFO] input_shape (C,F,T) = {input_shape}")

    # UNet (PyTorch) を構築
    print("[INFO] Building UNet (PyTorch)")
    # model_unet.UNet は in_channels/out_channels 引数名
    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)

    # 損失関数の切り替え
    if args.loss_type == "mse":
        loss_fn = loss_mse
        tag = "mse"
    else:
        loss_fn = loss_mse_plus_perc
        tag = "mse_perc"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("weights", exist_ok=True)
    best_ckpt = os.path.join("weights", f"model_unet_{tag}_best.pt")
    last_ckpt = os.path.join("weights", f"model_unet_{tag}_last.pt")

    best_val_loss = float("inf")

    print(f"[INFO] Start training (loss_type={args.loss_type})")

    for epoch in range(1, args.epochs + 1):
        # -------- train --------
        model.train()
        running_loss = 0.0
        n_batches = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # UNet 出力（ここでは「クリーン振幅推定」とみなす）
            pred = model(noisy)

            loss = loss_fn(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # -------- valid --------
        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for noisy, clean in valid_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)

                pred = model(noisy)
                vloss = loss_fn(pred, clean)

                val_running += vloss.item()
                val_batches += 1

        val_loss = val_running / max(val_batches, 1)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

        # ベストモデルを保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"[INFO]  best updated -> {best_ckpt}")

    # 最終エポックの重み
    torch.save(model.state_dict(), last_ckpt)
    print(f"[INFO] Finished. best={best_ckpt}, last={last_ckpt}")


if __name__ == "__main__":
    main()
