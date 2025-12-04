import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import librosa
import numpy as np
import torch
from srmrpy import srmr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model_unet import UNet  # noqa: E402


def split_segments(wav: np.ndarray, sr: int, seg_sec: float) -> List[np.ndarray]:
    seg_len = int(sr * seg_sec)
    n = len(wav) // seg_len
    return [wav[i * seg_len:(i + 1) * seg_len] for i in range(n) if seg_len > 0]


def load_wav(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav.astype(np.float32), target_sr


def denoise_segment(model: UNet, seg: np.ndarray, device: torch.device) -> np.ndarray:
    """U-Net 推論でマスク推定し、ISTFT で波形を復元する。"""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(seg).to(device).unsqueeze(0)  # (1, T)
        window = torch.hann_window(1024, device=device)
        X = torch.stft(
            x,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=window,
            return_complex=True,
        )
        mag = X.abs()
        mask = model(mag.unsqueeze(1))  # (1, 1, F, T)
        mag_hat = mask.squeeze(1) * mag
        phase = torch.angle(X)
        complex_hat = mag_hat * torch.exp(1j * phase)
        y = torch.istft(
            complex_hat,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=window,
            length=x.shape[1],
        )
    return y.squeeze(0).cpu().numpy()


def eval_condition(
    condition: str,
    segments: Iterable[np.ndarray],
    sr: int,
    model: UNet | None,
    device: torch.device,
) -> List[dict]:
    rows = []
    for seg_idx, seg in enumerate(segments):
        if model is None:
            out = seg
        else:
            out = denoise_segment(model, seg, device)

        score, _ = srmr(out, sr)
        rows.append(
            {
                "condition": condition,
                "segment": seg_idx,
                "srmr": float(score),
            }
        )
    return rows


def load_model(weights_path: Path, device: torch.device) -> UNet:
    state = torch.load(weights_path, map_location=device)
    model = UNet(in_channels=1, out_channels=1, base_ch=64).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_dir", type=str, default="data/processed/test/noisy")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seg_sec", type=float, default=4.0)
    parser.add_argument("--mse_weights", type=str, default="weights/unet_mse.pt")
    parser.add_argument("--perc_weights", type=str, default="weights/unet_mse_perc.pt")
    parser.add_argument("--csv_out", type=str, default="results/srmr_comparison.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    noisy_dir = Path(args.noisy_dir)
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # モデル読込
    mse_model = load_model(Path(args.mse_weights), device)
    perc_model = load_model(Path(args.perc_weights), device)

    rows = []

    for wav_path in sorted(noisy_dir.glob("*.wav")):
        wav, sr = load_wav(wav_path, args.sr)
        segs = split_segments(wav, sr, args.seg_sec)
        if not segs:
            print(f"[WARN] skip (too short): {wav_path.name}")
            continue

        # Noisy
        rows_file = eval_condition("Noisy", segs, sr, None, device)
        for r in rows_file:
            r["file"] = wav_path.name
        rows.extend(rows_file)

        # MSE
        rows_file = eval_condition("MSE", segs, sr, mse_model, device)
        for r in rows_file:
            r["file"] = wav_path.name
        rows.extend(rows_file)

        # Perceptual
        rows_file = eval_condition("Perceptual", segs, sr, perc_model, device)
        for r in rows_file:
            r["file"] = wav_path.name
        rows.extend(rows_file)

    import statistics

    means = {}
    for cond in ["Noisy", "MSE", "Perceptual"]:
        vals = [r["srmr"] for r in rows if r["condition"] == cond]
        if vals:
            means[cond] = statistics.mean(vals)

    # まとめて CSV に保存（末尾に集計行を付ける）
    summary_rows = []
    for cond, v in means.items():
        summary_rows.append(
            {
                "file": "__overall__",
                "segment": "__mean__",
                "condition": cond,
                "srmr": v,
            }
        )

    noisy_mean = means.get("Noisy")
    if noisy_mean is not None:
        for cond in ["MSE", "Perceptual"]:
            if cond in means:
                summary_rows.append(
                    {
                        "file": "__overall__",
                        "segment": "__improvement__",
                        "condition": f"{cond}-Noisy",
                        "srmr": means[cond] - noisy_mean,
                    }
                )

    fieldnames = ["file", "segment", "condition", "srmr"]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        if summary_rows:
            writer.writerows(summary_rows)

    # コンディション別平均も最後に表示
    print("[DONE] CSV saved to", csv_out)
    for cond, v in means.items():
        print(f"  {cond} mean SRMR: {v:.3f}")


if __name__ == "__main__":
    main()
