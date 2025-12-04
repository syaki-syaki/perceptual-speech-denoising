# scripts/prepare_data_youtube.py

import os
import glob
import random
import csv
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# ===== 設定値 =====
SR = 16000          # サンプリング周波数
SEG_SEC = 4.0       # セグメント長 [秒]
SNR_MIN = -5.0      # SNR 下限 [dB]
SNR_MAX = 15.0      # SNR 上限 [dB]

RAW_ROOT = Path("data/raw")
PROC_ROOT = Path("data/processed")
MANIFEST_PATH = PROC_ROOT / "manifest.csv"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def mix_with_snr(clean, noise, snr_db):
    """クリーンとノイズを指定 SNR[dB] で混合する"""
    eps = 1e-8
    Pc = np.mean(clean**2) + eps
    Pn = np.mean(noise**2) + eps

    # SNR = 10 log10(Pc / (alpha^2 Pn)) から alpha を求める
    alpha2 = Pc / (Pn * 10 ** (snr_db / 10.0))
    alpha = np.sqrt(alpha2)
    noise_scaled = noise * alpha
    mixed = clean + noise_scaled

    # クリッピングしないように簡易正規化
    max_amp = max(
        np.max(np.abs(mixed)),
        np.max(np.abs(clean)),
        np.max(np.abs(noise_scaled)),
        1.0,
    )
    mixed /= max_amp * 1.01
    clean /= max_amp * 1.01

    return mixed.astype(np.float32), clean.astype(np.float32), snr_db


def process_split(split_name: str, manifest_writer):
    """
    train / valid 用：
      - RAW_ROOT/{split}/clean/*.wav
      - RAW_ROOT/{split}/noise/*.wav
    から 4 秒セグメントの (noisy, clean) を作って
      - PROC_ROOT/{split}/noisy/*.wav
      - PROC_ROOT/{split}/clean/*.wav
    に保存し、manifest.csv に書き込む
    """
    clean_dir = RAW_ROOT / split_name / "clean"
    noise_dir = RAW_ROOT / split_name / "noise"

    clean_files = sorted(glob.glob(str(clean_dir / "*.wav")))
    noise_files = sorted(glob.glob(str(noise_dir / "*.wav")))

    if len(clean_files) == 0 or len(noise_files) == 0:
        print(f"[WARN] {split_name}: clean か noise の wav が見つかりません。")
        return

    out_clean_dir = PROC_ROOT / split_name / "clean"
    out_noisy_dir = PROC_ROOT / split_name / "noisy"
    ensure_dir(out_clean_dir)
    ensure_dir(out_noisy_dir)

    seg_len = int(SR * SEG_SEC)
    idx = 0

    print(f"[INFO] {split_name}: clean={len(clean_files)} file(s), noise={len(noise_files)} file(s)")

    for c_path in clean_files:
        # クリーンをロード
        clean_wav, _ = librosa.load(c_path, sr=SR, mono=True)

        # 適当にノイズファイルを選ぶ（1:1 ではなくてもよい）
        n_path = random.choice(noise_files)
        noise_wav, _ = librosa.load(n_path, sr=SR, mono=True)

        n_segs = min(len(clean_wav), len(noise_wav)) // seg_len
        if n_segs == 0:
            continue

        for seg_id in range(n_segs):
            s = seg_id * seg_len
            e = s + seg_len
            c_seg = clean_wav[s:e]
            n_seg = noise_wav[s:e]

            snr_db = random.uniform(SNR_MIN, SNR_MAX)
            noisy_seg, clean_seg, snr_used = mix_with_snr(c_seg, n_seg, snr_db)

            noisy_name = f"{split_name}_noisy_{idx:05d}.wav"
            clean_name = f"{split_name}_clean_{idx:05d}.wav"

            noisy_out_path = out_noisy_dir / noisy_name
            clean_out_path = out_clean_dir / clean_name

            sf.write(noisy_out_path, noisy_seg, SR)
            sf.write(clean_out_path, clean_seg, SR)

            manifest_writer.writerow({
                "split": split_name,
                "noisy_path": str(noisy_out_path),
                "clean_path": str(clean_out_path),
                "snr_db": f"{snr_used:.2f}",
            })

            idx += 1

    print(f"[INFO] {split_name}: 生成セグメント数 = {idx}")


def process_test_noisy(manifest_writer):
    """
    test/noisy の wav を 16 kHz にそろえて
      - PROC_ROOT/test/noisy
    にコピーし、manifest には clean_path を空で記録
    """
    in_dir = RAW_ROOT / "test" / "noisy"
    files = sorted(glob.glob(str(in_dir / "*.wav")))
    if len(files) == 0:
        print("[WARN] test/noisy の wav が見つかりません。")
        return

    out_dir = PROC_ROOT / "test" / "noisy"
    ensure_dir(out_dir)

    print(f"[INFO] test/noisy: {len(files)} file(s)")

    for fpath in files:
        y, _ = librosa.load(fpath, sr=SR, mono=True)
        name = Path(fpath).name
        out_path = out_dir / name
        sf.write(out_path, y, SR)

        manifest_writer.writerow({
            "split": "test",
            "noisy_path": str(out_path),
            "clean_path": "",
            "snr_db": "",
        })


def main():
    ensure_dir(PROC_ROOT)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "noisy_path", "clean_path", "snr_db"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # train / valid 用の (noisy, clean) セグメント生成
        process_split("train", writer)
        process_split("valid", writer)

        # test/noisy の 16 kHz 化
        process_test_noisy(writer)

    print(f"[DONE] data/processed 以下にデータと {MANIFEST_PATH} を作成しました。")


if __name__ == "__main__":
    main()
