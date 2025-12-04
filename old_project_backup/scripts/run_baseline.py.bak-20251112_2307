# -*- coding: utf-8 -*-
# C:\卒論\scripts\run_baseline.py

import os
import glob
import csv
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, wiener

# --------- ユーティリティ ---------

TARGET_SR = 16000

def load_wav_mono16k(path: str) -> np.ndarray:
    """WAV を読み込み、mono/16k に正規化して [-1,1] に収めます。"""
    x, sr = sf.read(path, always_2d=False)
    # shape を 1次元へ
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    # サンプリング周波数を 16k に
    if sr != TARGET_SR:
        # なるべく劣化の少ない整数比リサンプル
        x = resample_poly(x, TARGET_SR, sr)
    # NaN/Inf 安全化
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # クリップ
    x = np.clip(x, -1.0, 1.0)
    return x.astype(np.float32)

def save_wav(path: str, x: np.ndarray, sr: int = TARGET_SR) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, sr, subtype="PCM_16")

def denoise_baseline(x: np.ndarray) -> np.ndarray:
    """
    ごく簡単なベースライン:
      1) ウィーナーフィルタ（時系列 1D）
      2) 軽い移動平均で平滑化
    GPU不要。高速。過度に滑らかになり過ぎる場合あり。
    """
    # 窓サイズは秒換算で ~10–20ms 程度を目安（16kHzなら 161 点前後）
    x1 = wiener(x, mysize=161)  # scipy.signal.wiener
    # 3点移動平均
    kernel = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
    x2 = np.convolve(x1, kernel, mode="same")
    # クリップ
    return np.clip(x2, -1.0, 1.0).astype(np.float32)

def seg_snr_like(noisy: np.ndarray, clean_est: np.ndarray) -> float:
    """
    簡易SNR改善指標: noisy と clean_est の残差を使った全体SNR。
    10*log10( sum(clean_est^2) / sum((noisy-clean_est)^2) )
    """
    n = min(len(noisy), len(clean_est))
    noisy = noisy[:n]
    clean_est = clean_est[:n]
    resid = noisy - clean_est
    num = np.sum(clean_est ** 2, dtype=np.float64) + 1e-12
    den = np.sum(resid ** 2, dtype=np.float64) + 1e-12
    return 10.0 * math.log10(num / den)

# --------- メイン処理 ---------

def main():
    # 入力ファイル探索（事前に 16k/mono へ整形済み想定）
    files = sorted(glob.glob(os.path.join("data", "processed", "*_16k.wav")))
    if not files:
        print("入力が見つかりません: data/processed/*_16k.wav を用意してください。")
        return

    print(f"評価対象ファイル数: {len(files)}")
    scores = []  # (ファイル名, SNR値) を格納

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"処理中: {fname}")
        noisy = load_wav_mono16k(fpath)
        clean_est = denoise_baseline(noisy)
        snr = seg_snr_like(noisy, clean_est)
        print(f"  改善度(SNR): {snr:.2f} dB")
        scores.append((fname, snr))

        # 生成音の保存（任意）
        out_wav = os.path.join("results", "denoised", fname.replace("_16k.wav", "_den_16k.wav"))
        save_wav(out_wav, clean_est, TARGET_SR)

    # 集計
    print("\n=== 集計結果 ===")
    for k, v in scores:
        print(f"{k:25s} {v:.2f} dB")
    avg = sum(v for _, v in scores) / max(1, len(scores))
    print(f"\n平均改善度: {avg:.2f} dB")

    # CSV 出力
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    csv_path = os.path.join("results", "tables", "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "snr_db"])
        for k, v in scores:
            w.writerow([k, f"{v:.2f}"])
        w.writerow(["AVERAGE", f"{avg:.2f}"])
    print(f"保存: {csv_path}")

if __name__ == "__main__":
    main()
