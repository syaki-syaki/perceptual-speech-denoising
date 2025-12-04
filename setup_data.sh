#!/usr/bin/env bash
set -e

########################################
# 0. データセットのルートパス（要変更）
########################################

# VoiceBank-DEMAND（Valentini）を展開したルート
VOICEBANK_ROOT="/path/to/voicebank_demand"

# 例:
#   $VOICEBANK_ROOT/clean_trainset_28spk
#   $VOICEBANK_ROOT/clean_testset

# DEMAND ノイズのルート
DEMAND_ROOT="/path/to/DEMAND"
# 例:
#   $DEMAND_ROOT/living_room/*.wav
#   $DEMAND_ROOT/office/*.wav
#   ... といったサブフォルダにノイズ wav が入っている想定

########################################
# 1. ディレクトリ作成
########################################

mkdir -p data/raw/train/clean
mkdir -p data/raw/train/noise
mkdir -p data/raw/valid/clean
mkdir -p data/raw/valid/noise
mkdir -p data/raw/test/noisy

########################################
# 2. VoiceBank（クリーン音声）の振り分け
########################################

# 学習用クリーン音声：clean_trainset_28spk → train/clean
if [ -d "$VOICEBANK_ROOT/clean_trainset_28spk" ]; then
  echo "Copying VoiceBank train clean -> data/raw/train/clean"
  find "$VOICEBANK_ROOT/clean_trainset_28spk" -type f -name '*.wav' \
    -exec ln -s "{}" data/raw/train/clean/ \;
else
  echo "WARN: $VOICEBANK_ROOT/clean_trainset_28spk が見つかりませんでした" >&2
fi

# 検証用クリーン音声：clean_testset → valid/clean
if [ -d "$VOICEBANK_ROOT/clean_testset" ]; then
  echo "Copying VoiceBank valid clean -> data/raw/valid/clean"
  find "$VOICEBANK_ROOT/clean_testset" -type f -name '*.wav' \
    -exec ln -s "{}" data/raw/valid/clean/ \;
else
  echo "WARN: $VOICEBANK_ROOT/clean_testset が見つかりませんでした" >&2
fi

########################################
# 3. DEMAND（環境ノイズ）の振り分け
########################################

# DEMAND から全ノイズ wav を集めて 8:2 で train/valid に分割
if [ -d "$DEMAND_ROOT" ]; then
  echo "Collecting DEMAND noises from $DEMAND_ROOT"
  tmp_list="$(mktemp)"
  find "$DEMAND_ROOT" -type f -name '*.wav' | sort > "$tmp_list"

  total=$(wc -l < "$tmp_list")
  if [ "$total" -eq 0 ]; then
    echo "WARN: DEMAND_ROOT から .wav が見つかりませんでした" >&2
  else
    train_n=$(( total * 8 / 10 ))
    valid_n=$(( total - train_n ))

    echo "Total DEMAND noise files: $total (train: $train_n, valid: $valid_n)"

    # train/noise に 8 割
    head -n "$train_n" "$tmp_list" \
      | xargs -I{} ln -s "{}" data/raw/train/noise/

    # valid/noise に残り 2 割
    tail -n "$valid_n" "$tmp_list" \
      | xargs -I{} ln -s "{}" data/raw/valid/noise/
  fi

  rm -f "$tmp_list"
else
  echo "WARN: DEMAND_ROOT=$DEMAND_ROOT が存在しません" >&2
fi

########################################
# 4. YouTube クリーン音源の配置
#    （yt-dlp が必要：pip install yt-dlp 等）
########################################

# 検証用クリーン → valid/clean
yt-dlp -x --audio-format wav \
  -o "data/raw/valid/clean/youtube_clean_valid01.wav" \
  "https://youtu.be/xJaIrB-VkyA?si=xa4u4Wo-oZwE_57E"

# 学習用クリーン 4 本 → train/clean
yt-dlp -x --audio-format wav \
  -o "data/raw/train/clean/youtube_clean_train01.wav" \
  "https://youtu.be/J1Vc89-mYZM?si=oWdLV_HfD1JyuHE1"

yt-dlp -x --audio-format wav \
  -o "data/raw/train/clean/youtube_clean_train02.wav" \
  "https://youtu.be/U978hwPnL4I?si=TJ8TN5HXNKqPNCe5"

yt-dlp -x --audio-format wav \
  -o "data/raw/train/clean/youtube_clean_train03.wav" \
  "https://youtu.be/-inaWMdRv-o?si=8pxoBveisdvG41l7"

yt-dlp -x --audio-format wav \
  -o "data/raw/train/clean/youtube_clean_train04.wav" \
  "https://youtu.be/6NjS6gaGU48?si=-mUxvV3xVqr-5vly"

########################################
# 5. YouTube ノイジー音源の配置
########################################

# 学習用ノイジー
yt-dlp -x --audio-format wav \
  -o "data/raw/train/noise/youtube_noise_train01.wav" \
  "https://youtu.be/Vyx3AcVp6N8?si=ZH4BVrAPNcpYTXrv"

yt-dlp -x --audio-format wav \
  -o "data/raw/train/noise/youtube_noise_train02.wav" \
  "https://youtu.be/E6O6IvOehQc?si=XiAbqPdd1p5gfbq7"

yt-dlp -x --audio-format wav \
  -o "data/raw/train/noise/youtube_noise_train03.wav" \
  "https://youtu.be/qyL7PsNj9RA?si=V2AWv-O5f9ic_Kkz"

yt-dlp -x --audio-format wav \
  -o "data/raw/train/noise/youtube_noise_train04.wav" \
  "https://youtu.be/P31OhJ9LF4Y?si=SpXM5seKrkL0Gq09"

# 検証用ノイジー → valid/noise
yt-dlp -x --audio-format wav \
  -o "data/raw/valid/noise/youtube_noise_valid01.wav" \
  "https://youtu.be/4PaFCQZOBKU?si=yB-W0qEPWUc4I_eY"

# 推論用ノイジー → test/noisy
yt-dlp -x --audio-format wav \
  -o "data/raw/test/noisy/youtube_noisy_test01.wav" \
  "https://youtu.be/INWL0PeOez0?si=bS2PRpvY_wvo8YEN"

yt-dlp -x --audio-format wav \
  -o "data/raw/test/noisy/youtube_noisy_test02.wav" \
  "https://youtu.be/gyVcajRVhlQ?si=DVOAOCSsUwNpprXg"

echo "データディレクトリ構成とダウンロード処理が完了しました。"
