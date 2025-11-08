# ===== upload_all.ps1 =====
# 事前: gcloud / gsutil / ffmpeg / python(yt_dlp) が使えること
# 使い方:
#   cd C:\卒論\data
#   Set-ExecutionPolicy -Scope Process RemoteSigned
#   gcloud auth login
#   gcloud config set project studious-lore-473812-m0
#   gsutil mb -l asia-northeast1 gs://cheer-studious-lore-473812-m0 2>$null
#   .\upload_all.ps1

$ErrorActionPreference = "Stop"

# ==== 設定 ====
$PROJECT = "studious-lore-473812-m0"
$BUCKET  = "gs://cheer-studious-lore-473812-m0"
$REGION  = "asia-northeast1"
$PY      = "python"                               # python -m yt_dlp を使う
$TMPDIR  = Join-Path $env:TEMP "yt2gcs"
New-Item -ItemType Directory -Force -Path $TMPDIR | Out-Null

# 1.25倍にする“遅テンポ”チャンネル（クリーン）
$CLEAN_CHANNELS = @(
  "UCze00Dglxs8RsBnFsTLBLRQ",
  "UCdWUgneQfKR3hJHlJsTuc7Q"
)

# クリーン（単発URL）
$CLEAN_URLS = @(
  "https://youtu.be/xJaIrB-VkyA?si=hUlN4q9OHjmfNl_w",
  "https://youtu.be/XP8R1lMWEtc?si=_NWA_7FoxzFdpn_B",
  "https://youtu.be/Owj17xOyEJE?si=y6qFqHQiay3a6GZR"
)

# ノイジー（球場系）
$NOISY_URLS = @(
  "https://youtu.be/4PaFCQZOBKU?si=rGYyIiYzchz7izp8",
  "https://youtu.be/_MLCKlaH7f4?si=Hb5z9TuWTDboMNJD",
  "https://youtu.be/qRiUr0ne6Lw?si=r49XWTG4nQd3NaiZ",
  "https://youtu.be/GdbqSM-Oc6k?si=VHbA4Jinhqt3OLw8",
  "https://youtu.be/XbjLx8T6DwA?si=B4tioNrpzv-Gq12k"
)

# チャンネルから取る最大本数（全部=0）
$MAX_PER_CHANNEL = 0

function UploadFromUrl($url, $label, $tempo) {
  try {
    # 動画ID取得
    $id = (& $PY -m yt_dlp --get-id "$url" 2>$null | Select-Object -Last 1).Trim()
    if (-not $id) { Write-Host "ID取得失敗: $url"; return }

    $dest = "$BUCKET/$label/$id.flac"
    $af   = if ($tempo -ne "1.0") { "atempo=$tempo" } else { "anull" }
    Write-Host "=> $label | $id | tempo=$tempo -> $dest"

    # 1) 音声だけDL（拡張子は自動；idで特定）
    $tpl = Join-Path $TMPDIR "%(id)s.%(ext)s"
    & $PY -m yt_dlp -f ba --no-part -o $tpl "$url" | Out-Null

    $infile = Get-ChildItem -Path (Join-Path $TMPDIR "$id.*") -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $infile) { throw "DLファイル見つからず: $id" }

    # 2) 16kHz/mono/FLACへ変換
    $outfile = Join-Path $TMPDIR "$id.flac"
    ffmpeg -y -hide_banner -loglevel error -i $infile.FullName -ar 16000 -ac 1 -af $af -c:a flac $outfile

    # 3) GCSへアップロード
    gsutil -m cp $outfile $dest | Out-Null
    Write-Host "   OK: $dest"

  } catch {
    Write-Host "   SKIP: $url  ($_)" -ForegroundColor Yellow
  } finally {
    # 後始末
    Get-ChildItem -Path (Join-Path $TMPDIR "$id.*") -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  }
}

function UploadChannelAll($channelId, $label, $tempo, $max) {
  Write-Host "== Channel: $channelId (label=$label, tempo=$tempo) =="
  $ids = & $PY -m yt_dlp --no-warnings --flat-playlist --print id "https://www.youtube.com/channel/$channelId/videos" 2>$null
  if (-not $ids) { Write-Host "  取得ゼロ: $channelId"; return }
  $count = 0
  foreach ($vid in $ids) {
    if ([string]::IsNullOrWhiteSpace($vid)) { continue }
    UploadFromUrl "https://youtu.be/$vid" $label $tempo
    $count++
    if ($max -gt 0 -and $count -ge $max) { break }
  }
}

# ===== 実行 =====
try { & gcloud config set project $PROJECT | Out-Null } catch {}
try { & gsutil ls $BUCKET   | Out-Null } catch { & gsutil mb -l $REGION $BUCKET | Out-Null }

# 1) クリーン：チャンネル（1.25x）
foreach ($cid in $CLEAN_CHANNELS) { UploadChannelAll $cid "clean" "1.25" $MAX_PER_CHANNEL }

# 2) クリーン：単発URL（チャンネル一致なら自動1.25x）
foreach ($url in $CLEAN_URLS) {
  $cid = (& $PY -m yt_dlp --print channel_id "$url" 2>$null | Select-Object -Last 1).Trim()
 if ($CLEAN_CHANNELS -contains $cid) {
  $tempo = "1.25"
} else {
  $tempo = "1.0"
}
  UploadFromUrl $url "clean" $tempo
}

# 3) ノイジーURL
foreach ($url in $NOISY_URLS) { UploadFromUrl $url "noisy" "1.0" }

Write-Host "== 完了：gsutil ls $BUCKET/clean/ と $BUCKET/noisy/ で確認してください =="
# ===== /upload_all.ps1 =====
