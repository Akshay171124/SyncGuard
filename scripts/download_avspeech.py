"""Download AVSpeech clips from YouTube using yt-dlp.

Downloads a subset of clips from the AVSpeech CSV metadata and saves them
in a structured directory layout ready for preprocessing.

Directory structure created:
    AVSpeech/
    ├── avspeech_train.csv          # Metadata (already downloaded)
    ├── avspeech_test.csv
    ├── clips/
    │   ├── <youtube_id_1>.mp4
    │   ├── <youtube_id_2>.mp4
    │   └── ...
    ├── download_log.json           # Tracks successful/failed downloads
    └── manifest.csv                # Final manifest: youtube_id, start, end, face_x, face_y, clip_path

Usage:
    # Download 25K clips (default)
    python scripts/download_avspeech.py

    # Download a custom number of clips
    python scripts/download_avspeech.py --num_clips 1000

    # Resume a previously interrupted download
    python scripts/download_avspeech.py --resume

    # Use test set instead of train
    python scripts/download_avspeech.py --split test --num_clips 1000

    # Download in batches (e.g., 5K at a time for manual upload)
    python scripts/download_avspeech.py --num_clips 5000 --batch_id 1
    python scripts/download_avspeech.py --num_clips 5000 --batch_id 2
"""

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_NUM_CLIPS = 25000
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "AVSpeech"
CLIPS_SUBDIR = "clips"


def load_csv(csv_path: Path) -> list[dict]:
    """Load AVSpeech CSV. Format: youtube_id, start, end, face_x, face_y."""
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            rows.append({
                "youtube_id": row[0].strip(),
                "start": float(row[1].strip()),
                "end": float(row[2].strip()),
                "face_x": float(row[3].strip()),
                "face_y": float(row[4].strip()),
            })
    return rows


def load_download_log(log_path: Path) -> dict:
    """Load existing download log for resume support."""
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        # Migrate old format: failed was a list, now a dict with error messages
        if isinstance(log.get("failed"), list):
            log["failed"] = {yt_id: "unknown" for yt_id in log["failed"]}
        return log
    return {"successful": [], "failed": {}, "unavailable": []}


def save_download_log(log_path: Path, log: dict):
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def download_clip(
    youtube_id: str, start: float, end: float, output_path: Path,
    cookies_browser: str | None = None,
) -> tuple[bool, str]:
    """Download a single clip segment using yt-dlp.

    Returns:
        (success, message)
    """
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        # Download section
        "--download-sections", f"*{start}-{end}",
        # Format: 360p mp4 preferred (smaller files)
        "-f", "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best[height<=360]",
        "--merge-output-format", "mp4",
        # Output
        "-o", str(output_path),
        # Limits
        "--socket-timeout", "15",
        "--retries", "2",
        "--no-playlist",
        url,
    ]

    if cookies_browser:
        cmd.insert(1, "--cookies-from-browser")
        cmd.insert(2, cookies_browser)
        cmd.insert(3, "--remote-components")
        cmd.insert(4, "ejs:github")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and output_path.exists():
            return True, "ok"
        else:
            stderr = result.stderr.strip()
            if "Video unavailable" in stderr or "Private video" in stderr:
                return False, "unavailable"
            return False, stderr[:200] if stderr else "unknown error"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:200]


def write_manifest(manifest_path: Path, entries: list[dict]):
    """Write final manifest CSV with clip paths."""
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["youtube_id", "start", "end", "face_x", "face_y", "clip_path"])
        for e in entries:
            writer.writerow([
                e["youtube_id"], e["start"], e["end"],
                e["face_x"], e["face_y"], e["clip_path"],
            ])


def main():
    parser = argparse.ArgumentParser(description="Download AVSpeech clips")
    parser.add_argument(
        "--num_clips", type=int, default=DEFAULT_NUM_CLIPS,
        help=f"Number of clips to download (default: {DEFAULT_NUM_CLIPS})",
    )
    parser.add_argument(
        "--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
        help="Path to AVSpeech data directory",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"],
        help="Which split to download from",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous download log",
    )
    parser.add_argument(
        "--batch_id", type=int, default=None,
        help="Batch ID for downloading in chunks (e.g., 1, 2, 3). "
             "Each batch downloads --num_clips starting from batch_id * num_clips offset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible clip selection",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of concurrent downloads (be gentle on YouTube)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Delay in seconds between downloads to avoid YouTube throttling (default: 2.0)",
    )
    parser.add_argument(
        "--cookies-from-browser", type=str, default=None,
        dest="cookies_browser",
        help="Browser to extract cookies from (e.g., chrome, firefox, safari)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    clips_dir = data_dir / CLIPS_SUBDIR
    clips_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f"avspeech_{args.split}.csv"
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        logger.error("Download it first from https://huggingface.co/datasets/bbrothers/avspeech-metadata")
        sys.exit(1)

    log_path = data_dir / "download_log.json"
    manifest_path = data_dir / "manifest.csv"

    # Load metadata
    logger.info(f"Loading {csv_path} ...")
    all_rows = load_csv(csv_path)
    logger.info(f"Total entries in CSV: {len(all_rows):,}")

    # Shuffle deterministically and select subset
    random.seed(args.seed)
    random.shuffle(all_rows)

    if args.batch_id is not None:
        offset = (args.batch_id - 1) * args.num_clips
        selected = all_rows[offset : offset + args.num_clips]
        logger.info(f"Batch {args.batch_id}: clips {offset+1} to {offset+len(selected)}")
    else:
        selected = all_rows[: args.num_clips]

    logger.info(f"Selected {len(selected):,} clips for download")

    # Resume support
    log = load_download_log(log_path) if args.resume else {"successful": [], "failed": {}, "unavailable": []}
    already_done = set(log["successful"] + log["unavailable"] + list(log["failed"].keys()))

    if args.resume:
        logger.info(f"Resuming: {len(log['successful'])} done, {len(log['unavailable'])} unavailable, {len(log['failed'])} failed (skipping failed)")

    # Download loop
    success_count = len(log["successful"])
    fail_count = len(log["failed"])
    unavailable_count = len(log["unavailable"])
    manifest_entries = []

    # Rebuild manifest entries for already-downloaded clips
    if args.resume:
        for row in selected:
            yt_id = row["youtube_id"]
            if yt_id in set(log["successful"]):
                clip_path = clips_dir / f"{yt_id}.mp4"
                if clip_path.exists():
                    manifest_entries.append({**row, "clip_path": str(clip_path)})

    start_time = time.time()

    for i, row in enumerate(selected):
        yt_id = row["youtube_id"]

        # Skip already processed
        if yt_id in already_done:
            continue

        clip_path = clips_dir / f"{yt_id}.mp4"

        # Skip if file already exists on disk
        if clip_path.exists():
            log["successful"].append(yt_id)
            already_done.add(yt_id)
            manifest_entries.append({**row, "clip_path": str(clip_path)})
            success_count += 1
            continue

        ok, msg = download_clip(yt_id, row["start"], row["end"], clip_path, cookies_browser=args.cookies_browser)

        if ok:
            log["successful"].append(yt_id)
            manifest_entries.append({**row, "clip_path": str(clip_path)})
            success_count += 1
        elif msg == "unavailable":
            log["unavailable"].append(yt_id)
            unavailable_count += 1
        else:
            log["failed"][yt_id] = msg
            fail_count += 1

        already_done.add(yt_id)

        # Throttle to avoid YouTube rate-limiting
        time.sleep(args.delay)

        # Progress logging every 50 clips
        total_processed = success_count + unavailable_count + fail_count
        if total_processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"[{total_processed}/{len(selected)}] "
                f"OK: {success_count} | Unavail: {unavailable_count} | Fail: {fail_count} | "
                f"Rate: {rate:.1f} clips/s"
            )
            # Save checkpoint
            save_download_log(log_path, log)

    # Final save
    save_download_log(log_path, log)
    write_manifest(manifest_path, manifest_entries)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Download complete in {elapsed/60:.1f} minutes")
    logger.info(f"  Successful:   {success_count}")
    logger.info(f"  Unavailable:  {unavailable_count}")
    logger.info(f"  Failed:       {fail_count}")
    logger.info(f"  Manifest:     {manifest_path}")
    logger.info(f"  Clips dir:    {clips_dir}")
    logger.info("=" * 60)

    # Estimate storage
    total_size = sum(f.stat().st_size for f in clips_dir.glob("*.mp4"))
    logger.info(f"  Total storage: {total_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
