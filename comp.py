import subprocess
import time
import os
import re
import math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
INPUT_VIDEO   = "/content/drive/MyDrive/kik/bbb10_lossless.mov"
RESOLUTIONS   = ["176x144", "352x288", "720x480", "1280x720", "1920x1080"]
BITRATES_KBPS = [100, 500, 1000, 2500, 5000, 7500, 10000, 15000]

# ★ SPEEDUP #1 — only encode this many seconds of video instead of the full clip.
#   For a ~10-minute video this alone gives a ~60× wall-clock reduction.
CLIP_DURATION = 10    # seconds

# ★ SPEEDUP #2 — run this many encode jobs at the same time.
#   Colab standard tier has 2 vCPUs; 4 workers keeps both busy while one waits on I/O.
MAX_WORKERS   = 4

CODECS = {
    "H.261":  "h261",
    "MPEG-1": "mpeg1video",
    "MPEG-4": "mpeg4",
    "H.264":  "libx264",
    "H.265":  "libx265",
}

# ★ SPEEDUP #3 — ultrafast presets for software encoders (5-10× vs default).
SPEED_FLAGS = {
    "libx264": ["-preset", "ultrafast"],
    "libx265": ["-preset", "ultrafast", "-x265-params", "log-level=error"],
}

# ★ SPEEDUP #4 — hardware encoding when an NVIDIA GPU is present (automatic).
GPU_MAP = {
    "libx264": "h264_nvenc",
    "libx265": "hevc_nvenc",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def detect_nvenc() -> bool:
    """Return True if NVIDIA GPU with h264_nvenc support is present."""
    try:
        if subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode != 0:
            return False
        enc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_nvenc" in enc.stdout
    except Exception:
        return False


def get_psnr(reference: str, encoded: str) -> float:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", encoded, "-i", reference,
         "-lavfi", "psnr", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"average:([0-9.]+)", result.stderr)
    return float(m.group(1)) if m else 0.0


def make_reference(src: str, resolution: str, duration: int) -> str:
    """Create a lossless clipped+scaled reference video."""
    out = f"ref_{resolution}.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y",
         "-i", src, "-t", str(duration),
         "-vf", f"scale={resolution}",
         "-c:v", "libx264", "-crf", "0", "-an", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


# ─── CORE ENCODE TASK (runs inside thread pool) ───────────────────────────────

def encode_task(args: tuple):
    """Single (codec × bitrate) encode + PSNR measurement. Thread-safe."""
    ref_video, ref_size, codec_name, ffmpeg_codec, br, res, use_gpu = args

    # Unique temp filename per thread to avoid collisions
    tmp = f"tmp_{res}_{codec_name}_{br}k_{threading.get_ident()}.mkv"
    actual_codec = GPU_MAP.get(ffmpeg_codec, ffmpeg_codec) if use_gpu else ffmpeg_codec

    cmd = ["ffmpeg", "-hide_banner", "-y",
           "-i", ref_video,
           "-c:v", actual_codec, "-b:v", f"{br}k"]
    if not use_gpu:
        cmd += SPEED_FLAGS.get(ffmpeg_codec, [])
    cmd += ["-an", tmp]

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.time() - t0

    if proc.returncode != 0 or not os.path.exists(tmp):
        print(f"\n  ✗ {codec_name} @ {br}kbps failed", flush=True)
        return None

    comp_size = os.path.getsize(tmp)
    psnr      = get_psnr(ref_video, tmp)
    ratio     = ref_size / comp_size if comp_size > 0 else 0.0
    os.remove(tmp)
    return br, psnr, ratio, elapsed


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: {INPUT_VIDEO} not found.")
        return

    use_gpu = detect_nvenc()
    print(f"GPU:     {'✓ NVENC (h264_nvenc / hevc_nvenc)' if use_gpu else '✗ CPU — ultrafast preset active'}")
    print(f"Clip:    {CLIP_DURATION}s  |  Workers: {MAX_WORKERS}  |  Resolutions: {RESOLUTIONS}\n")

    all_data = {
        res: {c: {"psnr": [], "ratio": [], "time": [], "bitrate": []} for c in CODECS}
        for res in RESOLUTIONS
    }

    t_total = time.time()

    for res in RESOLUTIONS:
        print(f"\n{'─'*55}\nResolution: {res}")
        ref      = make_reference(INPUT_VIDEO, res, CLIP_DURATION)
        ref_size = os.path.getsize(ref)

        # Build all (codec × bitrate) tasks for this resolution
        tasks = []
        for codec_name, ffmpeg_codec in CODECS.items():
            if codec_name == "H.261" and res not in ("176x144", "352x288"):
                continue
            for br in BITRATES_KBPS:
                tasks.append((ref, ref_size, codec_name, ffmpeg_codec, br, res, use_gpu))

        done = 0
        print(f"  {len(tasks)} jobs queued...", flush=True)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(encode_task, t): t[2] for t in tasks}  # t[2] = codec_name
            for future in as_completed(futures):
                codec_name = futures[future]
                result     = future.result()
                if result:
                    br, psnr, ratio, elapsed = result
                    d = all_data[res][codec_name]
                    d["bitrate"].append(br)
                    d["psnr"].append(psnr)
                    d["ratio"].append(ratio)
                    d["time"].append(elapsed)
                done += 1
                print(f"\r  Progress: {done}/{len(tasks)}", end="", flush=True)

        print("  ✓")

        # Sort each codec's data by bitrate for clean plot curves
        for c in CODECS:
            d = all_data[res][c]
            if not d["bitrate"]:
                continue
            sorted_rows = sorted(zip(d["bitrate"], d["psnr"], d["ratio"], d["time"]))
            d["bitrate"], d["psnr"], d["ratio"], d["time"] = map(list, zip(*sorted_rows))

        os.remove(ref)

    print(f"\nTotal encode time: {time.time() - t_total:.1f}s")
    print("Generating plots...")
    generate_plots(all_data)


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def generate_plots(data: dict):
    n    = len(RESOLUTIONS)
    cols = 2
    rows = math.ceil(n / cols)

    plot_specs = [
        ("Compression Ratio vs PSNR",  "PSNR (dB)",      "Compression Ratio",    "psnr",    "ratio"),
        ("PSNR vs Bitrate",             "Bitrate (kbps)", "PSNR (dB)",            "bitrate", "psnr"),
        ("Compression Time vs Bitrate", "Bitrate (kbps)", "Compression Time (s)", "bitrate", "time"),
    ]
    filenames = [
        "1_Compression_Ratio_vs_PSNR.png",
        "2_PSNR_vs_Bitrate.png",
        "3_Compression_Time_vs_Bitrate.png",
    ]

    for (title, xlabel, ylabel, xkey, ykey), fname in zip(plot_specs, filenames):
        # squeeze=False ensures axs is always 2D regardless of row/col count
        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)
        fig.suptitle(title, fontsize=16)
        axs_flat = axs.flatten()

        for idx, res in enumerate(RESOLUTIONS):
            ax = axs_flat[idx]
            for codec in CODECS:
                d = data[res][codec]
                if not d["psnr"]:
                    continue
                ax.plot(d[xkey], d[ykey], marker="x", label=codec)
            ax.set_title(res)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--")
            ax.legend()

        for ax in axs_flat[n:]:   # hide leftover subplots (e.g. 6th cell for 5 resolutions)
            ax.axis("off")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(fname)
        print(f"  Saved {fname}")

    plt.show()


if __name__ == "__main__":
    main()