"""Chain several short clips into one long loop, without losing their speed.

Run: python chain.py <out.mp4> --fade 14 <a.mp4> <b.mp4> <c.mp4> ...

## Why this exists

MiniMax H3 trades speed for duration, and the trade is close to intrinsic.
Measured on one image, one prompt, one set of settings, changing only length:

     73 frames   median frame step 4.55   cumulative 325
    192 frames   median frame step 1.25   cumulative 293

The same journey, 2.6x the time — slow motion. Rewriting the prompt as
sequential beats ("she strides and keeps striding, a gust tears through,
petals burst...", plus an explicit "no slow motion") barely moved it: 1.25 to
1.30. The model paces whatever it is given across whatever duration it is
given, near enough regardless of what the prompt asks for.

So a long clip that still moves cannot come from one render. It comes from
several short fast ones, chained — which also fixes the thing that actually
makes a short loop tiring. A 2.2s loop repeats 27 times a minute; what the eye
picks up is not the seam (that measures clean) but the REPETITION. Three takes
at different seeds play ~7 seconds before anything recurs.

## The construction

For N clips of F frames and a fade of K, each clip contributes one segment:

    segment[i] = crossfade(clip[i-1] tail K, clip[i] head K)  ++  clip[i][K : F-K]

Every segment is `F-K` long, so the output is `N x (F-K)`. The wrap closes
itself: the last segment ends on `clip[N-1][F-K-1]` and the output's first
frame opens on `clip[N-1][F-K]` blended toward `clip[0][0]` — consecutive
frames of the same clip, so the join is an ordinary frame step.

The joins BETWEEN clips are dissolves between different takes of the same
scene. That reads as a deliberate transition rather than a stutter, which a
repeat of identical footage never does.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(args):
    return subprocess.run(args, capture_output=True, text=True)


def frames_of(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)])
    return int(json.loads(r.stdout)["streams"][0]["nb_read_frames"])


def size_of(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", str(path)])
    st = json.loads(r.stdout)["streams"][0]
    return st["width"], st["height"]


def build_segment(prev_clip, this_clip, total, fade, out, tmp):
    """crossfade(prev tail, this head) ++ this[fade : total-fade]"""
    graph = (
        f"[0:v]trim=start_frame={total - fade},setpts=PTS-STARTPTS[tail];"
        f"[1:v]split=2[h][b];"
        f"[h]trim=start_frame=0:end_frame={fade},setpts=PTS-STARTPTS[head];"
        f"[b]trim=start_frame={fade}:end_frame={total - fade},setpts=PTS-STARTPTS[body];"
        # A is the incoming clip's head, B the outgoing clip's tail: N=0 opens
        # on pure tail, N=fade-1 lands on pure head.
        f"[head][tail]blend=all_expr='A*(N/{fade - 1})+B*(1-N/{fade - 1})'[mix];"
        f"[mix][body]concat=n=2:v=1[out]"
    )
    gp = Path(tmp) / f"g{out.stem}.txt"
    gp.write_text(graph, encoding="utf-8")
    r = run(["ffmpeg", "-v", "error", "-i", str(prev_clip), "-i", str(this_clip),
             "-filter_complex_script", str(gp), "-map", "[out]", "-r", "24",
             "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", "-an",
             str(out), "-y"])
    if r.returncode:
        sys.exit(r.stderr[-800:])


def measure(path):
    """Frame-step distribution and the wrap, both off a lossless source."""
    r = run(["ffmpeg", "-v", "info", "-i", str(path),
             "-vf", "tblend=all_mode=difference,signalstats,metadata=print",
             "-f", "null", "-"])
    steps = [float(x) for x in
             re.findall(r"signalstats\.YAVG=([\d.]+)", r.stderr + r.stdout)][1:]
    n = frames_of(path)
    with tempfile.TemporaryDirectory() as t:
        a, b = Path(t) / "a.png", Path(t) / "b.png"
        for idx, dst in ((0, a), (n - 1, b)):
            run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})",
                 "-fps_mode", "passthrough", "-frames:v", "1", str(dst), "-y"])
        r = run(["ffmpeg", "-v", "info", "-i", str(b), "-i", str(a), "-filter_complex",
                 "[0][1]blend=all_mode=difference,signalstats,metadata=print",
                 "-f", "null", "-"])
        m = re.findall(r"signalstats\.YAVG=([\d.]+)", r.stderr + r.stdout)
        seam = float(m[-1]) if m else float("nan")
    return steps, seam, n


def main():
    out = Path(sys.argv[1])
    fade = 14
    if "--fade" in sys.argv:
        fade = int(sys.argv[sys.argv.index("--fade") + 1])
    clips = [Path(a) for a in sys.argv[2:] if a.endswith(".mp4")]
    if len(clips) < 2:
        sys.exit("give at least two clips")

    # `blend` needs identical dimensions and fails deep inside the filtergraph
    # with "Error reinitializing filters!", which names neither the cause nor
    # the file. Checked up front instead, and NOT silently rescaled: clips of
    # different sizes are a mistake about which renders belong together, and
    # stretching one to match would hide it.
    sizes = {c.name: size_of(c) for c in clips}
    if len(set(sizes.values())) > 1:
        for name, wh in sizes.items():
            print(f"  {name}: {wh[0]}x{wh[1]}")
        sys.exit("clips must all be the same size; re-render the odd one to match")

    counts = [frames_of(c) for c in clips]
    total = min(counts)
    if len(set(counts)) > 1:
        print(f"clips differ in length {counts}; using {total} from each")
    if total <= fade * 2:
        sys.exit(f"{total} frames is too few for a {fade}-frame fade")

    with tempfile.TemporaryDirectory() as tmp:
        segs = []
        for i, c in enumerate(clips):
            s = Path(tmp) / f"seg{i}.mp4"
            build_segment(clips[i - 1], c, total, fade, s, tmp)  # i-1 wraps at 0
            segs.append(s)

        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), encoding="utf-8")

        # Lossless join first, so the measurement below is not reading codec
        # noise off an I-frame (see seamless.py for why that matters).
        joined = Path(tmp) / "joined.mp4"
        r = run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                 "-c", "copy", str(joined), "-y"])
        if r.returncode:
            sys.exit(r.stderr[-800:])

        steps, seam, n = measure(joined)

        r = run(["ffmpeg", "-v", "error", "-i", str(joined), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow",
                 "-movflags", "+faststart", "-an", str(out), "-y"])
        if r.returncode:
            sys.exit(r.stderr[-800:])

    srt = sorted(steps)
    med, mx = srt[len(srt) // 2], srt[-1]
    kb = out.stat().st_size / 1024
    print(f"{out.name}: {len(clips)} clips -> {n} frames, {n / 24:.2f}s, {kb:.0f} KB")
    print(f"  motion  median step {med:.2f}  max {mx:.2f}")
    print(f"  wrap    {seam:.2f}")
    if seam <= mx and seam <= med * 1.5:
        print("  OK - the wrap is inside the movement the chain already has")
    else:
        print(f"  WRAP VISIBLE - {seam / med:.1f}x the typical step; widen --fade")


if __name__ == "__main__":
    main()
