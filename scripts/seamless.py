"""Turn a MiniMax clip into a genuinely seamless web loop, and prove it did.

Run: python seamless.py <in.mp4> <out.mp4> [--fade 16]

## Why this step exists at all

`MiniMaxH3ImageToVideo` takes `first_frame` and `last_frame`, which reads like a
way to pin both ends of a clip to the same picture and get a perfect loop for
free. It is not. Measured on the first render here:

    source still vs frame 0      19.18   <- the model did not reproduce the input
    frame 0   vs frame 72        16.53   <- ends nowhere near each other
    frame 0   vs frame 36        17.30   <- ...which is the whole motion budget

Both frames are CONDITIONING, not pixels. The model paints its own
interpretation of each end, so the seam came out as large as the motion and the
loop visibly jumped. Global brightness drift was ruled out separately: first to
last moved 1.43 of luminance, while the middle rose 7.6 and came back, which is
a bloom-and-return, not a fade.

So the loop is closed here instead, by crossfading the tail back over the head.
The arithmetic, for a clip of F frames and a fade of K:

    L = F - K                      the loop's real length
    out[i] = x[i]                  for i in [K, L)
    out[i] = lerp(x[i+L], x[i], w) for i in [0, K), w running 0 -> 1

`out[L-1]` is `x[L-1]` and `out[0]` opens on `x[L]`, which is the frame that
genuinely followed it — so the wrap is an ordinary frame step rather than a cut.
It costs K frames of duration and keeps every motion running forwards, which the
reverse/forward (ping-pong) sandwich does not: that one
loops perfectly and plays the second half backwards, and on petals and water it
reads as breathing.

## The check at the end is the point

A seam is only invisible if it is no larger than the frame-to-frame change
around it, so that is what gets compared — not the seam against zero. Measured
with ffmpeg alone (`blend=difference` into `signalstats`), so this needs no
Python imaging stack.
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
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "json", str(path)])
    return int(json.loads(r.stdout)["streams"][0]["nb_read_frames"])


def grab(src, n, dest):
    run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"select=eq(n\\,{n})",
         "-fps_mode", "passthrough", "-frames:v", "1", str(dest), "-y"])


def mean_diff(a, b):
    """Mean luma difference between two stills, 0-255, via ffmpeg only.

    `-v info`, not `-v error`: `metadata=print` writes its lines at info level,
    so quietening ffmpeg the way every other call here does removes the one
    thing this function reads and returns nan for a clip that is fine.
    """
    r = run(["ffmpeg", "-v", "info", "-i", str(a), "-i", str(b),
             "-filter_complex", "[0][1]blend=all_mode=difference,signalstats,metadata=print",
             "-f", "null", "-"])
    m = re.findall(r"lavfi\.signalstats\.YAVG=([\d.]+)", r.stderr + r.stdout)
    return float(m[-1]) if m else float("nan")


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    fade = 16
    if "--fade" in sys.argv:
        fade = int(sys.argv[sys.argv.index("--fade") + 1])

    total = frames_of(src)
    keep = total - fade
    if keep <= fade:
        sys.exit(f"clip is {total} frames; a {fade}-frame fade leaves nothing")

    # Written to a file rather than passed inline: the blend expression carries
    # commas, which a filtergraph on the command line treats as filter
    # separators however they are quoted.
    graph = (
        f"[0:v]split=2[s0][s1];"
        f"[s0]trim=start_frame=0:end_frame={keep},setpts=PTS-STARTPTS[base];"
        f"[s1]trim=start_frame={keep},setpts=PTS-STARTPTS[tail];"
        f"[base]split=2[b0][b1];"
        f"[b0]trim=start_frame=0:end_frame={fade},setpts=PTS-STARTPTS[bh];"
        f"[b1]trim=start_frame={fade},setpts=PTS-STARTPTS[bt];"
        f"[bh][tail]blend=all_expr='A*(N/{fade - 1})+B*(1-N/{fade - 1})'[mix];"
        f"[mix][bt]concat=n=2:v=1[out]"
    )
    with tempfile.TemporaryDirectory() as tmp:
        gp = Path(tmp) / "graph.txt"
        gp.write_text(graph, encoding="utf-8")

        def build(out, *enc):
            return run(["ffmpeg", "-v", "error", "-i", str(src),
                        "-filter_complex_script", str(gp), "-map", "[out]",
                        "-r", "24", *enc, "-an", str(out), "-y"])

        # The deliverable.
        r = build(dst, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                  "-preset", "slow", "-movflags", "+faststart")
        if r.returncode:
            sys.exit(r.stderr[-800:])

        # ...and a LOSSLESS twin, which is what gets measured.
        #
        # Measuring the shipped file overstates the seam badly, and it took a
        # while to see why: its frame 0 is an I-frame while everything around
        # it is a P-frame, so a difference against it carries the gap between
        # two encodings on top of the gap between two pictures. On this clip
        # that was 4.65 lossy against 2.99 lossless — 1.66 of pure codec, which
        # is enough to fail a loop that is actually clean, and it does not move
        # when the clip is trimmed, which is what gave it away.
        m = Path(tmp) / "measure.mp4"
        r = build(m, "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p")
        if r.returncode:
            m = dst  # fall back rather than refuse to report at all

        n = frames_of(m)

        # Every interior frame-to-frame step, in ONE pass. An earlier version
        # sampled a single transition near the middle and compared the seam to
        # that -- which passed a clip whose wrap was in fact its largest jump.
        # A quiet midpoint is not "normal"; the distribution is.
        r = run(["ffmpeg", "-v", "info", "-i", str(m),
                 "-vf", "tblend=all_mode=difference,signalstats,metadata=print",
                 "-f", "null", "-"])
        steps = [float(x) for x in
                 re.findall(r"signalstats\.YAVG=([\d.]+)", r.stderr + r.stdout)][1:]

        for label, idx in (("first", 0), ("mid", n // 2), ("last", n - 1)):
            grab(m, idx, Path(tmp) / f"{label}.png")
        f = lambda x: Path(tmp) / f"{x}.png"
        seam = mean_diff(f("last"), f("first"))
        motion = mean_diff(f("first"), f("mid"))

    srt = sorted(steps)
    med, p90, mx = srt[len(srt) // 2], srt[int(len(srt) * .9)], srt[-1]
    pct = sum(1 for x in steps if x < seam) / len(steps)

    kb = dst.stat().st_size / 1024
    print(f"{dst.name}: {n} frames, {n / 24:.2f}s, {kb:.0f} KB")
    print(f"  motion   median step {med:.2f}  p90 {p90:.2f}  max {mx:.2f}")
    print(f"  seam     {seam:.2f}  ({pct * 100:.0f}th percentile of ordinary steps)")
    print(f"  spread   first->mid {motion:.2f}")
    # Two signals, because either alone misleads. A percentile is harsh on a
    # clip whose steps are tightly bunched — a seam 27% above the median can
    # land at the 98th percentile and still be smaller than the largest
    # ordinary step in the shot. A ratio alone is harsh the other way, on a
    # clip that is mostly still with one big move in it. The wrap passes when
    # it is inside the range the clip already contains AND not far above its
    # typical step.
    within = seam <= mx
    typical = seam <= med * 1.5
    if within and typical:
        print("  OK - the wrap is inside the movement the clip already has")
    elif typical:
        print("  MARGINAL - larger than any ordinary step, but close to typical;")
        print("  look at the wrap before shipping it.")
    else:
        print(f"  SEAM VISIBLE - {seam / med:.1f}x the typical step.")
        print("  A longer --fade will not fix a shot that travels one way;")
        print("  shorten the clip or re-render with less directional motion.")

if __name__ == "__main__":
    main()
