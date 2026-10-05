---
name: seamless-loop-video
description: |
  Turn a still image into a seamlessly looping ambient video for a web
  header or background, using MiniMax H3 in the local ComfyUI. Use when asked
  for a looping background/hero video, "make this image move", ambient motion
  from artwork, or when a generated loop visibly jumps, has no wind/atmosphere,
  or barely moves. Covers the settings that actually produce motion, how to
  close the loop, and how to measure whether it worked.
---

# Seamless looping video from a still

Two stages, and keeping them separate is the whole trick:

1. **`make-loop.py`** renders motion. It does NOT try to loop.
2. The loop is closed with REAL frames (bridge take, cut point, or flow in-betweens) composed in Remotion/HyperFrames. **Avoid crossfades** for closing the loop; they ghost on directional motion.
3. **`chain.py`** joins several short takes into one long loop, for when a
   single render cannot be both long and fast (§5).

The three scripts are in `scripts/`. Set `COMFY_URL` if your ComfyUI is not on
`http://127.0.0.1:8188`. Everything below was learned by getting it wrong first; each rule has the measurement that produced it.

```bash
python make-loop.py <image> "<prompt>" --width 544 --height 800 --length 73 --steps 8 --physics 80 --seed 11
python seamless.py  <raw.mp4> <out.mp4> --fade 20
python chain.py     <out.mp4> --fade 14 <a.mp4> <b.mp4> <c.mp4>
```

---

## 1. Getting motion at all

The first renders had no wind, no atmosphere and no walk. Every one of these
was set to the motion-suppressing value:

| Setting | Wrong | Right | Why |
|---|---|---|---|
| `last_frame` | = `first_frame` | **omit it** | Pinning the end to the opening still is an instruction to *return to where it started*, so the model buys the loop by refusing to move. This is the single biggest cause. |
| steps | 4 | **8** | The turbo LoRA is marketed as 4-step. Both published i2v graphs use 8 (euler) or 20 (res_multistep). Four is the floor for a coherent picture, not for movement. |
| `shift_video` | 12.0 (node default) | **6.0** | Every reference workflow passes 6. |
| LoRA stack | turbo only | **turbo + `wushu_spatial_physics_v2`** | A spatial-physics adapter. This is what puts weight behind cloth, water and a walking figure. |
| prompt | "gentle", "subtle", "camera perfectly still" | active physics verbs | Asking for subtlety gets subtlety. |

Measured effect of fixing all five, same source image:

```
                    before   after
median frame step     1.73  →  4.55     2.6x
cumulative movement    133  →   325     2.4x
```

**Write prompts as physics, not mood.** Name what moves and how: *"she walks
forward, her hair and dress blown by strong wind; petals tear loose and tumble
spinning; the splash surges and ripples; mist rolls across the ground"*. Avoid
"ambient", "gentle", "subtle", "slow", "camera locked" — they all cost motion.

### Length, and the trap in it

`--length` sits on the model's 17k+5 grid (5, 22, 39, 56, 73 … 192, 243). The
node snaps up silently otherwise. The model's trained range is ~124-362 frames
(its own tooltip), so 73 is below what it was trained on and 192 is exactly 8.0s.

**But length does not buy motion, and assuming it does is a wasted render.**
Same image, same prompt, same settings, only the frame count changed:

```
 73 frames   median frame step 4.55   cumulative movement 325
192 frames   median frame step 1.25   cumulative movement 293
```

Identical journey, 2.6x the time — i.e. **slow motion**. The model paces
whatever the prompt describes across whatever duration it is given;
motion-per-second x duration is roughly constant for a given prompt.

So the two levers are independent and both are needed:

- **Duration** comes from `--length`.
- **Speed** comes from the PROMPT having enough to do. A scene description
  ("wind blows, petals drift") is about three seconds of material; stretched to
  eight it crawls.

For a long clip that still moves, write **sequential beats** and say so
explicitly: *"she strides forward and keeps striding, never pausing; a hard gust
tears through; petals rip loose and burst into a swirling cloud; the splash
surges up and crashes back; the lattice flares and pulses rapidly; mist sweeps
across and clears. Continuous rapid movement from first frame to last, no
stillness, no slow motion."*

Slow pacing is not always wrong — for a scrimmed background it can be better
than frantic. Judge it per surface: heroes want speed, plates want drift.

192 frames is ~2.6x the VRAM-time of 73: 662s at 544x800 on a 12 GB RTX 3060,
which held with 1.7 GB spare. Drop resolution rather than length to fit.

### Models this uses (all local, all must be installed in your ComfyUI)

```
UNET    minimax_h3_fl_or_ref2va_w4a8_pruned_a.safetensors
LoRA 1  minimax_h3_fl2v_turbo_4step_v1.2_768p.safetensors      1.0
LoRA 2  Minimax H3\wushu_spatial_physics_v2_1000_pruned.safetensors  0.8
CLIP    minimaxH3Fl2vaPruned_fl2vaPrunedFp8Scaled_txt.safetensors   type "minimax"
VAE     minimax_h3_video_vae_fp16.safetensors
```

Do **not** reach for `qwen3vl_32b_minimax_h3_nvfp4_awq` as the text encoder on a
12 GB card; the fp8 pruned one above is what fits. ~5.5 min per 73-frame clip at
608x896 on an RTX 3060.

---

## 2. Closing the loop: prefer real frames over a crossfade

A fade or cross-dissolve ghosts anything with directional motion (a walking
figure, drifting fog). `seamless.py` and `chain.py` do crossfade, so treat them
as the quick option and build the seam from real frames when quality matters,
using any compositor that gives hard cuts (Remotion, HyperFrames, ffmpeg, a
video editor):

1. **Bridge take (preferred).** Render a second MiniMax H3 clip with
   `first_frame` = the loop clip's LAST frame and `last_frame` = its FIRST
   frame. Join A + bridge with hard cuts in Remotion or HyperFrames. The
   motion carries the wrap; nothing dissolves.
2. **Pick a real loop point.** Search the clip for a late/early frame pair
   that is within ~1.5x the median frame step and cut there.
3. **Motion-compensated in-betweens** (optical flow: RIFE/FILM, ffmpeg
   `minterpolate` mci, or Resolve Speed Warp) only when the end-to-start gap
   is small; over a big gap they morph.

Compose and export the final loop in Remotion or HyperFrames (hard cuts,
`<Loop>`/looping composition), then measure the wrap as in section 3.

## 3. Measuring it — where every mistake was made

Three separate measurement bugs each rejected a loop that was fine. If a loop
"fails", suspect the measurement before re-rendering.

**Compare the seam to the clip's own frame-step distribution**, not to zero and
not to one sample. A seam is invisible when it is no bigger than the changes
around it.

1. **A single mid-clip sample is not "normal".** Motion is uneven; the middle is
   often the quiet part. Sampling one transition there passed a clip whose wrap
   was genuinely its largest jump. Read every interior step in one pass:
   `ffmpeg -vf "tblend=all_mode=difference,signalstats,metadata=print"`.
2. **A percentile alone is harsh on a tight distribution.** A seam 27% above the
   median can sit at the 98th percentile and still be smaller than the largest
   ordinary step. Judge on **both**: inside the observed range AND within ~1.5x
   the median.
3. **Measure a LOSSLESS encode, not the shipped file.** The output's frame 0 is
   an I-frame; its neighbours are P-frames, so differencing against it measures
   two *encodings* on top of two pictures.
   ```
   same loop, lossless : seam 2.99
   same loop, CRF 26   : seam 4.65    <- 1.66 of pure codec
   ```
   The tell: the number **does not move** when you trim the clip. Content-
   independent constant = artifact, not content.

`seamless.py` now builds a lossless twin purely to measure and ships the CRF 26
file. Its verdict line is the thing to read.

### `metadata=print` needs `-v info`

It logs at info level, so `-v error` silently removes the only output the
measurement reads and every number comes back `nan`.

---

## 4. When the wrap genuinely fails

`SEAM VISIBLE` after the lossless fix means directional motion — a figure
walking one way, fog accumulating — and **a longer `--fade` will not fix it**.
Verified: fades of 16, 24 and 30 all left the seam unchanged.

In order of preference:

1. **Shorten the loop** — but only after checking it is not the codec (above).
   Fewer source frames means less accumulated drift. Note this trades against
   the length rule: a shorter loop makes the fade more frequent, so prefer
   re-prompting over trimming when the clip is already under ~6s.
2. **Re-prompt for cyclical rather than one-way motion** — gusting wind,
   swirling petals, rippling water, pulsing light. Drop the walk and the
   building fog.
3. **Render longer** (243 frames) so there is more material to find a loop in,
   and a wider fade costs proportionally less of it.

---

## 5. Long AND fast: chain short takes

One render cannot give you both — see the length section above. `chain.py`
gets there by joining several short fast takes, crossfading each clip's tail
into the next clip's head, cyclically, so the wrap closes on consecutive
frames of one clip.

```
segment[i] = crossfade(clip[i-1] tail K, clip[i] head K) ++ clip[i][K : F-K]
output length = N x (F - K)
```

Three 73-frame takes at a 14-frame fade give ~7.4s. Measured, three takes of
the same image at seeds 7 / 11 / 23 / 41 all land within median 4.36-5.04, so
the takes are interchangeable in energy while differing in detail.

**This is also the real fix for a loop that feels tiring.** A 2.2s loop repeats
27 times a minute; what the eye catches is not the seam — that measures clean —
but the REPETITION of the same footage. Different takes remove the repetition;
the joins become dissolves between genuinely different content, which read as
transitions rather than stutters.

**All clips must be the same size.** `blend` fails with
`Error reinitializing filters!`, naming neither the cause nor the file.
`chain.py` checks up front and refuses rather than rescaling — mismatched sizes
mean a mistake about which renders belong together, and stretching one hides it.

Resolution costs nothing in motion: 544x800 takes measured the same as 608x896
ones, so render the chain small and spend the budget on takes.

## 6. Loops you did not make

Check them. An asset already in use with `loop` on it is not necessarily a
loop: a hand-and-flowers background carried into one project measured a
wrap of **29.95 against a median step of 4.65** — a 6.4x jump that had been
cutting visibly wherever it was used. `seamless.py` fixed it to 4.98 without a
re-render, because crossfading needs only the file.

## 7. Putting it on a page

```html
<video class="plate" id="plateLoop" autoplay loop muted playsinline
       preload="auto" poster="assets/still.webp" aria-hidden="true">
  <source src="assets/loop.mp4" type="video/mp4">
</video>
```

- **`autoplay` alone is not enough.** It is silently refused inside an iframe
  without an autoplay grant, and before the page has user activation. The
  rejection leaves the poster up, which looks like a still nobody animated.
  Retry on `canplay` and on the first `pointerdown`/`keydown`/`scroll`/
  `touchstart`, and pause under `prefers-reduced-motion`.
- **`poster` carries the sharp still** so a blocked autoplay or a slow
  connection shows the original art rather than black.
- **Remove any CSS drift/ken-burns from the same element.** Two competing
  motions, and the transform amplifies upscale softness.
- A 600px-wide clip covering a 1920 viewport is a ~3x upscale and visibly softer
  than the still. Acceptable under a scrim; for a crisp full-width header render
  at ~900px+ with 20 steps and `res_multistep`.

Budget: ~500 KB per 2.2s clip at CRF 26.
