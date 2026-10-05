"""Ambient motion from a still, via MiniMax H3 in ComfyUI.

Run: python make-loop.py <image> "<prompt>" [--width W --height H --length N
                                             --steps N --physics 0-100 --zerog 0-100 --cine 0-100]

This renders MOTION. It does not close the loop — `seamless.py` does that
afterwards by crossfading the tail over the head.

That split is deliberate and was learned the hard way. The obvious route is to
hand `MiniMaxH3ImageToVideo` the same still as `first_frame` AND `last_frame`
and get a loop for free. Measured, that does not work twice over: the ends are
CONDITIONING rather than pixels, so the model paints its own version of each
(source still vs frame 0 differed by 19.18) and the seam came out as large as
the motion; and asking it to finish where it started is an instruction to stay
put, which is most of why the first attempts had no wind and no walk in them.

So: let the shot move freely, then close it in post.

`length` sits on the model's 17k+5 grid (5, 22, 39, 56, 73, ...). The node
snaps up silently otherwise, which quietly lengthens the clip you asked for.
"""
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import os
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

UNET = "minimax_h3_fl_or_ref2va_w4a8_pruned_a.safetensors"
# Speed, then physics. The turbo LoRA buys the step count back; the wushu one
# is a spatial-physics adapter and is what puts weight behind cloth, water and
# a walking figure instead of the whole frame breathing in place.
LORA = "minimax_h3_fl2v_turbo_4step_v1.2_768p.safetensors"
PHYSICS = r"Minimax H3\wushu_spatial_physics_v2_1000_pruned.safetensors"
# Zero Gravity 101 (CivitAI 2952734, base MiniMax H3). Everything floats,
# drifts and tumbles instead of falling. Stacked on top rather than replacing
# the physics adapter: that one decides whether motion has weight, this one
# decides which way "down" is, and they are not the same question.
ZEROG = r"Minimax H3\zeroGravity101_v1.safetensors"
# Cinematic look (CivitAI 2908686, base MiniMax H3). Trigger word is DY, which
# has to appear in the PROMPT or the adapter does nothing. Author's note: 0.7
# normally, 0.5 for high-motion shots -- and everything this script is used for
# is a high-motion shot, so 0.5 is the sane default here.
CINE = r"Minimax H3\minimax_h3_cinematic_dy_v01.safetensors"
CLIP = "minimaxH3Fl2vaPruned_fl2vaPrunedFp8Scaled_txt.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"


def upload(path: Path) -> str:
    """Put the still in ComfyUI's input folder; LoadImage reads by name."""
    boundary = uuid.uuid4().hex
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
        f'filename="{path.name}"\r\nContent-Type: image/png\r\n\r\n'.encode(),
        path.read_bytes(),
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{COMFY}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]


def graph(image_name, prompt, width, height, length, seed, prefix,
          steps=8, shift=6.0, physics=0.8, zerog=0.0, cine=0.0, hold_end=False):
    """The i2v graph.

    Four settings here were wrong in the first pass and every one of them
    damped motion, which is why the first loops drifted instead of moving:

      steps 4      -> 8. The turbo LoRA is advertised as 4-step, but both
                     published i2v graphs on this machine run 8 with euler (or
                     20 with res_multistep). Four is the floor for a coherent
                     picture, not the number that animates it.
      shift 12.0   -> 6.0. 12 is the node default; every reference workflow
                     here passes 6, which spends more of the schedule where
                     movement is decided.
      last_frame   -> dropped. Pinning the end to the same still as the start
                     tells the model to come back to where it began, so it
                     buys the loop by refusing to go anywhere. The loop is
                     closed afterwards by crossfade instead (seamless.py).
      no physics   -> the wushu spatial-physics LoRA, stacked on the turbo one.

    `hold_end` restores the old behaviour for the rare shot that genuinely
    wants to return to its opening pose.
    """
    wf = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": LORA, "strength_model": 1.0}},
        "phys": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["lora", 0], "lora_name": PHYSICS,
                            "strength_model": physics}},
        "shift": {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["phys", 0], "shift_video": shift, "shift_audio": 3.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "i2v": {"class_type": "MiniMaxH3ImageToVideo",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                           "width": width, "height": height, "length": length,
                           "first_frame": ["img", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["i2v", 0]}},
        "ks": {"class_type": "KSampler",
               "inputs": {"model": ["shift", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                          "positive": ["i2v", 0], "negative": ["neg", 0],
                          "latent_image": ["i2v", 1]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "PixaromaSaveMp4",
                 "inputs": {"video_frames": ["dec", 0], "fps": 24.0,
                            "filename_prefix": prefix, "save_mode": "save",
                            "trim_to_audio": False}},
    }
    if hold_end:
        wf["i2v"]["inputs"]["last_frame"] = ["img", 0]
    if physics <= 0:
        wf["shift"]["inputs"]["model"] = ["lora", 0]
        del wf["phys"]
    # Appended to whatever the chain currently ends in, so this stays correct
    # whether or not the physics LoRA above survived.
    if zerog > 0:
        wf["zerog"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": wf["shift"]["inputs"]["model"],
                                  "lora_name": ZEROG, "strength_model": zerog}}
        wf["shift"]["inputs"]["model"] = ["zerog", 0]
    if cine > 0:
        wf["cine"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": wf["shift"]["inputs"]["model"],
                                 "lora_name": CINE, "strength_model": cine}}
        wf["shift"]["inputs"]["model"] = ["cine", 0]
    return wf


def post(path, payload):
    req = urllib.request.Request(
        f"{COMFY}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    args = sys.argv[1:]
    img = Path(args[0])
    prompt = args[1]
    opts = {"width": 544, "height": 800, "length": 73, "seed": 7,
            "steps": 8, "physics": 80, "zerog": 0, "cine": 0}
    for i, a in enumerate(args):
        for k in opts:
            if a == f"--{k}":
                opts[k] = int(args[i + 1])

    # 17k+5 grid. The node snaps up on its own, which silently lengthens the
    # clip; snapping here means the number reported is the number rendered.
    if (opts["length"] - 5) % 17:
        opts["length"] = 5 + 17 * round((opts["length"] - 5) / 17)

    name = upload(img)
    prefix = f"loop/{img.stem}"
    wf = graph(name, prompt, opts["width"], opts["height"], opts["length"],
               opts["seed"], prefix, steps=opts["steps"],
               physics=opts["physics"] / 100.0, zerog=opts["zerog"] / 100.0,
               cine=opts["cine"] / 100.0)

    print(f"{img.name} -> {opts['width']}x{opts['height']}, {opts['length']}f "
          f"(~{opts['length']/24:.1f}s), {opts['steps']} steps, "
          f"physics {opts['physics']/100:.2f}, zero-g {opts['zerog']/100:.2f}, "
          f"cine {opts['cine']/100:.2f}, shift 6.0, free end", flush=True)
    r = post("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    pid = r["prompt_id"]
    print("queued", pid, flush=True)

    start = time.time()
    while time.time() - start < 3600:
        time.sleep(5)
        try:
            with urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=30) as h:
                hist = json.load(h)
        except Exception:
            continue
        e = hist.get(pid)
        if not e:
            continue
        if e.get("status", {}).get("status_str") == "error":
            print("FAILED after %.0fs" % (time.time() - start), flush=True)
            for m in e.get("status", {}).get("messages", []):
                print(" ", json.dumps(m)[:600], flush=True)
            return 1
        if e.get("status", {}).get("completed"):
            print("done in %.0fs" % (time.time() - start), flush=True)
            print(json.dumps(e.get("outputs", {}))[:800], flush=True)
            return 0
    print("timed out", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
