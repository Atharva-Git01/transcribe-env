"""Stage: convert. Input copy -> per-track mono WAVs (48 kHz for denoise, 16 kHz for recognition).

channels = "downmix": one track (duplicated mono and real mono always end here — never fake speakers, E01)
channels = "split":   one track per channel, ONLY if the passport says true_stereo; otherwise downmix + log why
"""
from __future__ import annotations

from ..audio import ffmpeg
from ..context import Ctx, Track


def _filters(params: dict) -> str:
    chain = []
    if params.get("highpass_hz"):
        chain.append(f"highpass=f={int(params['highpass_hz'])}")
    if params.get("loudnorm_target_lufs") is not None:
        chain.append(f"loudnorm=I={float(params['loudnorm_target_lufs'])}:TP=-2:LRA=11")
    return ",".join(chain)


def run(ctx: Ctx, params: dict) -> None:
    layout = ctx.passport["channels"]["layout_class"]
    mode = params.get("channels", "downmix")
    if mode == "split" and layout != "true_stereo":
        ctx.log.add("convert", "used downmix instead of split",
                    f"Recipe asked for split but the passport layout is '{layout}'; splitting would invent speakers (E01).")
        mode = "downmix"
    n_tracks = 2 if mode == "split" else 1
    filt = _filters(params)
    sr16 = int(params.get("sample_rate", 16000))
    for i in range(n_tracks):
        label = f"SPK{i + 1}" if n_tracks > 1 else "SPK?"
        tr = Track(index=i, label=label)
        pan = f"pan=mono|c0=c{i}" if n_tracks > 1 else "pan=mono|c0=0.5*c0+0.5*c1" if ctx.passport["channels"]["count"] >= 2 else ""
        chain = ",".join(x for x in (pan, filt) if x)
        tr.wav48 = ctx.work / f"track{i + 1}_48k.wav"
        args = ["-i", str(ctx.input_file), "-map", "0:a:0"]
        if chain:
            args += ["-af", chain]
        ffmpeg(args + ["-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(tr.wav48)])
        tr.wav16 = ctx.work / f"track{i + 1}_16k.wav"
        ffmpeg(["-i", str(tr.wav48), "-ac", "1", "-ar", str(sr16), "-c:a", "pcm_s16le", str(tr.wav16)])
        ctx.custody.add(tr.wav16, "derived", "convert")
        ctx.tracks.append(tr)
    ctx.log.add("convert", f"made {n_tracks} mono track(s)",
                "Every engine receives clean 16 kHz mono audio; loudness normalised so quiet calls are audible.",
                mode=mode, filters=filt or "none", tracks=[t.label for t in ctx.tracks])
