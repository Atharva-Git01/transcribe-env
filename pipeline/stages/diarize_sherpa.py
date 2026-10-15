"""Diarization with sherpa-onnx (phase 4): pyannote segmentation 3.0 + WeSpeaker ResNet34 embeddings, clustered.
No torch, no account, fully local. Produces speaker turns; polish assigns each segment its majority speaker.
"""
from __future__ import annotations

from ..audio import read_wav
from ..common import TRX_HOME, write_json
from ..context import Ctx

MODELS = TRX_HOME / "toolchain" / "models" / "sherpa-diarization"
SEG_MODEL = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS / "wespeaker_en_voxceleb_resnet34_LM.onnx"

try:
    import sherpa_onnx  # noqa: F401
    AVAILABLE = SEG_MODEL.is_file() and EMB_MODEL.is_file()
except ImportError:
    AVAILABLE = False


def diarize(audio, num_speakers: int = 0, threshold: float = 0.5, min_on: float = 0.3, min_off: float = 0.5):
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            num_threads=4),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers if num_speakers > 0 else -1,
                                                    threshold=threshold),
        min_duration_on=min_on, min_duration_off=min_off)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    result = sd.process(audio).sort_by_start_time()
    return [{"start": round(r.start, 3), "end": round(r.end, 3), "speaker": int(r.speaker)} for r in result]


def run(ctx: Ctx, params: dict) -> None:
    tr = ctx.tracks[0]
    audio, _ = read_wav(tr.wav16)
    raw = diarize(audio, int(params.get("num_speakers", 0)), float(params.get("cluster_threshold", 0.5)),
                  float(params.get("min_turn_s", 0.3)))
    order = []
    for t in raw:  # SPK1, SPK2 … in order of first appearance
        if t["speaker"] not in order:
            order.append(t["speaker"])
    ctx.turns = [{"start": t["start"], "end": t["end"], "speaker": f"SPK{order.index(t['speaker']) + 1}"} for t in raw]
    path = ctx.work / "diarization.json"
    write_json(path, {"engine": "sherpa-onnx pyannote-3.0 + wespeaker-resnet34", "turns": ctx.turns})
    ctx.custody.add(path, "derived", "diarize")
    ctx.metrics["diarization"] = "sherpa_onnx"
    ctx.metrics["speakers_found"] = len(order)
    ctx.log.add("diarize", f"found {len(order)} speaker(s) in {len(ctx.turns)} turns",
                "Speakers separated by voice (local sherpa-onnx); labels are SPK1.. in order of appearance.")
