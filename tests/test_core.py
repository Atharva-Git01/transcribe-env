"""Unit tests for the deterministic parts of the environment (no models needed). Run: `trx test`."""
from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common import load_config, load_toml  # noqa: E402
from pipeline.guards import offline  # noqa: E402
from pipeline.metrics import cer, normalise, wer  # noqa: E402
from pipeline.recipes import all_recipe_ids, check_recipe  # noqa: E402
from pipeline.schema_check import validate  # noqa: E402
from pipeline.stages.select import check_rules, condition_true, select  # noqa: E402

HOME = Path(__file__).resolve().parents[1]


def passport(**over):
    p = {"container": {"duration_sec": 60.0}, "channels": {"layout_class": "mono"},
         "speakers": {"estimate": None}, "noise": {"snr_db_estimate": 20.0},
         "language": {"code_switch_risk": "low"}, "bandwidth": {"class": "wideband"},
         "levels": {"clipping_ratio": 0.0}}
    for k, v in over.items():
        a, b = k.split("__")
        p[a][b] = v
    return p


class TestSelection(unittest.TestCase):
    def test_rules_are_valid(self):
        self.assertEqual(check_rules(load_config("selection_rules")["rule"]), [])

    def test_null_field_never_matches(self):
        self.assertFalse(condition_true({"speakers": {"estimate": None}}, ["speakers.estimate", ">=", 3]))

    def test_routing(self):
        self.assertEqual(select(passport(container__duration_sec=4000))["recipe"], "long_file")
        self.assertEqual(select(passport(channels__layout_class="true_stereo"))["recipe"], "true_stereo_call")
        self.assertEqual(select(passport(speakers__estimate=4))["recipe"], "meeting_multi_speaker")
        self.assertEqual(select(passport(noise__snr_db_estimate=5.0))["recipe"], "noisy_field_audio")
        self.assertEqual(select(passport(language__code_switch_risk="high"))["recipe"], "code_switched")
        self.assertEqual(select(passport(bandwidth__class="narrowband"))["recipe"], "phone_call_mono")
        self.assertEqual(select(passport(noise__snr_db_estimate=35.0))["recipe"], "clean_studio")
        self.assertEqual(select(passport())["recipe"], "_fallback")

    def test_duplicated_mono_is_not_true_stereo(self):
        self.assertNotEqual(select(passport(channels__layout_class="duplicated_mono"))["recipe"], "true_stereo_call")


class TestRecipesAndConfig(unittest.TestCase):
    def test_every_recipe_valid(self):
        for rid in all_recipe_ids():
            with self.subTest(recipe=rid):
                self.assertEqual(check_recipe(load_toml(HOME / "recipes" / f"{rid}.toml"), rid), [])

    def test_forensic_invariants_in_recipes(self):
        for rid in all_recipe_ids():
            fw = load_toml(HOME / "recipes" / f"{rid}.toml")["recognise"]["faster_whisper"]
            with self.subTest(recipe=rid):
                self.assertFalse(fw["condition_on_previous_text"])   # E03
                self.assertEqual(fw["no_repeat_ngram_size"], 0)      # E02

    def test_exit_codes_unique(self):
        codes = load_config("exit_codes")
        self.assertEqual(len({v["code"] for v in codes.values()}), len(codes))


class TestSchema(unittest.TestCase):
    def test_validator_catches_errors(self):
        errs = validate([{"seq": 0, "utc": "x", "stage": "s", "action": "", "reason": "r", "data": {}}], "decisions")
        self.assertTrue(any("minimum" in e for e in errs))
        self.assertTrue(any("shorter" in e for e in errs))


class TestMetrics(unittest.TestCase):
    def test_cer_wer(self):
        self.assertEqual(cer("हा हा।", "हा हा"), 0.0)
        self.assertAlmostEqual(cer("abcd", "abce"), 0.25)
        self.assertAlmostEqual(wer("a b c d", "a b x d"), 0.25)
        self.assertEqual(normalise("[inaudible 00:01-00:02] Hello,  World"), "hello world")


class TestPolish(unittest.TestCase):
    def test_loops(self):
        from pipeline.stages.polish import CHAR_RUN, FOREIGN_LETTER, find_loop
        doubled = {"हा", "हो"}
        self.assertIsNone(find_loop(["हा", "हा", "ठीक"], doubled))            # real doubled word
        self.assertIsNotNone(find_loop(["हां", "बोलो"] * 4, doubled))        # n-gram loop
        self.assertIsNotNone(find_loop(["अरे"] * 3, doubled))                # same word 3x
        self.assertIsNotNone(CHAR_RUN.search("বববববববববব"))
        self.assertTrue(FOREIGN_LETTER.findall("ববব"))
        self.assertFalse(FOREIGN_LETTER.findall("हॅलो, ठीक आहे. Okay 2"))


class TestSegmentation(unittest.TestCase):
    def test_speech_groups(self):
        from pipeline.stages.recognise import speech_groups
        speech = [(0.0, 5.0), (5.5, 12.0), (12.3, 22.0), (25.0, 26.0)]
        self.assertEqual(speech_groups(speech, 20.0, 0.8), [(0.0, 12.0), (12.3, 22.0), (25.0, 26.0)])
        self.assertEqual(speech_groups([], 20.0, 0.8), [])

    def test_split_keeps_text(self):
        from pipeline.stages.recognise import split_segment
        words = [{"start": i * 1.0, "end": i * 1.0 + 0.8, "word": f" w{i}" + ("." if i == 4 else ""), "probability": 0.9}
                 for i in range(20)]
        seg = {"start": 0.0, "end": 19.8, "text": "".join(w["word"] for w in words).strip(), "words": words,
               "avg_logprob": -0.5, "compression_ratio": 1.2}
        parts = split_segment(seg, 8.0)
        self.assertGreater(len(parts), 1)
        self.assertEqual(" ".join(p["text"] for p in parts), seg["text"])          # nothing lost or added
        self.assertTrue(all(p["end"] - p["start"] <= 8.0 for p in parts))
        self.assertTrue(parts[0]["text"].endswith("w4."))                          # cut at the sentence end

    def test_engine_choices(self):
        from pipeline.stages.recognise import engine_choices
        table = {"mr": ["large", "turbo"], "en": "turbo", "unknown": "medium"}
        self.assertEqual(engine_choices(table, "mr"), ["large", "turbo"])
        self.assertEqual(engine_choices(table, "en"), ["turbo"])
        self.assertEqual(engine_choices(table, "hi"), ["medium"])   # falls back to "unknown"


class TestCleanupGates(unittest.TestCase):
    def test_gates(self):
        from pipeline.stages.cleanup import gates
        self.assertTrue(gates("कलत नाहि मी तयार आहें", "कळत नाही मी तयार आहे", 0.4, 0.5)[0])
        self.assertFalse(gates("मी उद्या येतो", "Here is the corrected text: मी उद्या येतो", 0.4, 0.5)[0])
        self.assertFalse(gates("मी उद्या येतो", "", 0.4, 0.5)[0])
        self.assertFalse(gates("मी उद्या 5 वाजता office ला येतो", "मी उद्या येतो", 0.4, 0.5)[0])
        self.assertFalse(gates("मी 5 वाजता येतो", "मी 6 वाजता येतो", 0.4, 0.5)[0])   # entity guard


class TestLidEvidence(unittest.TestCase):
    def test_distinct_markers(self):
        from pipeline.lid import text_evidence
        ev = text_evidence(["मी आता घरी आहे, तू कुठे आहेस? बोलो बोलो बोलो बोलो"])
        self.assertEqual(ev["hi_markers"], 1)          # a loop counts once
        self.assertGreaterEqual(ev["mr_markers"], 3)


class TestOfflineGuard(unittest.TestCase):
    def test_blocks_public_hosts(self):
        offline.install()
        with self.assertRaises(offline.NetworkBlocked):
            socket.create_connection(("example.com", 443), timeout=1)
        with self.assertRaises(offline.NetworkBlocked):
            socket.getaddrinfo("huggingface.co", 443)


if __name__ == "__main__":
    unittest.main()
