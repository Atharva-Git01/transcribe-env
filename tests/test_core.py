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

    def test_chunk_spans_cut_in_silence(self):
        from pipeline.stages.recognise import chunk_spans
        speech = [(0, 100), (110, 700), (705, 1300), (1310, 1320)]
        self.assertEqual(chunk_spans(speech, 600), [(0.0, 702.5), (702.5, None)])   # cut mid-silence 700..705
        self.assertEqual(chunk_spans(speech, 0), [(0.0, None)])                      # chunking off

    def test_shift_moves_words_too(self):
        from pipeline.stages.recognise import _shift
        segs = _shift([{"start": 1.0, "end": 2.0, "words": [{"start": 1.0, "end": 1.5}]}], 600.0)
        self.assertEqual((segs[0]["start"], segs[0]["words"][0]["end"]), (601.0, 601.5))

    def test_engine_choices(self):
        from pipeline.stages.recognise import engine_choices
        table = {"mr": ["large", "turbo"], "en": "turbo", "unknown": "medium"}
        self.assertEqual(engine_choices(table, "mr"), ["large", "turbo"])
        self.assertEqual(engine_choices(table, "en"), ["turbo"])
        self.assertEqual(engine_choices(table, "hi"), ["medium"])   # falls back to "unknown"


class TestSpeakerSplit(unittest.TestCase):
    def test_split_at_speaker_change_keeps_text(self):
        from pipeline.stages.polish import split_by_turns
        words = [{"start": float(i), "end": i + 1.0, "word": f" w{i}"} for i in range(4)]
        seg = {"start": 0.0, "end": 4.0, "text": "w0 w1 w2 w3", "words": words}
        turns = [{"start": 0.0, "end": 2.0, "speaker": "SPK1"}, {"start": 2.0, "end": 4.0, "speaker": "SPK2"}]
        parts = split_by_turns([seg], turns)
        self.assertEqual([p["text"] for p in parts], ["w0 w1", "w2 w3"])
        self.assertEqual(" ".join(p["text"] for p in parts), seg["text"])
        self.assertEqual(split_by_turns([seg], [])[0]["text"], seg["text"])   # no diarization -> unchanged


class TestCleanupGates(unittest.TestCase):
    def test_whole_suggestion_gates(self):
        from pipeline.stages.cleanup import gates
        self.assertTrue(gates("कलत नाहि मी तयार आहें", "कळत नाही मी तयार आहे", 0.5)[0])
        self.assertFalse(gates("मी उद्या येतो", "Here is the corrected text: मी उद्या येतो", 0.5)[0])
        self.assertFalse(gates("मी उद्या येतो", "", 0.5)[0])
        self.assertFalse(gates("मी उद्या 5 वाजता office ला येतो", "मी उद्या येतो", 0.5)[0])
        self.assertFalse(gates("मी 5 वाजता येतो", "मी 6 वाजता येतो", 0.5)[0])   # entity guard

    def test_word_gate_keeps_only_safe_edits(self):
        from pipeline.stages.cleanup import word_gate
        ok = lambda a, b: word_gate(a, b, "mr", 0.25, use_lm=False)[0]   # the rules alone
        self.assertEqual(ok("ठीक है", "ठीक आहे"), "ठीक आहे")                      # Hindi form -> Marathi
        self.assertEqual(ok("मी आज नहीं येणार", "मी आज नाही येणार"), "मी आज नाही येणार")
        self.assertEqual(ok("ते पुस्तका वाचतोय", "ते पुस्तकं विकतोय"), "ते पुस्तकं वाचतोय")  # different word kept
        self.assertEqual(ok("मी घरी जातो", "मी Gharी जातो"), "मी घरी जातो")              # no other script
        self.assertEqual(ok("दुकान में है का", "दुकान में है क्या"), "दुकान में है का")  # no new Hindi form
        self.assertEqual(ok("सांगतो मावशी", "सांगतोवशी"), "सांगतो मावशी")          # no merged/removed words
        self.assertEqual(ok("मी येतो", "मी उद्या येतो"), "मी येतो")                   # no added words

    def test_language_model_check(self):
        from pipeline import lm
        from pipeline.stages.cleanup import word_gate
        if not lm.installed("mr"):
            self.skipTest("Marathi language model not installed")
        # a Hindi -> Marathi form fix that makes the line more Marathi is kept
        self.assertEqual(word_gate("माझी गाडी खराब है", "माझी गाडी खराब आहे", "mr", 0.25)[0], "माझी गाडी खराब आहे")
        self.assertGreater(lm.load("mr").per_word("मी उद्या सकाळी बाजारात जाणार आहे"),
                           lm.load("mr").per_word("मैं कल सुबह बाजार जाने वाला हूं"))


class TestLidEvidence(unittest.TestCase):
    def test_distinct_markers(self):
        from pipeline.lid import text_evidence
        ev = text_evidence(["मी आता घरी आहे, तू कुठे आहेस? बोलो बोलो बोलो बोलो"])
        self.assertEqual(ev["hi_markers"], 1)          # a loop counts once
        self.assertGreaterEqual(ev["mr_markers"], 3)


class TestSegmentLanguage(unittest.TestCase):
    def test_english_needs_strong_evidence(self):
        from pipeline.stages.lid import segment_language
        pr = ["mr", "hi", "en"]
        self.assertEqual(segment_language({"mr": 0.01, "hi": 0.5, "en": 0.49}, "mr", pr, 0.8), "mr")  # E20 case
        self.assertEqual(segment_language({"mr": 0.01, "hi": 0.04, "en": 0.95}, "mr", pr, 0.8), "en")
        self.assertEqual(segment_language({}, "mr", pr, 0.8), "mr")
        self.assertEqual(segment_language({"mr": 0.1, "hi": 0.85, "en": 0.05}, "en", pr, 0.8), "mr")


class TestLanguageConfirm(unittest.TestCase):
    def _ask(self, answers):
        import builtins
        from pipeline import langcheck
        feed = iter(answers)
        orig_input, orig_out = builtins.input, langcheck.out
        builtins.input, langcheck.out = (lambda prompt="": next(feed)), (lambda *a, **k: None)
        try:
            return langcheck.ask({"lang": "mr", "name": "Marathi", "confidence": 0.62, "basis": "test"})
        finally:
            builtins.input, langcheck.out = orig_input, orig_out

    def test_enter_accepts_guess(self):
        self.assertEqual(self._ask([""]), "mr")

    def test_other_language_overrides(self):
        self.assertEqual(self._ask(["hi"]), "hi")
        self.assertEqual(self._ask(["xx", "EN"]), "en")   # invalid answer -> asked again; case-insensitive

    def test_lang_argument(self):
        from pipeline.common import TrxError
        from pipeline.langcheck import confirm
        self.assertEqual(confirm("unused", "hi")[0], "hi")
        self.assertIsNone(confirm("unused", "auto")[0])
        with self.assertRaises(TrxError):
            confirm("unused", "fr")


class TestResume(unittest.TestCase):
    def test_lock_and_unfinished_detection(self):
        import json
        import tempfile
        from pipeline.run import RunLock, find_unfinished
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            audio = runs / "a.wav"
            audio.write_bytes(b"RIFF-test")
            import hashlib
            sha = hashlib.sha256(b"RIFF-test").hexdigest()
            run = runs / "a-1"
            (run / "work").mkdir(parents=True)
            (run / "plan.json").write_text("{}", encoding="utf-8")
            (run / "work" / "run_state.json").write_text(json.dumps(
                {"state": "running", "input_sha256": sha, "overrides": {"lang": "mr"}}), encoding="utf-8")
            held = RunLock(run)
            self.assertTrue(held.acquire())
            self.assertIsNone(find_unfinished(str(audio), {}, runs))        # still running (lock held)
            held.release()
            self.assertEqual(find_unfinished(str(audio), {}, runs), run)    # process gone -> resumable
            self.assertEqual(find_unfinished(str(audio), {"lang": "mr"}, runs), run)
            self.assertIsNone(find_unfinished(str(audio), {"lang": "hi"}, runs))       # different options
            self.assertIsNone(find_unfinished(str(audio), {"engine": "x"}, runs))


class TestNumbersAsWords(unittest.TestCase):
    def test_marathi_and_hindi(self):
        from pipeline.numbers import digits_to_words, to_words
        self.assertEqual(to_words(2, "mr"), "दोन")
        self.assertEqual(to_words(125, "mr"), "एकशे पंचवीस")
        self.assertEqual(to_words(100, "mr"), "शंभर")
        self.assertEqual(to_words(150000, "hi"), "एक लाख पचास हज़ार")
        text, n, left = digits_to_words("आम्ही ३ पुस्तके, 25% सवलत आणि 4:15 वाजता", "mr")
        self.assertEqual(text, "आम्ही तीन पुस्तके, पंचवीस टक्के सवलत आणि 4:15 वाजता")
        self.assertEqual((n, left), (2, ["4:15"]))                 # times are left as written, and reported
        self.assertEqual(digits_to_words("no digits here", "mr"), ("no digits here", 0, []))


class TestTurnUnits(unittest.TestCase):
    def test_merge_and_split(self):
        from pipeline.stages.recognise import turn_units
        turns = [{"start": 0.0, "end": 2.0, "speaker": "SPK1"}, {"start": 2.2, "end": 4.0, "speaker": "SPK1"},
                 {"start": 4.5, "end": 5.0, "speaker": "SPK2"}]
        u = turn_units(turns, [(0.0, 4.0), (4.5, 5.0)])
        self.assertEqual([(x["start"], x["end"], x["speaker"]) for x in u], [(0.0, 4.0, "SPK1"), (4.5, 5.0, "SPK2")])
        long = turn_units([{"start": 0.0, "end": 70.0, "speaker": "SPK1"}], [(0.0, 25.0), (26.0, 50.0), (51.0, 70.0)])
        self.assertTrue(all(x["end"] - x["start"] <= 30.0 for x in long))   # long turns cut at speech boundaries
        self.assertEqual(long[0]["end"], 25.0)


class TestEvalPrivacy(unittest.TestCase):
    def test_results_carry_no_words(self):
        from pipeline.evaluate import turns_without_text
        rows = turns_without_text([{"turn": 1, "ref_speaker": "A", "ref": "मी उद्या येतो", "hyp": "मी उद्या येतो",
                                    "hyp_speakers": ["SPK1"], "cer": 0.0},
                                   {"turn": 2, "ref_speaker": "B", "ref": "हो", "hyp": "", "hyp_speakers": [], "cer": 1.0}])
        self.assertTrue(all("ref" not in r and "hyp" not in r for r in rows))
        self.assertEqual([r["missing"] for r in rows], [False, True])

    def test_owner_transcript_lines(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("make_fixture", HOME / "setup" / "make_fixture.py")
        mf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mf)
        segs, bad = mf.parse("# call 1\nमी: नमस्कार, कसे आहात?\n\nबाबा: ठीक आहे: आणि तू?\nno colon here\n")
        self.assertEqual(segs, [{"speaker": "मी", "text": "नमस्कार, कसे आहात?"},
                                {"speaker": "बाबा", "text": "ठीक आहे: आणि तू?"}])
        self.assertEqual(bad, [5])


class TestOfflineGuard(unittest.TestCase):
    def test_blocks_public_hosts(self):
        offline.install()
        with self.assertRaises(offline.NetworkBlocked):
            socket.create_connection(("example.com", 443), timeout=1)
        with self.assertRaises(offline.NetworkBlocked):
            socket.getaddrinfo("huggingface.co", 443)


if __name__ == "__main__":
    unittest.main()
