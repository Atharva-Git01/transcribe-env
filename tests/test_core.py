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

    def test_uncertain_lines(self):
        # E40: the decoder's own confidence points at the bad lines; acknowledgements are left alone
        from pipeline.stages.polish import uncertain
        thr = load_config("thresholds")["verify"]
        line = {"text": "उद्या बाजारात भाजी घेऊ", "token_logprob_mean": -0.05}
        self.assertIsNone(uncertain(line, "mr", thr))                                 # sure of itself
        self.assertTrue(uncertain(dict(line, token_logprob_mean=-0.6), "mr", thr).startswith("uncertain"))
        self.assertIsNone(uncertain({"text": "हं हं", "token_logprob_mean": -0.9}, "mr", thr))
        self.assertIsNone(uncertain({"text": "उद्या भेटू", "avg_logprob": None}, "mr", thr))   # other engines


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

    def test_short_reply_after_a_pause_keeps_its_speaker(self):
        # E39: a short interjection between two pauses was smoothed into the other party's turn
        import numpy as np
        from pipeline.stages.diarize_embed import penalties, viterbi, windows
        speech = [(0.0, 5.0), (5.5, 6.3), (6.4, 10.0)]          # long turn, pause, 0.8 s reply, long turn
        wins = windows(speech, 1.0, 0.25)
        reply = np.array([5.5 <= s < 6.3 for s, _ in wins])
        S = np.where(reply[:, None], [0.25, 0.40], [0.7, 0.2])  # the reply sounds more like speaker 1 (by 0.15)
        self.assertEqual(viterbi(S, 0.1).tolist(), [0] * len(wins))                 # one cost everywhere: absorbed
        pen = penalties(wins, speech, 0.1, 0.02, 0.2)
        self.assertEqual(sorted(set(pen.tolist())), [0.02, 0.1])                     # cheap only after the pause
        self.assertEqual(viterbi(S, pen).tolist(), reply.astype(int).tolist())


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
        self.assertEqual([r["ref_speaker"] for r in rows], ["A", "B"])          # names of people stay here too
        self.assertEqual(turns_without_text(rows), rows)                         # idempotent

    def test_no_private_words_in_pushable_files(self):
        from pipeline import privacy
        self.assertEqual(privacy.phrases(["मी उद्या सकाळी येतो."]), {"मी उद्या सकाळी", "उद्या सकाळी येतो"})
        if not privacy.private_texts():
            self.skipTest("no private fixtures or runs on this PC")
        self.assertEqual(privacy.leaks(), [], "words of a private recording in a file git would push (E29)")
        self.assertEqual([(f, n) for f, n, _ in privacy.rare_word_leaks()], [],
                         "a rare word of a private recording in prose or results (E29)")

    def test_eval_runs_are_private_unless_their_fixture_is_open(self):
        # E39: after a fixture was renamed, its earlier eval run no longer counted as private
        import json
        import tempfile
        from pipeline import privacy
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fx = tmp / "evals" / "fixtures" / "open_read"
            fx.mkdir(parents=True)
            (fx / "truth.json").write_text(json.dumps({"source": "FLEURS, CC-BY-4.0", "segments": []}))
            for name, text in (("eval_open_read-20260101-000000", "खुला मजकूर"),
                               ("eval_renamed_call-20260101-000000", "खासगी मजकूर")):
                out = tmp / "evals" / "runs" / name / "out"
                out.mkdir(parents=True)
                (out / "transcript.json").write_text(json.dumps({"segments": [{"text": text}]}), encoding="utf-8")
            self.assertEqual(privacy.private_texts(tmp), ["खासगी मजकूर"])

    def test_library_words_are_generic(self):
        # E39: a written "hmm" in code matched the owner's call; words the library holds on purpose are not private
        from pipeline import privacy
        self.assertIn("हम्म", privacy.generic_words())

    def test_owner_transcript_lines(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("make_fixture", HOME / "setup" / "make_fixture.py")
        mf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mf)
        segs, bad = mf.parse("# call 1\nमी: नमस्कार, कसे आहात?\n\nबाबा: ठीक आहे: आणि तू?\nno colon here\n")
        self.assertEqual(segs, [{"speaker": "मी", "text": "नमस्कार, कसे आहात?"},
                                {"speaker": "बाबा", "text": "ठीक आहे: आणि तू?"}])
        self.assertEqual(bad, [5])


class TestCleanupDecision(unittest.TestCase):
    def test_engine_filter_and_missing_model(self):
        from pipeline.stages.polish import cleanup_decision
        llm = {"enabled": True, "engines": ["faster_whisper"]}
        never = lambda: self.fail("Gemma must not be probed when the cleanup does not apply")  # noqa: E731
        self.assertEqual(cleanup_decision(llm, ["indic_transcribe_core"], never), "engine")       # E26
        self.assertEqual(cleanup_decision(llm, ["faster_whisper_large_v3_turbo"], lambda: True), "run")
        self.assertEqual(cleanup_decision(llm, ["faster_whisper_large_v3_turbo"], lambda: False), "missing")  # USB out
        self.assertEqual(cleanup_decision(dict(llm, enabled=False), ["faster_whisper_large_v3"], never), "off")


class TestMemoryCheck(unittest.TestCase):
    def test_enough(self):
        from pipeline import resources
        self.assertTrue(resources.enough(6.5, None))        # unknown headroom never blocks a run
        self.assertTrue(resources.enough(6.5, 7.0))
        self.assertFalse(resources.enough(6.5, 1.7))         # the 2026-10-23 crash (E31)
        self.assertTrue(resources.enough(0.0, 0.1))          # engines without a need are never stopped
        h = resources.headroom_gb()
        self.assertTrue(h is None or h >= 0)

    def test_every_family_need_is_a_known_engine(self):
        from pipeline.stages.recognise import FAMILY_TABLE
        need = load_config("defaults")["runtime"].get("memory_gb", {})
        self.assertTrue({k.removesuffix("_gpu") for k in need} <= set(FAMILY_TABLE))


class TestEvalRun(unittest.TestCase):
    def test_eval_keeps_the_language_override(self):
        # E30: the second fixture of an eval was transcribed in the FIRST fixture's language
        import json
        import tempfile
        from unittest import mock
        from pipeline import evaluate
        seen = []

        def fake_transcribe(path, forced_recipe=None, case=None, engine=None, lang=None, runs_dir=None):
            seen.append(lang)
            run = runs_dir / f"{case}-{len(seen)}"
            (run / "out").mkdir(parents=True)
            (run / "report.json").write_text(json.dumps({"metrics": {"engines": ["x"], "language": "mr"}}))
            (run / "out" / "transcript.json").write_text(json.dumps({"segments": [{"text": "मी येतो", "speaker": "SPK1"}]}))
            return "OK", {"run_id": run.name, "verdict": "PASS", "recipe": "r"}

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for fid, lang in (("a_en", "en"), ("b_mr", "mr")):
                d = tmp / "fixtures" / fid
                d.mkdir(parents=True)
                (d / "audio.wav").write_bytes(b"")
                (d / "truth.json").write_text(json.dumps({"schema": "trx.truth/1", "id": fid, "audio": "audio.wav",
                                                          "language": lang, "reference_text": "मी येतो",
                                                          "segments": [{"speaker": "A", "text": "मी येतो"}]}))
            with mock.patch.multiple(evaluate, TRX_HOME=tmp, FIXTURES=tmp / "fixtures", EVAL_RUNS=tmp / "runs",
                                     RESULTS=tmp / "results", COMPARE=tmp / "compare", BASELINE=tmp / "baseline.json"), \
                    mock.patch("pipeline.run.transcribe", fake_transcribe):
                res = evaluate.run_eval()
        self.assertEqual(seen, [None, None])                       # no override -> each run measures its own
        self.assertIsNone(res["overrides"]["lang"])

    def test_scored_until_ignores_speech_after_the_transcript(self):
        # E39: the owner's transcript stopped at 12:00 of a 20:35 call; the untranscribed rest counted as errors
        import json
        import tempfile
        from unittest import mock
        from pipeline import evaluate
        segs = [{"start": 0.0, "end": 2.0, "text": "मी येतो", "speaker": "SPK1"},
                {"start": 6.0, "end": 9.0, "text": "उद्या सकाळी परत बोलू", "speaker": "SPK2"}]

        def fake_transcribe(path, forced_recipe=None, case=None, engine=None, lang=None, runs_dir=None):
            run = runs_dir / f"{case}-1"
            (run / "out").mkdir(parents=True)
            (run / "report.json").write_text(json.dumps({"metrics": {"engines": ["x"], "language": "mr"}}))
            (run / "out" / "transcript.json").write_text(json.dumps({"segments": segs}))
            return "OK", {"run_id": run.name, "verdict": "PASS", "recipe": "r"}

        rows = {}
        for until in (None, 5.0):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                d = tmp / "fixtures" / "part"
                d.mkdir(parents=True)
                (d / "audio.wav").write_bytes(b"")
                truth = {"schema": "trx.truth/1", "id": "part", "audio": "audio.wav", "language": "mr",
                         "reference_text": "मी येतो", "segments": [{"speaker": "A", "text": "मी येतो"}]}
                if until:
                    truth["scored_until_sec"] = until
                (d / "truth.json").write_text(json.dumps(truth))
                with mock.patch.multiple(evaluate, TRX_HOME=tmp, FIXTURES=tmp / "fixtures", EVAL_RUNS=tmp / "runs",
                                         RESULTS=tmp / "results", COMPARE=tmp / "compare",
                                         BASELINE=tmp / "baseline.json"), \
                        mock.patch("pipeline.run.transcribe", fake_transcribe):
                    rows[until] = evaluate.run_eval()["rows"][0]
        self.assertGreater(rows[None]["cer"], 1.0)                  # the later line counted as inserted errors
        self.assertEqual(rows[5.0]["cer"], 0.0)                     # only lines starting before 5 s are scored
        self.assertEqual(rows[5.0]["scored_until_sec"], 5.0)
        mon = rows[5.0]["monitor"]                                   # the failure monitor rides along (E39)
        self.assertEqual((mon["turn_outcomes"], mon["negation_errors"], mon["bad_lines"]), ({"correct": 1}, 0, 0))
        self.assertEqual(rows[5.0]["turn_rows"][0]["what_happened"], "correct")

    def test_truth_written_as_spoken_is_judged_by_content_cer(self):
        # the owner writes transcripts as spoken (2026-10-23); standard spelling in the output is no regression
        import json
        import tempfile
        from unittest import mock
        from pipeline import evaluate

        def fake_transcribe(path, forced_recipe=None, case=None, engine=None, lang=None, runs_dir=None):
            run = runs_dir / f"{case}-1"
            (run / "out").mkdir(parents=True)
            (run / "report.json").write_text(json.dumps({"metrics": {"engines": ["x"], "language": "mr"}}))
            (run / "out" / "transcript.json").write_text(json.dumps({"segments": [
                {"start": 0.0, "end": 3.0, "text": "ती शाळेत आहे", "speaker": "SPK1"}]}))
            return "OK", {"run_id": run.name, "verdict": "PASS", "recipe": "r"}

        found = {}
        for spelling in ("as_spoken", "standard"):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                d = tmp / "fixtures" / "talk"
                d.mkdir(parents=True)
                (d / "audio.wav").write_bytes(b"")
                (d / "truth.json").write_text(json.dumps({
                    "schema": "trx.truth/1", "id": "talk", "audio": "audio.wav", "language": "mr",
                    "spelling": spelling, "reference_text": "ती शाळेतये",
                    "segments": [{"speaker": "A", "text": "ती शाळेतये"}]}))
                (tmp / "baseline.json").write_text(json.dumps({"scores": {"talk|r|x|mr": {"cer": 0.0,
                                                                                         "cer_content": 0.0}}}))
                with mock.patch.multiple(evaluate, TRX_HOME=tmp, FIXTURES=tmp / "fixtures", EVAL_RUNS=tmp / "runs",
                                         RESULTS=tmp / "results", COMPARE=tmp / "compare",
                                         BASELINE=tmp / "baseline.json"), \
                        mock.patch("pipeline.run.transcribe", fake_transcribe):
                    found[spelling] = evaluate.run_eval()["regressions"]
        self.assertEqual(found["as_spoken"], [])                    # only the spelling differs: content CER 0
        self.assertEqual(len(found["standard"]), 1)                 # a standard truth still counts every letter


class TestRepetitionFlags(unittest.TestCase):
    NATURAL = ("आज सकाळी आम्ही बाजारात गेलो होतो. तिथे ताज्या भाज्या आणि फळे स्वस्त मिळाली. नंतर आईने घरी "
               "वरणभात केला आणि सगळ्यांनी एकत्र जेवण केले. दुपारी पाऊस आला म्हणून मुले खेळायला बाहेर गेली नाहीत. "
               "संध्याकाळी काकांचा फोन आला की ते पुढच्या आठवड्यात पुण्याला येणार आहेत. त्यांच्यासाठी पाहुण्यांची "
               "खोली साफ करायची आहे आणि नवीन चादरी आणायच्या आहेत. रात्री उशिरा आम्ही जुने फोटो पाहत बसलो.")

    def test_repetitive_means_repeated_phrases_not_long_text(self):
        from pipeline.stages.polish import repetition
        thr = load_config("thresholds")["verify"]
        self.assertGreater(len(self.NATURAL), 350)
        self.assertIsNone(repetition(self.NATURAL, 2.7, thr))       # long natural line: ratio 2.7 is normal (E37)
        loop = "मी उद्या बाजारात जाणार आहे " + "तिथे भाजी घेणार आहे " * 4
        self.assertIsNotNone(repetition(loop, None, thr))            # a phrase again and again
        self.assertIsNotNone(repetition("abc" * 60, 6.0, thr))       # repetition without words: ratio backstop

    def test_backchannel_turn_is_not_a_loop(self):
        from pipeline.stages.polish import find_loop
        doubled = {"हं", "हां"}
        self.assertIsNone(find_loop(["हं"] * 4, doubled))             # a whole turn of acknowledgement
        self.assertIsNotNone(find_loop(["हं"] * 6, doubled))
        self.assertIsNotNone(find_loop(["मग"] + ["हं"] * 4, doubled))  # inside longer text the old limit stays
        self.assertIsNotNone(find_loop(["काय"] * 3, doubled))          # not a listed backchannel


class TestEngineCache(unittest.TestCase):
    def test_one_kept_model_loaded_once(self):
        from unittest import mock
        from pipeline.engines import cache
        made = []

        class Fake:
            family = "fake"

            def __init__(self, eid):
                self.id, self.keep_loaded, self.loaded = eid, eid.startswith("keep"), False

            def load(self, device, params):
                self.loaded = True
                return {"engine": self.id, "device": device, "compute_type": "x"}

            def unload(self):
                self.loaded = False

            def configure(self, params):
                self.params = params

        def fake_make(eid):
            made.append(Fake(eid))
            return made[-1]

        with mock.patch("pipeline.engines.faster_whisper_engine.make", fake_make):
            cache.release()
            a, _ = cache.get("keep_a", "cpu", {})
            cache.done("keep_a", a)
            self.assertTrue(a.loaded)                                     # kept for the next stage (E34)
            b, info = cache.get("keep_a", "cpu", {"piece_max_s": 20.0})
            self.assertIs(a, b)
            self.assertTrue(info["reused"])
            self.assertEqual(len(made), 1)                                # loaded once per run
            self.assertEqual(b.params, {"piece_max_s": 20.0})
            c, _ = cache.get("other", "cpu", {})                          # another engine: the held one goes first (E06)
            self.assertFalse(a.loaded)
            cache.done("other", c)
            self.assertFalse(c.loaded)                                    # not kept: unloaded after use
            self.assertEqual(cache.held(), [])

    def test_trusted_language_id(self):
        from pipeline.lid import acoustic_decision
        lid = {"top3": [{"lang": "mr", "prob": 1.0}, {"lang": "hi", "prob": 0.0}],
               "per_window": [{"top": "mr"}] * 5 + [{"top": "en"}]}
        d = acoustic_decision(lid, "indic_transcribe_core")
        self.assertEqual((d["decision"], d["confidence"], d["code_switch_risk"]), ("mr", 1.0, "low"))
        weak = dict(lid, top3=[{"lang": "hi", "prob": 0.55}, {"lang": "mr", "prob": 0.45}])
        self.assertEqual(acoustic_decision(weak, "indic_transcribe_core")["decision"], "mr")   # priority language


class TestSpokenForms(unittest.TestCase):
    # PLAN_v0.9: the spoken-variant list and the spoken-form metric (invented sentences, never words of a case)
    def test_variant_list_rows(self):
        import tomllib
        rows = tomllib.loads((HOME / "library" / "spoken_variants" / "mr.toml").read_text(encoding="utf-8"))["pairs"]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(len(r), 6, r)
            self.assertIn(r[5], ("yes", "ambiguous", "not here"))
            self.assertNotEqual(r[0], r[1])

    def test_recall_and_switches(self):
        from pipeline.spelling import spoken_forms
        truth = "मला नाय माहिती पन तुम्ही या"
        std = spoken_forms(truth, "मला नाही माहिती पण तुम्ही या", "mr")
        self.assertEqual((std["hits"], std["misses"], std["kept"], std["switched"]), (0, 2, 1, 0))
        spoken = spoken_forms(truth, "मला नाय माहिती पन तुमी या", "mr")
        self.assertEqual((spoken["recall"], spoken["switch_rate"]), (1.0, 1.0))
        self.assertEqual(spoken_forms("मला नाय माहिती", "मला माहिती", "mr")["spoken_unmatched"], 1)

    def test_contractions_spacing_and_same_sound(self):
        from pipeline.spelling import ortho, spoken_forms
        self.assertEqual(spoken_forms("हो ठीके उद्या भेटू", "हो ठीक आहे उद्या भेटू", "mr")["misses"], 1)
        self.assertEqual(spoken_forms("हो ठीक आहे उद्या भेटू", "हो ठीके उद्या भेटू", "mr")["switched"], 1)
        self.assertEqual(ortho("सम्बन्ध", "mr"), ortho("संबंध", "mr"))      # nasal consonant = anusvara
        self.assertEqual(ortho("गाडी ला", "mr"), ortho("गाडीला", "mr"))      # written apart or together
        self.assertEqual(ortho("दिली", "mr"), ortho("दीली", "mr"))           # short or long i

    def test_content_level_folds_listed_spoken_forms(self):
        from pipeline.metrics import cer_level
        self.assertEqual(cer_level("मला नाय माहिती पन ठीके", "मला नाही माहिती पण ठीक आहे", "mr", "content"), 0.0)

    def test_screening_scores(self):
        from pipeline.screen import score
        lines = [{"id": "a", "fixture": "f", "language": "mr", "duration": 3.0, "text": "मला नाय माहिती",
                  "slots": [{"i": 1, "n": 1, "kind": "spoken", "truth": "नाय", "standard": "नाही"}], "tags": ["S", "N"]},
                 {"id": "b", "fixture": "f", "language": "mr", "duration": 1.0, "text": "हो", "slots": [],
                  "tags": ["R"]}]
        r = score(lines, {"a": "मला नाय माहिती", "b": "हो बोला"})
        self.assertEqual((r["S"]["hits"], r["S"]["misses"], r["S"]["recall"]), (1, 0, 1.0))
        self.assertEqual(r["R"]["words_not_in_truth"], 1)
        self.assertEqual((r["N"]["negations"], r["N"]["negation_errors"]), (1, 0))
        r = score(lines, {"a": "मला नाही माहिती", "b": "हो"})
        self.assertEqual((r["S"]["recall"], r["R"]["words_not_in_truth"]), (0.0, 0))


class TestPhonemeJudge(unittest.TestCase):
    # PLAN_v0.9 A3: Devanagari and the recogniser's IPA in the same coarse sound classes (invented examples)
    def test_reader(self):
        from pipeline.phonemes import read_devanagari, read_ipa
        self.assertEqual(read_devanagari("नाही"), ["n", "a", "h", "i"])
        self.assertEqual(read_devanagari("नाय"), ["n", "a", "y"])
        self.assertEqual(read_devanagari("घर"), ["g", "H", "@", "r"])          # aspiration; final schwa silent
        self.assertEqual(read_ipa("n aː h iː"), ["n", "a", "h", "i"])
        self.assertEqual(read_ipa("ɡʱ ə r"), ["g", "H", "@", "r"])

    def test_judge_picks_the_spelling_that_was_heard_and_keeps_ties(self):
        from pipeline.phonemes import judge
        self.assertEqual(judge("m ə l aː n aː j m aː h i t i", ["मला नाही माहिती", "मला नाय माहिती"])[0], 1)
        self.assertEqual(judge("m ə l aː n aː h iː", ["मला नाही", "मला नाय"])[0], 0)
        self.assertIsNone(judge("", ["मला नाही", "मला नाहि"])[0])            # same sounds: a tie, no change


class TestStandardCopyAndNames(unittest.TestCase):
    def test_spelling_line(self):
        from pipeline.stages.export import _spelling_line
        self.assertIn("no as-spoken spelling step", _spelling_line("as_spoken", 0, None))
        self.assertIn("3 line(s) differ", _spelling_line("as_spoken", 3, "spelling transfer"))

    def test_rare_names_written_several_ways(self):
        from pipeline.names import written_ways
        common = {"फोन", "केला", "आला", "येईल", "उद्या"}
        segs = [{"start": 1.0, "text": "सुनीलला फोन केला"}, {"start": 5.0, "text": "सुनिल आला"},
                {"start": 9.0, "text": "उद्या सुनील येईल"}, {"start": 12.0, "text": "मंगेश आला"}]
        groups = written_ways(segs, "mr", lambda w: 10_000 if w in common else 0, hotwords=["सुनील"], endings=("ला",))
        self.assertEqual(len(groups), 1)
        self.assertEqual([(st, n) for st, n, _ in groups[0]["spellings"]], [("सुनील", 2), ("सुनिल", 1)])
        self.assertEqual(groups[0]["listed"], "सुनील")
        self.assertEqual(written_ways(segs, "mr", lambda w: 10_000), [])   # common words: never a names list


class TestOfflineGuard(unittest.TestCase):
    def test_blocks_public_hosts(self):
        offline.install()
        with self.assertRaises(offline.NetworkBlocked):
            socket.create_connection(("example.com", 443), timeout=1)
        with self.assertRaises(offline.NetworkBlocked):
            socket.getaddrinfo("huggingface.co", 443)


if __name__ == "__main__":
    unittest.main()
