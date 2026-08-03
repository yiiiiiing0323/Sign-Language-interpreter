import os
import tempfile
import unittest

import pandas as pd

from b_stream import BStreamGestureMatcher
from compound_phrase import CompoundPhraseResolver
from core.safe_rule_engine import SafeRuleEvaluator
from core.word_normalization import canonical_word
from fusion import DecisionFusion


def write_database(rows):
    handle = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    handle.close()
    pd.DataFrame(rows).to_excel(handle.name, sheet_name="工作表3", index=False)
    return handle.name


class SafeRuleEvaluatorTests(unittest.TestCase):
    def test_unknown_distance_fails_closed(self):
        result = SafeRuleEvaluator().evaluate("dist_typo < 0.15", {})
        self.assertFalse(result.matched)
        self.assertEqual(result.unknown_names, {"dist_typo"})


class WordIntegrationTests(unittest.TestCase):
    def test_canonical_word_handles_slash_and_variant_suffixes(self):
        self.assertEqual(canonical_word("爸爸/父親"), "爸爸")
        self.assertEqual(canonical_word("車_A_N/巴士"), "車")

    def test_fusion_recognizes_canonical_and_variant_as_synonyms(self):
        fusion = DecisionFusion()
        self.assertTrue(fusion.are_similar("我", "我_A"))
        self.assertEqual(fusion.fuse(("我", 0.8), ("我_A", 0.8), logic_kind="static")[0], "我")

    def test_logic_event_is_not_replaced_by_previous_ai_word(self):
        fusion = DecisionFusion()
        for _ in range(4):
            fusion.fuse(("吃", 0.9), None)
        self.assertEqual(
            fusion.fuse(None, ("路/道路", 0.9), logic_kind="sequence"),
            ("路", "LOGIC"),
        )

    def test_stable_static_logic_beats_unrelated_high_confidence_ai(self):
        fusion = DecisionFusion()
        self.assertEqual(
            fusion.fuse(("wrong_ai", 0.99), ("logic_static", 0.8), logic_kind="static"),
            ("logic_static", "LOGIC"),
        )

    def test_compound_rule_matches_slash_output(self):
        resolver = CompoundPhraseResolver("database.xlsx")
        match = resolver.resolve_tail(["爸爸/父親", "弟弟"])
        self.assertIsNotNone(match)
        self.assertEqual(match.output, "叔叔")


class BStreamTests(unittest.TestCase):
    def tearDown(self):
        for path in getattr(self, "paths", []):
            if os.path.exists(path):
                os.unlink(path)

    def make_matcher(self, rows):
        self.paths = getattr(self, "paths", [])
        path = write_database(rows)
        self.paths.append(path)
        return BStreamGestureMatcher(path)

    def test_static_rule_keeps_three_frame_debounce(self):
        matcher = self.make_matcher([
            {"ID": "S_1", "中文": "靜態", "MediaPipe 關鍵特徵": "is_ready == True"},
        ])
        self.assertIsNone(matcher.evaluate_frame_with_confidence({"is_ready": True}))
        self.assertIsNone(matcher.evaluate_frame_with_confidence({"is_ready": True}))
        self.assertEqual(matcher.evaluate_frame_with_confidence({"is_ready": True})[0], "靜態")
        self.assertEqual(matcher.last_match_kind, "static")

    def test_sequence_emits_immediately_without_static_debounce(self):
        matcher = self.make_matcher([
            {
                "ID": "V_1",
                "中文": "動態",
                "MediaPipe 關鍵特徵": "sequence([step_one == True], [step_two == True])",
            },
        ])
        self.assertIsNone(matcher.evaluate_frame_with_confidence({"step_one": True, "step_two": False}))
        result = matcher.evaluate_frame_with_confidence({"step_one": False, "step_two": True})
        self.assertEqual(result[0], "動態")
        self.assertEqual(matcher.last_match_kind, "sequence")

    def test_reset_prevents_sequence_completion_after_tracking_loss(self):
        matcher = self.make_matcher([
            {
                "ID": "N_1",
                "中文": "路",
                "MediaPipe 關鍵特徵": "sequence([step_one == True], [distance > 0.55])",
            },
        ])
        matcher.evaluate_frame_with_confidence({"step_one": True, "distance": 0.1})
        matcher.reset_transient_state()
        self.assertIsNone(matcher.evaluate_frame_with_confidence({"step_one": False, "distance": 99.0}))

    def test_more_specific_rule_wins_regardless_of_excel_row(self):
        matcher = self.make_matcher([
            {"ID": "TOP", "中文": "上方泛用詞", "MediaPipe 關鍵特徵": "shape == True"},
            {"ID": "BOTTOM", "中文": "下方精確詞", "MediaPipe 關鍵特徵": "shape == True and near == True"},
        ])
        result = None
        for _ in range(3):
            result = matcher.evaluate_frame_with_confidence({"shape": True, "near": True})
        self.assertEqual(result[0], "下方精確詞")

    def test_equal_rules_return_candidates_instead_of_first_row(self):
        matcher = self.make_matcher([
            {"ID": "TOP", "中文": "上方詞", "MediaPipe 關鍵特徵": "shape == True"},
            {"ID": "BOTTOM", "中文": "下方詞", "MediaPipe 關鍵特徵": "shape == True"},
        ])
        result = None
        for _ in range(3):
            result = matcher.evaluate_frame_with_confidence({"shape": True})
        self.assertEqual(result[0], "[上方詞/下方詞]")


class BStreamVisibilityTests(unittest.TestCase):
    def tearDown(self):
        for path in getattr(self, "paths", []):
            if os.path.exists(path):
                os.unlink(path)

    def make_matcher(self, rows):
        self.paths = getattr(self, "paths", [])
        path = write_database(rows)
        self.paths.append(path)
        return BStreamGestureMatcher(path)

    def test_static_pending_match_is_visible_before_debounce_output(self):
        matcher = self.make_matcher([
            {"ID": "S_1", "中文": "ME_A", "MediaPipe 關鍵特徵": "is_ready == True"},
        ])
        self.assertIsNone(matcher.evaluate_frame_with_confidence({"is_ready": True}))
        self.assertEqual(matcher.last_candidate_words, ("ME_A",))
        self.assertEqual(matcher.last_candidate_ids, ("S_1",))
        self.assertIsNone(matcher.last_match_kind)

    def test_suppressed_static_match_is_visible_when_another_rule_wins(self):
        matcher = self.make_matcher([
            {"ID": "TOP", "中文": "generic", "MediaPipe 關鍵特徵": "shape == True"},
            {"ID": "BOTTOM", "中文": "specific", "MediaPipe 關鍵特徵": "shape == True and near == True"},
        ])
        result = None
        for _ in range(3):
            result = matcher.evaluate_frame_with_confidence({"shape": True, "near": True})
        self.assertEqual(result[0], "specific")
        self.assertIn("generic", matcher.last_suppressed_words)
        self.assertIn("TOP", matcher.last_suppressed_ids)


if __name__ == "__main__":
    unittest.main()
