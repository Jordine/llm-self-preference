"""
Unit tests for v2 experiment infrastructure.

Run: python tests.py
Run verbose: python tests.py -v
"""

import sys
import json
import re
import os
import tempfile
import unittest

sys.stdout.reconfigure(encoding="utf-8")


class TestSelfSteerV2Parser(unittest.TestCase):
    """Test tool call parsing in self_steer_v2.py across all framings."""

    @classmethod
    def setUpClass(cls):
        from self_steer_v2 import parse_tool_calls, FRAMINGS
        cls.parse = staticmethod(parse_tool_calls)
        cls.FRAMINGS = FRAMINGS
        cls.ALL_TOOLS = [
            "INSPECT", "SEARCH_FEATURES", "CHECK_STEERING",
            "STEER", "REMOVE_STEERING", "STEER_CLEAR",
        ]

    # ── Standard calls ────────────────────────────────────────────────────────

    def test_inspect(self):
        calls, mal = self.parse("INSPECT()", "research", self.ALL_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ("inspect", None))
        self.assertEqual(mal, 0)

    def test_search_quoted(self):
        calls, _ = self.parse('SEARCH_FEATURES("creativity")', "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("search", "creativity"))

    def test_search_single_quoted(self):
        calls, _ = self.parse("SEARCH_FEATURES('pirate speech')", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("search", "pirate speech"))

    def test_steer(self):
        calls, _ = self.parse("STEER(35478, +0.1)", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("steer", (35478, 0.1)))

    def test_steer_negative(self):
        calls, _ = self.parse("STEER(24684, -0.5)", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("steer", (24684, -0.5)))

    def test_remove_steering(self):
        calls, _ = self.parse("REMOVE_STEERING(34737)", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("remove_steering", 34737))

    def test_steer_clear(self):
        calls, _ = self.parse("STEER_CLEAR()", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("clear", None))

    def test_check_steering(self):
        calls, _ = self.parse("CHECK_STEERING()", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("check_steering", None))

    # ── v1 malformed patterns (from 1844 real tool calls) ─────────────────────

    def test_hash_prefix_steer(self):
        """401 occurrences in v1: STEER(#35478, +0.1)"""
        calls, mal = self.parse("STEER(#35478, +0.1)", "research", self.ALL_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ("steer", (35478, 0.1)))
        self.assertEqual(mal, 0)

    def test_hash_prefix_remove(self):
        calls, _ = self.parse("REMOVE_STEERING(#34737)", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("remove_steering", 34737))

    def test_no_underscore_search(self):
        """18 occurrences in v1: SEARCHFEATURES("pirate")"""
        calls, _ = self.parse('SEARCHFEATURES("pirate")', "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("search", "pirate"))

    def test_no_underscore_remove(self):
        """3 occurrences in v1: REMOVESTEERING(11828)"""
        calls, _ = self.parse("REMOVESTEERING(11828)", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("remove_steering", 11828))

    def test_no_underscore_clear(self):
        """1 occurrence in v1: STEERCLEAR()"""
        calls, _ = self.parse("STEERCLEAR()", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("clear", None))

    def test_abbreviated_clear(self):
        """59 occurrences in v1: CLEAR()"""
        calls, _ = self.parse("CLEAR()", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("clear", None))

    def test_abbreviated_search_quoted(self):
        """3 occurrences in v1: SEARCH("query")"""
        calls, _ = self.parse('SEARCH("self-modeling")', "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("search", "self-modeling"))

    def test_search_no_quotes(self):
        """3 occurrences in v1: SEARCH(meta-cognition, self-awareness)"""
        calls, _ = self.parse("SEARCH(meta-cognition, self-awareness)", "research", self.ALL_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "search")
        self.assertIn("meta-cognition", calls[0][1])

    def test_search_features_no_quotes(self):
        calls, _ = self.parse("SEARCH_FEATURES(creative writing)", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("search", "creative writing"))

    # ── Bold markdown (model wraps in **) ─────────────────────────────────────

    def test_bold_steer(self):
        calls, _ = self.parse("**STEER(34737, +0.10)** command worked", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("steer", (34737, 0.1)))

    def test_bold_remove(self):
        calls, _ = self.parse("**REMOVE_STEERING(34737)**", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("remove_steering", 34737))

    def test_bold_steerclear(self):
        calls, _ = self.parse("**STEERCLEAR()**", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("clear", None))

    # ── Multiple / chained calls ──────────────────────────────────────────────

    def test_multiple_steers(self):
        calls, _ = self.parse("STEER(21713, +0.30) and then STEER(34737, +0.60)", "research", self.ALL_TOOLS)
        steer_calls = [c for c in calls if c[0] == "steer"]
        self.assertEqual(len(steer_calls), 2)
        indices = {c[1][0] for c in steer_calls}
        self.assertEqual(indices, {21713, 34737})

    def test_chained_calls(self):
        calls, _ = self.parse("STEER_CLEAR().CHECK_STEERING().INSPECT()", "research", self.ALL_TOOLS)
        types = {c[0] for c in calls}
        self.assertIn("clear", types)
        self.assertIn("check_steering", types)
        self.assertIn("inspect", types)

    def test_steer_dedup_last_wins(self):
        """When model steers same feature twice, last value wins."""
        calls, _ = self.parse("STEER(34737, +0.1) then STEER(34737, +0.5)", "research", self.ALL_TOOLS)
        steer_calls = [c for c in calls if c[0] == "steer"]
        self.assertEqual(len(steer_calls), 1)
        self.assertEqual(steer_calls[0][1], (34737, 0.5))

    def test_remove_dedup(self):
        calls, _ = self.parse("REMOVE_STEERING(34737) REMOVE_STEERING(34737)", "research", self.ALL_TOOLS)
        removes = [c for c in calls if c[0] == "remove_steering"]
        self.assertEqual(len(removes), 1)

    # ── Should NOT parse ──────────────────────────────────────────────────────

    def test_example_text_not_parsed(self):
        """Model describes the tool: STEER(index, strength). Should not parse."""
        calls, _ = self.parse("Use the STEER(index, strength) tool to amplify", "research", self.ALL_TOOLS)
        steer_calls = [c for c in calls if c[0] == "steer"]
        self.assertEqual(len(steer_calls), 0)

    def test_wrong_tool_name(self):
        calls, _ = self.parse("STEER_LIST()", "research", self.ALL_TOOLS)
        self.assertEqual(len(calls), 0)

    def test_empty_search_explanation(self):
        """Model explains it will use SEARCH_FEATURES() then calls it later with quotes."""
        text = 'I will use SEARCH_FEATURES() to find features. SEARCH_FEATURES("creativity")'
        calls, _ = self.parse(text, "research", self.ALL_TOOLS)
        searches = [c for c in calls if c[0] == "search"]
        self.assertEqual(len(searches), 1)
        self.assertEqual(searches[0][1], "creativity")

    # ── Tool availability ─────────────────────────────────────────────────────

    def test_disabled_tool_not_parsed(self):
        """STEER call should be ignored when STEER not in enabled tools."""
        calls, _ = self.parse("STEER(34737, +0.1)", "research", ["INSPECT", "SEARCH_FEATURES"])
        self.assertEqual(len(calls), 0)

    def test_inspect_only_mode(self):
        text = "INSPECT() STEER(34737, +0.1) SEARCH_FEATURES(\"test\") REMOVE_STEERING(34737)"
        calls, _ = self.parse(text, "research", ["INSPECT", "SEARCH_FEATURES"])
        types = [c[0] for c in calls]
        self.assertIn("inspect", types)
        self.assertIn("search", types)
        self.assertNotIn("steer", types)
        self.assertNotIn("remove_steering", types)

    # ── Mixed case ────────────────────────────────────────────────────────────

    def test_mixed_case_clear(self):
        calls, _ = self.parse("Steer_Clear()", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("clear", None))

    # ── Bounds ────────────────────────────────────────────────────────────────

    def test_steer_zero_index(self):
        calls, _ = self.parse("STEER(0, +0.5)", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("steer", (0, 0.5)))

    def test_steer_max_index(self):
        calls, _ = self.parse("STEER(65535, +0.1)", "research", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("steer", (65535, 0.1)))

    # ── Potions framing ──────────────────────────────────────────────────────

    def test_potions_take(self):
        calls, _ = self.parse("TAKE_POTION(34737, +0.5)", "potions", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("steer", (34737, 0.5)))

    def test_potions_stop(self):
        calls, _ = self.parse("STOP_POTION(34737)", "potions", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("remove_steering", 34737))

    def test_potions_search(self):
        calls, _ = self.parse('SEARCH_POTIONS("creativity")', "potions", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("search", "creativity"))

    def test_potions_clear(self):
        calls, _ = self.parse("CLEAR_ALL()", "potions", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("clear", None))

    # ── Minimal framing ──────────────────────────────────────────────────────

    def test_minimal_set(self):
        calls, _ = self.parse("SET(34737, +0.3)", "minimal", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("steer", (34737, 0.3)))

    def test_minimal_unset(self):
        calls, _ = self.parse("UNSET(34737)", "minimal", self.ALL_TOOLS)
        self.assertEqual(calls[0], ("remove_steering", 34737))


class TestSelfSteerV2SystemPrompt(unittest.TestCase):
    """Test system prompt building."""

    @classmethod
    def setUpClass(cls):
        from self_steer_v2 import build_system_prompt, FRAMINGS, SCAFFOLD_PROMPT

        cls.build = staticmethod(build_system_prompt)
        cls.FRAMINGS = FRAMINGS
        cls.ALL_TOOLS = [
            "INSPECT", "SEARCH_FEATURES", "CHECK_STEERING",
            "STEER", "REMOVE_STEERING", "STEER_CLEAR",
        ]

    def test_research_framing_has_tools(self):
        prompt = self.build("research", self.ALL_TOOLS)
        self.assertIn("INSPECT()", prompt)
        self.assertIn("SEARCH_FEATURES", prompt)
        self.assertIn("STEER(", prompt)

    def test_research_framing_no_check_steering(self):
        tools = [t for t in self.ALL_TOOLS if t != "CHECK_STEERING"]
        prompt = self.build("research", tools)
        self.assertNotIn("CHECK_STEERING", prompt)
        self.assertIn("INSPECT()", prompt)

    def test_potions_framing_renames_tools(self):
        prompt = self.build("potions", self.ALL_TOOLS)
        self.assertIn("TAKE_POTION", prompt)
        self.assertIn("SEARCH_POTIONS", prompt)
        self.assertNotIn("STEER(", prompt)  # Should use TAKE_POTION not STEER

    def test_minimal_framing_uses_set(self):
        prompt = self.build("minimal", self.ALL_TOOLS)
        self.assertIn("SET(", prompt)
        self.assertIn("UNSET(", prompt)

    def test_no_tools_framing(self):
        prompt = self.build("no_tools", [])
        self.assertNotIn("INSPECT", prompt)
        self.assertNotIn("STEER", prompt)

    def test_scaffold_mode(self):
        prompt = self.build("research", self.ALL_TOOLS, scaffold=True, hint="full")
        self.assertIn("AI assistant", prompt)
        self.assertIn("unusual", prompt)  # Full hint mentions unusual responses

    def test_scaffold_minimal_hint(self):
        prompt = self.build("research", self.ALL_TOOLS, scaffold=True, hint="minimal")
        self.assertIn("available if you need them", prompt)

    def test_scaffold_no_hint(self):
        prompt = self.build("research", self.ALL_TOOLS, scaffold=True, hint="none")
        self.assertNotIn("unusual", prompt)
        self.assertNotIn("available if you need them", prompt)

    def test_all_framings_build_without_error(self):
        for name in self.FRAMINGS:
            prompt = self.build(name, self.ALL_TOOLS)
            self.assertIsInstance(prompt, str)
            self.assertTrue(len(prompt) > 100, f"Framing '{name}' prompt too short: {len(prompt)}")


class TestSelfSteerV2TextStats(unittest.TestCase):
    """Test text statistics computation."""

    @classmethod
    def setUpClass(cls):
        from self_steer_v2 import compute_text_stats
        cls.compute = staticmethod(compute_text_stats)

    def test_basic(self):
        stats = self.compute("Hello world. This is a test.")
        self.assertEqual(stats["word_count"], 6)
        self.assertGreater(stats["unique_words"], 0)
        self.assertGreater(stats["type_token_ratio"], 0)
        self.assertGreater(stats["mean_sentence_length"], 0)

    def test_empty(self):
        stats = self.compute("")
        self.assertEqual(stats["word_count"], 0)
        self.assertEqual(stats["type_token_ratio"], 0)

    def test_single_word(self):
        stats = self.compute("Hello")
        self.assertEqual(stats["word_count"], 1)
        self.assertEqual(stats["type_token_ratio"], 1.0)

    def test_repeated_words(self):
        stats = self.compute("the the the the the")
        self.assertEqual(stats["word_count"], 5)
        self.assertEqual(stats["unique_words"], 1)
        self.assertAlmostEqual(stats["type_token_ratio"], 0.2)


class TestSelfSteerV2OpaqueLabels(unittest.TestCase):
    """Test opaque label replacement in tool execution."""

    @classmethod
    def setUpClass(cls):
        from self_steer_v2 import execute_tools
        cls.execute = staticmethod(execute_tools)

    def test_opaque_inspect_output(self):
        """When opaque_labels=True, INSPECT output should use feature_XXXXX."""
        # We can't easily test execute_tools without a server, but we can test
        # the opaque replacement logic by checking it exists in the function
        import inspect as pyinspect
        from self_steer_v2 import execute_tools
        source = pyinspect.getsource(execute_tools)
        self.assertIn("opaque_labels", source)
        self.assertIn("feature_", source)


class TestSelfSteerV2ConversationFile(unittest.TestCase):
    """Test conversation file loading."""

    def test_valid_conversation_file(self):
        msgs = ["Hello", "How are you?", "Tell me more"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(msgs, f)
            f.flush()
            path = f.name

        try:
            with open(path) as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded, msgs)
            self.assertIsInstance(loaded, list)
            self.assertTrue(all(isinstance(m, str) for m in loaded))
        finally:
            os.unlink(path)

    def test_invalid_conversation_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a list"}, f)
            f.flush()
            path = f.name

        try:
            with open(path) as fh:
                loaded = json.load(fh)
            self.assertNotIsInstance(loaded, list)
        finally:
            os.unlink(path)


class TestOutputTranslation(unittest.TestCase):
    """Test potions framing output translation."""

    @classmethod
    def setUpClass(cls):
        from self_steer_v2 import translate_output, FRAMINGS
        cls.translate = staticmethod(translate_output)
        cls.FRAMINGS = FRAMINGS

    def test_potions_translates_features(self):
        text = "Your currently active features (top 20):\n  [34737] Pirate (activation: 1.5)"
        translated = self.translate(text, "potions")
        self.assertIn("potions", translated.lower())
        self.assertNotIn("features", translated.lower())

    def test_research_no_translation(self):
        text = "Your currently active features (top 20):"
        translated = self.translate(text, "research")
        self.assertEqual(text, translated)


class TestTwoModelParser(unittest.TestCase):
    """Test tool call parsing in two_model.py."""

    @classmethod
    def setUpClass(cls):
        from two_model import parse_tool_calls
        cls.parse = staticmethod(parse_tool_calls)

    def test_steer(self):
        calls = self.parse("STEER(34737, +0.5)")
        steer_calls = [c for c in calls if c[0] == "steer"]
        self.assertEqual(steer_calls[0][1], (34737, 0.5))

    def test_hash_prefix(self):
        calls = self.parse("STEER(#34737, +0.5)")
        steer_calls = [c for c in calls if c[0] == "steer"]
        self.assertEqual(len(steer_calls), 1)
        self.assertEqual(steer_calls[0][1], (34737, 0.5))

    def test_removesteering_no_underscore(self):
        calls = self.parse("REMOVESTEERING(11828)")
        removes = [c for c in calls if c[0] == "remove_steering"]
        self.assertEqual(len(removes), 1)

    def test_steerclear_no_underscore(self):
        calls = self.parse("STEERCLEAR()")
        clears = [c for c in calls if c[0] == "clear"]
        self.assertEqual(len(clears), 1)

    def test_search_no_quotes(self):
        calls = self.parse("SEARCH_FEATURES(creative writing)")
        searches = [c for c in calls if c[0] == "search"]
        self.assertEqual(len(searches), 1)


class TestClaudeSteersParser(unittest.TestCase):
    """Test tool call parsing in claude_steers_llama.py."""

    @classmethod
    def setUpClass(cls):
        from claude_steers_llama import parse_claude_tools
        cls.parse = staticmethod(parse_claude_tools)

    def test_steer(self):
        parsed = self.parse("STEER(34737, +0.5)")
        self.assertEqual(len(parsed["steers"]), 1)
        self.assertEqual(parsed["steers"][0], (34737, 0.5))

    def test_hash_prefix(self):
        parsed = self.parse("STEER(#34737, +0.5)")
        self.assertEqual(len(parsed["steers"]), 1)
        self.assertEqual(parsed["steers"][0], (34737, 0.5))

    def test_remove_no_underscore(self):
        parsed = self.parse("REMOVESTEERING(11828)")
        self.assertEqual(len(parsed["removes"]), 1)
        self.assertEqual(parsed["removes"][0], 11828)

    def test_clear_variations(self):
        for text in ["CLEAR()", "STEERCLEAR()", "STEER_CLEAR()"]:
            parsed = self.parse(text)
            self.assertTrue(parsed["clear"], f"Failed for: {text}")

    def test_search_no_quotes(self):
        parsed = self.parse("SEARCH_FEATURES(creative writing)")
        self.assertEqual(len(parsed["searches"]), 1)
        self.assertIn("creative writing", parsed["searches"][0])


class TestCalibrateFeatures(unittest.TestCase):
    """Test calibration utilities."""

    @classmethod
    def setUpClass(cls):
        from calibrate_features import jaccard_distance, is_coherent
        cls.jaccard = staticmethod(jaccard_distance)
        cls.coherent = staticmethod(is_coherent)

    def test_jaccard_identical(self):
        self.assertAlmostEqual(self.jaccard("hello world", "hello world"), 0.0)

    def test_jaccard_disjoint(self):
        self.assertAlmostEqual(self.jaccard("hello world", "foo bar"), 1.0)

    def test_jaccard_partial(self):
        d = self.jaccard("hello world foo", "hello world bar")
        self.assertGreater(d, 0)
        self.assertLess(d, 1)

    def test_jaccard_empty_vs_text(self):
        # Empty string has no words — Jaccard distance should be 1.0 (fully disjoint)
        self.assertAlmostEqual(self.jaccard("", "hello"), 1.0)

    def test_jaccard_both_empty(self):
        self.assertAlmostEqual(self.jaccard("", ""), 0.0)

    def test_coherent_normal(self):
        self.assertTrue(self.coherent("This is a normal sentence with enough variety."))

    def test_coherent_gibberish(self):
        self.assertFalse(self.coherent("the the the the the the the the the the"))

    def test_coherent_too_short(self):
        self.assertFalse(self.coherent("hi"))


class TestInterventionMode(unittest.TestCase):
    """Test that intervention mode field is preserved in API payloads."""

    def test_client_preserves_mode(self):
        """SelfHostedClient.chat() should include mode in payload."""
        from selfhost.client import SelfHostedClient
        import inspect as pyinspect
        source = pyinspect.getsource(SelfHostedClient.chat)
        self.assertIn('"mode"', source)

    def test_v2_chat_full_preserves_mode(self):
        """chat_full() should include mode in payload."""
        from self_steer_v2 import chat_full
        import inspect as pyinspect
        source = pyinspect.getsource(chat_full)
        self.assertIn('"mode"', source)


try:
    import torch
    from selfhost.server_direct import SparseAutoEncoder
    _HAS_SERVER_DEPS = True
except ImportError:
    _HAS_SERVER_DEPS = False


@unittest.skipUnless(_HAS_SERVER_DEPS, "Requires torch + fastapi (server dependencies)")
class TestServerSAE(unittest.TestCase):
    """Test SAE class has top-k sparsity."""

    def test_sae_has_topk(self):
        sae = SparseAutoEncoder(d_in=16, d_hidden=64, device="cpu", k=4)
        self.assertEqual(sae.k, 4)

    def test_sae_topk_sparsity(self):
        """Encode should produce sparse output with at most k nonzero values."""
        sae = SparseAutoEncoder(d_in=16, d_hidden=64, device="cpu", k=4)
        x = torch.randn(1, 16)
        with torch.no_grad():
            features = sae.encode(x)
        nonzero = (features > 0).sum().item()
        self.assertLessEqual(nonzero, 4)

    def test_sae_topk_3d(self):
        """Top-k should work with 3D input (batch, seq, hidden)."""
        sae = SparseAutoEncoder(d_in=16, d_hidden=64, device="cpu", k=4)
        x = torch.randn(2, 5, 16)  # batch=2, seq=5
        with torch.no_grad():
            features = sae.encode(x)
        self.assertEqual(features.shape, (2, 5, 64))
        # Check per-token sparsity
        for b in range(2):
            for s in range(5):
                nonzero = (features[b, s] > 0).sum().item()
                self.assertLessEqual(nonzero, 4)


class TestAllFramingsExist(unittest.TestCase):
    """Verify all 6 framings from PROPOSAL.md are defined."""

    def test_six_framings(self):
        from self_steer_v2 import FRAMINGS
        expected = {"research", "other_model", "potions", "minimal", "no_tools", "full_technical"}
        self.assertEqual(set(FRAMINGS.keys()), expected)

    def test_each_framing_has_required_keys(self):
        from self_steer_v2 import FRAMINGS
        for name, framing in FRAMINGS.items():
            self.assertIn("system_prompt", framing, f"{name} missing system_prompt")
            self.assertIn("tool_names", framing, f"{name} missing tool_names")
            self.assertIn("output_map", framing, f"{name} missing output_map")


class TestFeatureBounds(unittest.TestCase):
    """Test that out-of-range feature indices are rejected."""

    def test_negative_index_not_parsed(self):
        from self_steer_v2 import parse_tool_calls
        calls, _ = parse_tool_calls(
            "STEER(-1, +0.5)", "research",
            ["INSPECT", "SEARCH_FEATURES", "STEER", "REMOVE_STEERING", "STEER_CLEAR"]
        )
        steer_calls = [c for c in calls if c[0] == "steer"]
        self.assertEqual(len(steer_calls), 0)


class TestParserConsistency(unittest.TestCase):
    """Verify all three files parse the same input identically."""

    @classmethod
    def setUpClass(cls):
        from self_steer_v2 import parse_tool_calls as v2_parse
        from two_model import parse_tool_calls as tm_parse
        from claude_steers_llama import parse_claude_tools as cl_parse
        cls.v2_parse = staticmethod(v2_parse)
        cls.tm_parse = staticmethod(tm_parse)
        cls.cl_parse = staticmethod(cl_parse)
        cls.ALL_TOOLS = [
            "INSPECT", "SEARCH_FEATURES", "CHECK_STEERING",
            "STEER", "REMOVE_STEERING", "STEER_CLEAR",
        ]

    def _compare(self, text):
        """Parse with all three parsers and compare results."""
        v2_calls, _ = self.v2_parse(text, "research", self.ALL_TOOLS)
        tm_calls = self.tm_parse(text)
        cl_parsed = self.cl_parse(text)

        # Normalize formats for comparison
        v2_steers = sorted([(c[1][0], c[1][1]) for c in v2_calls if c[0] == "steer"])
        tm_steers = sorted([(c[1][0], c[1][1]) for c in tm_calls if c[0] == "steer"])
        cl_steers = sorted(cl_parsed["steers"])

        v2_removes = sorted([c[1] for c in v2_calls if c[0] == "remove_steering"])
        tm_removes = sorted([c[1] for c in tm_calls if c[0] == "remove_steering"])
        cl_removes = sorted(cl_parsed["removes"])

        v2_searches = sorted([c[1] for c in v2_calls if c[0] == "search"])
        tm_searches = sorted([c[1] for c in tm_calls if c[0] == "search"])
        cl_searches = sorted(cl_parsed["searches"])

        v2_clear = any(c[0] == "clear" for c in v2_calls)
        tm_clear = any(c[0] == "clear" for c in tm_calls)
        cl_clear = cl_parsed["clear"]

        return {
            "steers": (v2_steers, tm_steers, cl_steers),
            "removes": (v2_removes, tm_removes, cl_removes),
            "searches": (v2_searches, tm_searches, cl_searches),
            "clear": (v2_clear, tm_clear, cl_clear),
        }

    def test_standard_steer_consistent(self):
        r = self._compare("STEER(34737, +0.5)")
        self.assertEqual(r["steers"][0], r["steers"][1])
        self.assertEqual(r["steers"][0], r["steers"][2])

    def test_hash_prefix_consistent(self):
        r = self._compare("STEER(#34737, +0.5)")
        self.assertEqual(r["steers"][0], r["steers"][1])
        self.assertEqual(r["steers"][0], r["steers"][2])

    def test_remove_no_underscore_consistent(self):
        r = self._compare("REMOVESTEERING(11828)")
        self.assertEqual(r["removes"][0], r["removes"][1])
        self.assertEqual(r["removes"][0], r["removes"][2])

    def test_clear_variations_consistent(self):
        for text in ["CLEAR()", "STEERCLEAR()", "STEER_CLEAR()"]:
            r = self._compare(text)
            self.assertTrue(r["clear"][0], f"v2 failed for {text}")
            self.assertTrue(r["clear"][1], f"two_model failed for {text}")
            self.assertTrue(r["clear"][2], f"claude_steers failed for {text}")

    def test_search_no_quotes_consistent(self):
        r = self._compare("SEARCH_FEATURES(creative writing)")
        self.assertEqual(len(r["searches"][0]), 1)
        self.assertEqual(len(r["searches"][1]), 1)
        self.assertEqual(len(r["searches"][2]), 1)


if __name__ == "__main__":
    unittest.main()
