"""
Unit tests for the CodeInspector tool.
"""
import unittest

from qassistant.agent.tools.codeinspector import CodeInspector, ApiEntry, _collect_aliases, _extract_entries, _build_signature

import griffe


class TestCollectAliases(unittest.TestCase):
    """
    Validate alias collection from griffe modules.
    """

    def test_collects_reexported_classes(self):
        """
        Aliases for re-exported classes should map target_path to alias_path.
        """
        mod = griffe.inspect("json")
        aliases = _collect_aliases(mod)

        self.assertEqual(aliases["json.encoder.JSONEncoder"], "json.JSONEncoder")
        self.assertEqual(aliases["json.decoder.JSONDecoder"], "json.JSONDecoder")

    def test_skips_private_aliases(self):
        """
        Private aliases (leading underscore) should not appear in the map.
        """
        mod = griffe.inspect("json")
        aliases = _collect_aliases(mod)

        for target, alias_path in aliases.items():
            leaf = alias_path.rsplit(".", 1)[-1]
            self.assertFalse(leaf.startswith("_"), f"private alias collected: {alias_path}")

    def test_empty_for_module_without_aliases(self):
        """
        A module with no public aliases should return an empty dict.
        """
        mod = griffe.inspect("math")
        aliases = _collect_aliases(mod)

        self.assertEqual(aliases, {})


class TestExtractEntries(unittest.TestCase):
    """
    Validate extraction of public API entries from griffe modules.
    """

    def test_extracts_public_functions(self):
        """
        Public module-level functions should be extracted.
        """
        mod = griffe.inspect("json")
        entries = _extract_entries(mod)

        names = [e.full_name for e in entries]
        self.assertIn("json.dumps", names)
        self.assertIn("json.loads", names)

    def test_skips_private_members(self):
        """
        Private members (leading underscore) should not appear in entries.
        """
        mod = griffe.inspect("json")
        entries = _extract_entries(mod)

        for entry in entries:
            short = entry.full_name.rsplit(".", 1)[-1]
            self.assertFalse(short.startswith("_"), f"private member extracted: {entry.full_name}")

    def test_extracts_classes_and_methods(self):
        """
        Public classes and their public methods should be extracted from pathlib.
        """
        mod = griffe.inspect("pathlib")
        entries = _extract_entries(mod)

        names = [e.full_name for e in entries]
        self.assertIn("pathlib.Path", names)
        self.assertIn("pathlib.Path.glob", names)

    def test_function_signature_includes_parameters(self):
        """
        Extracted function signatures should contain parameter names.
        """
        mod = griffe.inspect("json")
        entries = _extract_entries(mod)

        dumps_entry = next(e for e in entries if e.full_name == "json.dumps")
        self.assertIn("obj", dumps_entry.signature)

    def test_docstring_is_populated(self):
        """
        Entries with docstrings in source should have non-empty docstring fields.
        """
        mod = griffe.inspect("json")
        entries = _extract_entries(mod)

        dumps_entry = next(e for e in entries if e.full_name == "json.dumps")
        self.assertTrue(len(dumps_entry.docstring) > 0)

    def test_alias_is_full_name_for_direct_members(self):
        """
        Non-aliased entries should have alias equal to their full_name.
        """
        mod = griffe.inspect("json")
        entries = _extract_entries(mod)

        dumps_entry = next(e for e in entries if e.full_name == "json.dumps")
        self.assertEqual(dumps_entry.alias, "json.dumps")

    def test_alias_for_reexported_class(self):
        """
        Re-exported class should carry the shortest alias path.
        """
        mod = griffe.inspect("json")
        entries = _extract_entries(mod)

        encoder_entry = next(e for e in entries if e.full_name == "json.encoder.JSONEncoder")
        self.assertEqual(encoder_entry.alias, "json.JSONEncoder")

    def test_alias_for_reexported_class_method(self):
        """
        Methods of a re-exported class should carry alias derived from the class alias.
        """
        mod = griffe.inspect("json")
        entries = _extract_entries(mod)

        method_entries = [e for e in entries if e.full_name.startswith("json.encoder.JSONEncoder.") and e.kind == "method"]
        self.assertTrue(len(method_entries) > 0)
        for entry in method_entries:
            mname = entry.full_name.rsplit(".", 1)[-1]
            self.assertEqual(entry.alias, f"json.JSONEncoder.{mname}")


class TestBuildSignature(unittest.TestCase):
    """
    Validate signature building from griffe objects.
    """

    def test_function_signature(self):
        """
        Function signatures should include the full path and parameters.
        """
        mod = griffe.inspect("json")
        func = mod.members["dumps"]
        sig = _build_signature(func)

        self.assertTrue(sig.startswith("json.dumps("))
        self.assertIn("obj", sig)

    def test_class_signature(self):
        """
        Class signatures should prefix with 'class' and include __init__ params.
        """
        mod = griffe.inspect("pathlib")
        cls = mod.members["Path"]
        sig = _build_signature(cls)

        self.assertTrue(sig.startswith("class pathlib.Path"))


class TestCodeInspector(unittest.IsolatedAsyncioTestCase):
    """
    Validate CodeInspector register, search, and get_signatures.
    """

    @staticmethod
    async def _dummy_embed(text: str) -> list[float]:
        """
        Deterministic embedding using character frequencies.
        """
        vec = [0.0] * 128
        for ch in text:
            idx = ord(ch) % 128
            vec[idx] += 1.0
        return vec

    async def test_register_populates_entries(self):
        """
        After register(), the inspector should contain API entries.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        self.assertTrue(len(inspector.entries) > 0)
        names = [e.full_name for e in inspector.entries]
        self.assertIn("json.dumps", names)

    async def test_register_embeds_entries(self):
        """
        All registered entries should have non-None embeddings.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        for entry in inspector.entries:
            self.assertIsNotNone(entry.embedding)

    async def test_search_returns_ranked_results(self):
        """
        Search should return a list of dict results.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        results = await inspector.search("serialize object to JSON string")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], dict)
        self.assertIn("full_name", results[0])

    async def test_search_returns_same_keys_as_list_members(self):
        """
        Search and list_members should return dicts with the same keys.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        search_results = await inspector.search("dumps")
        list_results = inspector.list_members("json")
        self.assertTrue(len(search_results) > 0)
        self.assertTrue(len(list_results) > 0)
        self.assertEqual(set(search_results[0].keys()), set(list_results[0].keys()))

    async def test_search_returns_full_docstring(self):
        """
        Search results should contain the complete docstring.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        results = await inspector.search("dumps")
        dumps = next(r for r in results if r["name"] == "dumps")
        self.assertNotEqual(dumps["description"], "")
        self.assertFalse(dumps["description"].endswith("..."))
        self.assertIn("\n", dumps["description"])

    async def test_search_respects_start_end_index(self):
        """
        Pagination via start_index and end_index should slice the results.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        all_results = await inspector.search("encode", start_index=0, end_index=100)
        sliced = await inspector.search("encode", start_index=1, end_index=3)
        self.assertEqual(sliced, all_results[1:3])

    async def test_search_on_empty_inspector(self):
        """
        Search on an inspector with no registered modules should return empty.
        """
        inspector = CodeInspector(self._dummy_embed)
        results = await inspector.search("anything")
        self.assertEqual(results, [])

    async def test_search_with_context_filters_by_module(self):
        """
        Search with context limits results to entries from that module.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")
        await inspector.register("pathlib")

        results = await inspector.search("path", context="json", end_index=100)
        for result in results:
            self.assertTrue(
                result["full_name"].startswith("json."),
                f"unexpected entry outside context: {result['full_name']}",
            )

    async def test_search_with_context_filters_by_class(self):
        """
        Search with context set to a class limits results to that class members.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("pathlib")

        results = await inspector.search("glob", context="pathlib.Path", end_index=100)
        self.assertTrue(len(results) > 0)
        for result in results:
            self.assertTrue(
                result["full_name"].startswith("pathlib.Path."),
                f"unexpected entry outside context: {result['full_name']}",
            )

    async def test_search_with_nonmatching_context_returns_empty(self):
        """
        Search with a context that matches no entries should return empty.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        results = await inspector.search("dumps", context="nonexistent")
        self.assertEqual(results, [])

    async def test_get_signatures_by_full_name(self):
        """
        get_signatures with fully qualified names returns matching signatures.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        sigs = inspector.get_signatures(["json.dumps", "json.loads"])
        self.assertIsNotNone(sigs["json.dumps"])
        self.assertIn("obj", sigs["json.dumps"])
        self.assertIsNotNone(sigs["json.loads"])

    async def test_get_signatures_by_short_name(self):
        """
        get_signatures with the re-exported alias returns matching entries.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        sigs = inspector.get_signatures(["json.JSONEncoder"])
        self.assertIsNotNone(sigs.get("json.JSONEncoder"))

    async def test_get_signatures_by_alias(self):
        """
        get_signatures with re-exported alias path returns matching entries.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        sigs = inspector.get_signatures(["json.JSONEncoder"])
        self.assertIsNotNone(sigs.get("json.JSONEncoder"))
        self.assertIn("JSONEncoder", sigs["json.JSONEncoder"])

    async def test_get_signatures_unknown_returns_none(self):
        """
        Unknown names should map to None.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        sigs = inspector.get_signatures(["nonexistent_function"])
        self.assertIsNone(sigs["nonexistent_function"])

    async def test_register_multiple_modules(self):
        """
        Registering multiple modules should accumulate entries.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")
        count_after_json = len(inspector.entries)

        await inspector.register("pathlib")
        count_after_both = len(inspector.entries)

        self.assertGreater(count_after_both, count_after_json)
        names = [e.full_name for e in inspector.entries]
        self.assertIn("json.dumps", names)
        self.assertIn("pathlib.Path", names)

    async def test_list_members_of_module(self):
        """
        list_members for a module should return its direct public members.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        members = inspector.list_members("json")
        names = [m["name"] for m in members]
        self.assertIn("dumps", names)
        self.assertIn("loads", names)

    async def test_list_members_of_class(self):
        """
        list_members for a class should return its direct methods and attributes.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("pathlib")

        members = inspector.list_members("pathlib.Path")
        names = [m["name"] for m in members]
        self.assertIn("glob", names)

    async def test_list_members_returns_expected_keys(self):
        """
        Each member dict should contain name, full_name, alias, type, signature, description.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        members = inspector.list_members("json")
        self.assertTrue(len(members) > 0)
        expected_keys = {"name", "full_name", "alias", "type", "signature", "description"}
        for member in members:
            self.assertEqual(set(member.keys()), expected_keys)

    async def test_list_members_description_is_first_line(self):
        """
        description should be the first line of the docstring.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        members = inspector.list_members("json")
        dumps = next(m for m in members if m["name"] == "dumps")
        self.assertTrue(len(dumps["description"]) > 0)
        self.assertNotIn("\n", dumps["description"])

    async def test_list_members_description_ellipsis_for_multiline(self):
        """
        description should end with '...' when the docstring spans multiple lines.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        members = inspector.list_members("json")
        dumps = next(m for m in members if m["name"] == "dumps")
        self.assertTrue(dumps["description"].endswith("..."))

    async def test_list_members_excludes_nested(self):
        """
        list_members should only return direct children, not nested members.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        members = inspector.list_members("json")
        for member in members:
            # full_name after 'json.' should not contain another dot
            suffix = member["full_name"][len("json."):]
            self.assertNotIn(".", suffix, f"nested member returned: {member['full_name']}")

    async def test_list_members_by_alias(self):
        """
        list_members should work with an alias prefix (e.g. json.JSONEncoder).
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        members = inspector.list_members("json.JSONEncoder")
        names = [m["name"] for m in members]
        self.assertTrue(len(members) > 0)
        # Methods of the re-exported class should appear
        self.assertTrue(any(not n.startswith("_") for n in names))

    async def test_list_members_unknown_returns_empty(self):
        """
        list_members for an unknown name should return an empty list.
        """
        inspector = CodeInspector(self._dummy_embed)
        await inspector.register("json")

        members = inspector.list_members("nonexistent")
        self.assertEqual(members, [])

