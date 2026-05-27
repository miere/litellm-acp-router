import os
import unittest
from unittest.mock import patch

from litellm_acp_router.adapters import AuggieAdapter, KimiAdapter
from litellm_acp_router.registry import Registry


_AUGGIE_ENV_CLEAR = {
    k: v for k, v in os.environ.items() if not k.startswith("AUGGIE_")
}


class AdapterTests(unittest.TestCase):
    def test_auggie_omits_model_flag_by_default(self) -> None:
        with patch.dict(os.environ, _AUGGIE_ENV_CLEAR, clear=True):
            spec = AuggieAdapter().build_spec({})

        self.assertEqual(spec.bin, "auggie")
        self.assertEqual(
            spec.args,
            ["--acp", "--allow-indexing"],
        )
        self.assertIsNone(spec.mode_id)

    def test_auggie_uses_generic_acp_model_parameter(self) -> None:
        with patch.dict(os.environ, _AUGGIE_ENV_CLEAR, clear=True):
            spec = AuggieAdapter().build_spec({"acp_model": "gpt-5.5"})

        self.assertEqual(
            spec.args,
            [
                "--acp",
                "--allow-indexing",
                "--model",
                "gpt-5.5",
            ],
        )

    def test_kimi_defaults_are_unchanged(self) -> None:
        spec = KimiAdapter().build_spec({})

        self.assertEqual(spec.bin, "kimi")
        self.assertEqual(spec.args, ["acp"])
        self.assertEqual(spec.mode_id, "code")
        self.assertEqual(spec.bootstrap_commands, ["/plan off", "/yolo"])

    def test_registry_resolves_acp_model_names(self) -> None:
        registry = Registry(default_agent="kimi")
        registry.register(KimiAdapter())
        registry.register(AuggieAdapter())

        self.assertIsInstance(registry.resolve("acp/kimi", {}), KimiAdapter)
        self.assertIsInstance(registry.resolve("acp/auggie", {}), AuggieAdapter)


if __name__ == "__main__":
    unittest.main()
