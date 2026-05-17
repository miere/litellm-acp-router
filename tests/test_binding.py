import hashlib
import unittest

from litellm_acp_router.binding import (
    parse_binding_strategy,
    resolve_session_key,
)


class ParseBindingStrategyTests(unittest.TestCase):
    def test_prompt_hashing_parses_without_arg(self) -> None:
        self.assertEqual(parse_binding_strategy("prompt_hashing"), ("prompt_hashing", None))

    def test_http_header_extracts_header_name(self) -> None:
        name, arg = parse_binding_strategy("http_header/X-Hermes-Conversation-Id")
        self.assertEqual(name, "http_header")
        self.assertEqual(arg, "X-Hermes-Conversation-Id")

    def test_http_header_without_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_binding_strategy("http_header/")

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_binding_strategy("magic")

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_binding_strategy("")

    def test_non_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_binding_strategy(None)  # type: ignore[arg-type]


class ResolveSessionKeyPromptHashingTests(unittest.TestCase):
    def test_hashes_system_plus_first_user_to_16_chars(self) -> None:
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
            {"role": "user", "content": "again"},
        ]
        key = resolve_session_key("prompt_hashing", messages=messages)
        expected = hashlib.sha256(b"be terse\nping").hexdigest()[:16]
        self.assertEqual(key, expected)
        self.assertEqual(len(key), 16)

    def test_is_deterministic_across_follow_up_turns(self) -> None:
        first = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
        ]
        second = first + [
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        self.assertEqual(
            resolve_session_key("prompt_hashing", messages=first),
            resolve_session_key("prompt_hashing", messages=second),
        )

    def test_handles_missing_system_prompt(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        key = resolve_session_key("prompt_hashing", messages=messages)
        expected = hashlib.sha256(b"\nhello").hexdigest()[:16]
        self.assertEqual(key, expected)

    def test_normalizes_content_list_blocks(self) -> None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        key = resolve_session_key("prompt_hashing", messages=messages)
        expected = hashlib.sha256(b"\nhi").hexdigest()[:16]
        self.assertEqual(key, expected)

    def test_missing_user_message_raises(self) -> None:
        messages = [{"role": "system", "content": "sys"}]
        with self.assertRaises(ValueError):
            resolve_session_key("prompt_hashing", messages=messages)


class ResolveSessionKeyHttpHeaderTests(unittest.TestCase):
    def _req(self, headers):
        return {"headers": headers}

    def test_extracts_header_value_case_insensitively(self) -> None:
        key = resolve_session_key(
            "http_header/X-Hermes-Conversation-Id",
            messages=[],
            proxy_server_request=self._req({"x-hermes-conversation-id": "abc-123"}),
        )
        self.assertEqual(key, "abc-123")

    def test_trims_whitespace_in_header_value(self) -> None:
        key = resolve_session_key(
            "http_header/X-Conv",
            messages=[],
            proxy_server_request=self._req({"X-Conv": "  conv-1  "}),
        )
        self.assertEqual(key, "conv-1")

    def test_missing_header_fails_fast_with_workaround(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_session_key(
                "http_header/X-Conv",
                messages=[],
                proxy_server_request=self._req({"X-Other": "x"}),
            )
        msg = str(ctx.exception)
        self.assertIn("X-Conv", msg)
        self.assertIn("resend", msg.lower())

    def test_empty_header_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            resolve_session_key(
                "http_header/X-Conv",
                messages=[],
                proxy_server_request=self._req({"X-Conv": "   "}),
            )

    def test_missing_proxy_server_request_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            resolve_session_key(
                "http_header/X-Conv",
                messages=[],
                proxy_server_request=None,
            )


if __name__ == "__main__":
    unittest.main()
