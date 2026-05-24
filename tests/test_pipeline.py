import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import build_translation_prompt, format_translation_output, keep_newest_items, translate_messages, translate_text


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class PipelineTests(unittest.TestCase):
    def test_build_translation_prompt_preserves_line_breaks(self):
        prompt = build_translation_prompt("对的\n你们有多少")

        self.assertIn("Preserve the original line breaks", prompt)
        self.assertTrue(prompt.endswith("对的\n你们有多少"))

    @patch("pipeline.urllib.request.urlopen")
    def test_translate_text_calls_ollama_generate_api(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse({"response": "That's right."})

        result = translate_text("对的", model="translategemma")

        self.assertEqual(result, "That's right.")
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "translategemma")
        self.assertEqual(payload["stream"], False)
        self.assertIn("professional Chinese (zh) to English (en) translator", payload["prompt"])
        self.assertIn("对的", payload["prompt"])
        self.assertNotIn("system", payload)

    @patch("pipeline.translate_text")
    def test_translate_messages_only_returns_unseen_blocks(self, mock_translate_text):
        mock_translate_text.side_effect = ["How much progress is there?"]
        messages = [
            {"header": "Web3Go Yesterday at 14:55", "source_text": "哪有进度", "boxes": []},
            {"header": "Web3Go Yesterday at 14:55", "source_text": "哪有进度", "boxes": []},
        ]

        translated = translate_messages(messages, seen_blocks=set(), model="translategemma")

        self.assertEqual(len(translated), 1)
        self.assertEqual(translated[0]["translated_text"], "How much progress is there?")
        self.assertEqual(mock_translate_text.call_count, 1)

    def test_format_translation_output_renders_console_block(self):
        output = format_translation_output(
            {
                "header": "Web3Go Yesterday at 14:55",
                "source_text": "哪有进度",
                "translated_text": "How much progress is there?",
            },
            "2026-05-21 16:30:00",
        )

        self.assertEqual(
            output,
            "HEADER: Web3Go Yesterday at 14:55\n"
            "How much progress is there?\n",
        )

    def test_keep_newest_items_removes_older_files_and_directories(self):
        root = Path("output/test-retention")
        root.mkdir(parents=True, exist_ok=True)

        for index in range(12):
            item = root / f"item-{index:02d}"
            if index % 2:
                item.mkdir(exist_ok=True)
                (item / "artifact.txt").write_text(str(index), encoding="utf-8")
            else:
                item.write_text(str(index), encoding="utf-8")
            timestamp = 1_700_000_000 + index
            item.touch()

        keep_newest_items(root, limit=10)

        remaining = sorted(item.name for item in root.iterdir())
        self.assertEqual(
            remaining,
            [f"item-{index:02d}" for index in range(2, 12)],
        )


if __name__ == "__main__":
    unittest.main()
