import json
import unittest
from pathlib import Path

from parser import parse_ocr_result, write_outputs


class ParserTests(unittest.TestCase):
    def test_parse_ocr_result_groups_center_chat_messages(self):
        data = {
            "rec_texts": [
                "Official",
                "Web3Go ×01 Yesterday at 18:47",
                "积分返利就是邀请的吗",
                "二狗子Dogquant X01 Yesterday at 20:05",
                "对的",
                "你们有多少",
                "不知道什么时候能领取",
                "Message #中文-general",
                "Rexy Yesterday at 23:37",
                "出来看吗，这出了直接能领取才是啊",
            ],
            "rec_boxes": [
                [48, 69, 139, 93],
                [420, 456, 718, 476],
                [420, 482, 661, 510],
                [426, 543, 813, 567],
                [424, 572, 478, 599],
                [427, 607, 552, 634],
                [428, 639, 704, 671],
                [462, 1008, 702, 1038],
                [524, 737, 758, 761],
                [444, 762, 826, 793],
            ],
            "rec_scores": [0.99] * 10,
        }

        messages = parse_ocr_result(data)

        self.assertEqual(
            messages,
            [
                {
                    "header": "Web3Go ×01 Yesterday at 18:47",
                    "source_text": "积分返利就是邀请的吗",
                    "boxes": [[420, 482, 661, 510]],
                },
                {
                    "header": "二狗子Dogquant X01 Yesterday at 20:05",
                    "source_text": "对的\n你们有多少\n不知道什么时候能领取",
                    "boxes": [
                        [424, 572, 478, 599],
                        [427, 607, 552, 634],
                        [428, 639, 704, 671],
                    ],
                },
                {
                    "header": "Rexy Yesterday at 23:37",
                    "source_text": "出来看吗，这出了直接能领取才是啊",
                    "boxes": [[444, 762, 826, 793]],
                },
            ],
        )

    def test_write_outputs_emits_json_and_text_files(self):
        messages = [
            {
                "header": "Web3Go ×01 Yesterday at 18:47",
                "source_text": "积分返利就是邀请的吗",
                "boxes": [[420, 482, 661, 510]],
            }
        ]
        out_dir = Path("output/test-parser")

        json_path, text_path, md_path = write_outputs(messages, out_dir)

        self.assertTrue(json_path.exists())
        self.assertTrue(text_path.exists())
        self.assertTrue(md_path.exists())
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), messages)
        self.assertEqual(
            text_path.read_text(encoding="utf-8"),
            "Web3Go ×01 Yesterday at 18:47\n积分返利就是邀请的吗\n",
        )
        self.assertEqual(
            md_path.read_text(encoding="utf-8"),
            "# Translation Blocks\n\n"
            "Translate each block from Simplified Chinese to English.\n"
            "Return exactly one translated block for each input block.\n"
            "Keep the block ids unchanged.\n"
            "Only translate `SOURCE_TEXT`.\n\n"
            "## Block 1\n\n"
            "- Header: Web3Go ×01 Yesterday at 18:47\n"
            "- SOURCE_TEXT:\n\n"
            "```text\n"
            "积分返利就是邀请的吗\n"
            "```\n",
        )

    def test_parse_ocr_result_combines_split_headers_and_skips_channel_title(self):
        data = {
            "rec_texts": [
                "中文-general",
                "Web3Go ×01",
                "Yesterday at 18:47",
                "积分返利就是邀请的吗",
            ],
            "rec_boxes": [
                [379, 17, 520, 47],
                [420, 456, 567, 476],
                [576, 457, 718, 476],
                [420, 482, 661, 510],
            ],
            "rec_scores": [0.99] * 4,
        }

        messages = parse_ocr_result(data)

        self.assertEqual(
            messages,
            [
                {
                    "header": "Web3Go ×01 Yesterday at 18:47",
                    "source_text": "积分返利就是邀请的吗",
                    "boxes": [[420, 482, 661, 510]],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
