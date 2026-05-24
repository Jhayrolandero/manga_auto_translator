import json
import re
from pathlib import Path


HEADER_RE = re.compile(r"(?:Yesterday|Today|at\s*\d{1,2}:\d{2}|\d{1,2}:\d{2})", re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
UI_PREFIXES = ("Search", "Message #", "Official", "01 Excha", "01 Team")
UI_SUBSTRINGS = ("-general",)


def parse_ocr_result(data, min_score=0.6, min_x=350, max_x=1000):
    items = []
    for text, box, score in zip(data["rec_texts"], data["rec_boxes"], data["rec_scores"]):
        text = text.strip()
        if not text or score < min_score:
            continue

        x1, y1, x2, y2 = box
        if x1 < min_x or x2 > max_x:
            continue

        items.append(
            {
                "text": text,
                "box": box,
                "score": score,
                "x1": x1,
                "y1": y1,
            }
        )

    items.sort(key=lambda item: (item["y1"], item["x1"]))

    messages = []
    current = None
    pending_header_prefix = None

    for item in items:
        text = item["text"]

        if text.startswith(UI_PREFIXES) or any(part in text for part in UI_SUBSTRINGS):
            continue

        if HEADER_RE.search(text):
            if pending_header_prefix:
                text = f"{pending_header_prefix} {text}".strip()
                pending_header_prefix = None
            if current and current["source_text"]:
                messages.append(current)
            current = {"header": text, "source_text": "", "boxes": []}
            continue

        if not CJK_RE.search(text):
            pending_header_prefix = text
            continue

        if current is None:
            current = {"header": None, "source_text": "", "boxes": []}

        if current["source_text"]:
            current["source_text"] += "\n"
        current["source_text"] += text
        current["boxes"].append(item["box"])

    if current and current["source_text"]:
        messages.append(current)

    return messages


def write_outputs(messages, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "translation_blocks.json"
    text_path = out_dir / "translation_blocks.txt"
    md_path = out_dir / "translation_blocks.md"

    json_path.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    for message in messages:
        if message["header"]:
            lines.append(message["header"])
        lines.append(message["source_text"])
        lines.append("")
    text_path.write_text("\n".join(lines), encoding="utf-8")

    md_lines = [
        "# Translation Blocks",
        "",
        "Translate each block from Simplified Chinese to English.",
        "Return exactly one translated block for each input block.",
        "Keep the block ids unchanged.",
        "Only translate `SOURCE_TEXT`.",
        "",
    ]
    for index, message in enumerate(messages, start=1):
        md_lines.append(f"## Block {index}")
        md_lines.append("")
        md_lines.append(f"- Header: {message['header'] or 'None'}")
        md_lines.append("- SOURCE_TEXT:")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.append(message["source_text"])
        md_lines.append("```")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path, text_path, md_path


def main():
    input_path = Path("output/detection/res.json")
    out_dir = Path("output/detection")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    messages = parse_ocr_result(data)
    json_path, text_path, md_path = write_outputs(messages, out_dir)

    print(f"Wrote {len(messages)} message blocks")
    print(json_path)
    print(text_path)
    print(md_path)


if __name__ == "__main__":
    main()
