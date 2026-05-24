import argparse
import json
import shutil
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from time import perf_counter

import mss

from parser import parse_ocr_result, write_outputs


MAX_RETAINED_ITEMS = 10


def build_translation_prompt(source_text):
    return (
        "You are a professional Chinese (zh) to English (en) translator.\n"
        "Preserve the original line breaks.\n"
        "Return only the English translation.\n\n"
        "Please translate the following Chinese text into English:\n"
        f"{source_text}"
    )


def translate_text(source_text, model="translategemma", ollama_url="http://127.0.0.1:11434/api/generate", timeout=120):
    payload = {
        "model": model,
        "prompt": build_translation_prompt(source_text),
        "stream": False,
    }
    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["response"].strip()


def translate_messages(messages, seen_blocks, model="translategemma-fast", ollama_url="http://127.0.0.1:11434/api/generate", timeout=120):
    translated = []
    for message in messages:
        block_key = f"{message.get('header') or ''}\n{message['source_text']}"
        if block_key in seen_blocks:
            continue

        started_at = perf_counter()
        translated_text = translate_text(
            message["source_text"],
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
        )
        elapsed = perf_counter() - started_at
        seen_blocks.add(block_key)
        print(
            f"translated block in {elapsed:.2f}s | "
            f"header={message.get('header') or 'None'}"
        )
        translated.append(
            {
                "header": message.get("header"),
                "source_text": message["source_text"],
                "translated_text": translated_text,
            }
        )
    return translated


def format_translation_output(message, timestamp):
    header = message.get("header") or "None"
    return f"HEADER: {header}\n{message['translated_text']}\n"


def build_ocr(device_type="gpu"):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="ch",
        engine="paddle_static",
        engine_config={
            "device_type": device_type,
            "cpu_threads": 2,
            "run_mode": "mkldnn",
        },
    )


def capture_screenshot(sct, output_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = output_dir / f"screenshot_{timestamp}.png"
    sct.shot(mon=0, output=str(image_path))
    return timestamp, image_path


def run_ocr(ocr, image_path, run_dir):
    result = ocr.predict(str(image_path))
    page = result[0]
    ocr_json_path = run_dir / "ocr.json"
    page.save_to_json(save_path=str(ocr_json_path))
    return json.loads(ocr_json_path.read_text(encoding="utf-8"))


def process_screenshot(ocr, image_path, run_dir, seen_blocks, model, ollama_url, timeout):
    ocr_data = run_ocr(ocr, image_path, run_dir)
    messages = parse_ocr_result(ocr_data)
    write_outputs(messages, run_dir)
    return translate_messages(
        messages,
        seen_blocks=seen_blocks,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
    )


def print_translations(translated_messages, timestamp):
    for message in translated_messages:
        print(format_translation_output(message, timestamp))


def keep_newest_items(path, limit=MAX_RETAINED_ITEMS):
    path = Path(path)
    if not path.exists():
        return

    items = sorted(
        [item for item in path.iterdir() if item.is_file() or item.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for item in items[limit:]:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--interval", type=int, default=5)
    arg_parser.add_argument("--model", default="translategemma")
    arg_parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    arg_parser.add_argument("--device-type", default="gpu")
    arg_parser.add_argument("--timeout", type=int, default=120)
    args = arg_parser.parse_args()

    screenshots_dir = Path("screenshots")
    runs_dir = Path("output/runs")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    ocr = build_ocr(device_type=args.device_type)
    seen_blocks = set()

    with mss.mss() as sct:
        while True:
            loop_started_at = perf_counter()
            timestamp, image_path = capture_screenshot(sct, screenshots_dir)
            run_dir = runs_dir / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                capture_seconds = perf_counter() - loop_started_at

                ocr_started_at = perf_counter()
                ocr_data = run_ocr(ocr, image_path, run_dir)
                ocr_seconds = perf_counter() - ocr_started_at

                parse_started_at = perf_counter()
                messages = parse_ocr_result(ocr_data)
                write_outputs(messages, run_dir)
                parse_seconds = perf_counter() - parse_started_at

                translate_started_at = perf_counter()
                translated_messages = translate_messages(
                    messages,
                    seen_blocks=seen_blocks,
                    model=args.model,
                    ollama_url=args.ollama_url,
                    timeout=args.timeout,
                )
                translate_seconds = perf_counter() - translate_started_at
                total_seconds = perf_counter() - loop_started_at

                print(
                    f"[{timestamp.replace('_', ' ')}] "
                    f"capture={capture_seconds:.2f}s "
                    f"ocr={ocr_seconds:.2f}s "
                    f"parse={parse_seconds:.2f}s "
                    f"translate={translate_seconds:.2f}s "
                    f"total={total_seconds:.2f}s "
                    f"messages={len(messages)} "
                    f"translated={len(translated_messages)}"
                )
                if translated_messages:
                    print_translations(translated_messages, timestamp.replace("_", " "))
            except Exception as exc:
                print(f"[{timestamp.replace('_', ' ')}] pipeline error: {exc}")

            keep_newest_items(screenshots_dir)
            keep_newest_items(runs_dir)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
