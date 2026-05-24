from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="ch",
    engine="paddle_static",
    engine_config={
        "device_type": "gpu",
        "cpu_threads": 2,
        "run_mode": "mkldnn",
    },
)

result = ocr.predict("./screenshots/test.png")

for page in result:
    print(page['rec_texts'])
    page.save_to_img(save_path="./output/detection/detection.png")
    page.save_to_json(save_path="./output/detection/res.json")