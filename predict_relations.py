
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import Dict, List, Tuple
import pathlib
import json
import argparse
import shutil
import os
import re

model_path = "Qwen/Qwen3-VL-2B-Instruct"
adapter_path = "adapter"
max_image_size = (448, 448)

valid_labels = {"_p", "_c", "_g", "_c!", "_g!"}

filename_pattern = re.compile(
    r"^(?P<id>.+)_(?:(?P<qualifier>c!|g!|p|c|g))?(?P<sequence>\d+)$"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16

base_model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
model.to(device)

processor = AutoProcessor.from_pretrained(model_path)
processor.image_processor.size = {"shortest_edge": 224, "longest_edge": 448}

def resize_image(image_path: str) -> str:
    resize_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "resized")
    os.makedirs(resize_dir, exist_ok=True)

    image_name = pathlib.Path(image_path).stem + ".jpg"
    output_path = os.path.join(resize_dir, image_name)

    if (os.path.exists(output_path)):
        return os.path.abspath(output_path)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail(max_image_size, Image.Resampling.BICUBIC)
        image.save(output_path, "JPEG", quality=95)

    return os.path.abspath(output_path)

def predict(image_paths: List[str], set_id: str) -> List[str]:
    resized_paths = [resize_image(path) for path in image_paths]

    prompt = (
        f"Classify the relationship qualifiers for image set {set_id}. "
        f"There are {len(resized_paths)} images, presented in filename "
        "sequence order. Return one qualifier per image, in the same order, "
        "as a JSON array. Valid qualifiers: _p, _c, _g, _c!, _g!."
    )

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": path} for path in resized_paths] + [{"type": "text", "text": prompt}]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max(32, len(image_paths) * 12), do_sample=False)

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_length:]

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0].strip()

    start = generated_text.find("[")
    end = generated_text.rfind("]")

    labels = json.loads(generated_text[start:end + 1])

    return labels


def process_dir(dir: str):
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) 
             and ".DS_Store" not in f and "predictions.json" not in f]
    obj: Dict[str, List[Tuple[int, str]]] = {}

    for file in files:
        name = pathlib.Path(file).stem
        match = filename_pattern.match(name)

        if not match:
            continue

        set_id = match.group("id")
        sequence = int(match.group("sequence"))
        image_path = os.path.abspath(os.path.join(dir, file))

        if set_id not in obj:
            obj[set_id] = []

        obj[set_id].append((sequence, image_path))

    results: Dict[str, str] = {}
    total_sets = len(obj)

    for i, (set_id, entries) in enumerate(sorted(obj.items()), start=1):
        entries.sort(key=lambda item: (item[0], os.path.basename(item[1]).lower()))
        image_paths = [path for _, path in entries]

        try:
            labels = predict(image_paths, set_id)

            print(f"{i} / {total_sets}: {set_id} -> {labels}")

            for image_path, label in zip(image_paths, labels):
                filename = os.path.basename(image_path)
                results[filename] = label

        except:
            pass

    output_path = os.path.join(dir, "predictions.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    resize_dir = os.path.join(os.path.abspath(dir), "resized")
    
    if os.path.isdir(resize_dir):
        shutil.rmtree(resize_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Relations")
    parser.add_argument("folderpath")
    args = parser.parse_args()
    process_dir(args.folderpath)