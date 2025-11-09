# evalPart2_fixed.py
# --- USAGE: set cache dir when prompted. Place necessary text files under one of the shown paths if not found.

import os
import torch
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
from PIL import Image
import requests

# ------------------ BASIC SETUP ------------------
cache_dir = input("Please enter the location of cache dir:")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------ PEFT / BASE MODEL LOADING (local-only, robust) ------------------
local_peft_dir = "/content/drive/MyDrive/FaithD2T-main/saved_peft/Part1-blip2"  # adjust if your adapter folder differs
if not os.path.exists(local_peft_dir):
    raise FileNotFoundError(f"Local PEFT folder not found: {local_peft_dir}")

config = PeftConfig.from_pretrained(local_peft_dir)
print("Loaded PeftConfig from local folder:", local_peft_dir)

base_model_name = getattr(config, "base_model_name_or_path", None)
if base_model_name is None:
    base_model_name = "Salesforce/blip2-flan-t5-xl"
print("Using base model:", base_model_name)

# load base model (match training quantization if used)
base = Blip2ForConditionalGeneration.from_pretrained(
    base_model_name,
    cache_dir=cache_dir,
    load_in_8bit=True,
    device_map="auto"
)

# offload directory to help large models
offload_dir = "/tmp/offload_faithd2t"
os.makedirs(offload_dir, exist_ok=True)

# If you produced a rekeyed adapter file, set adapter_name to that filename; otherwise keep None.
adapter_name = None

if adapter_name:
    model = PeftModel.from_pretrained(base, local_peft_dir, adapter_name=adapter_name, device_map="auto", offload_folder=offload_dir)
else:
    model = PeftModel.from_pretrained(base, local_peft_dir, device_map="auto", offload_folder=offload_dir)

model.eval()
model.to(device)

# ------------------ PROCESSOR ------------------
processor = AutoProcessor.from_pretrained(base_model_name, cache_dir=cache_dir)

# ------------------ ROBUST FILE FINDING (example: for eval part2 files) ------------------
def find_file(fname, extra_candidates=None):
    candidates = [fname,
                  os.path.join("/content", fname),
                  os.path.join("/content/drive/MyDrive/FaithD2T-main", fname),
                  os.path.join("/content/drive/MyDrive", fname),
                  os.path.join(os.getcwd(), fname)]
    if extra_candidates:
        candidates.extend(extra_candidates)
    for p in candidates:
        if os.path.exists(p):
            return p
    # Diagnostic printout
    print("ERROR: Could not find", fname)
    print("Checked these locations:")
    for p in candidates:
        print(" -", p, " ->", "FOUND" if os.path.exists(p) else "missing")
    # show current dir contents (helpful)
    try:
        print("\nFiles in current directory (top 200):")
        for i,fn in enumerate(sorted(os.listdir("."))):
            print(i, fn)
            if i >= 199:
                break
    except Exception as e:
        print("Could not list cwd:", e)
    drive_dir = "/content/drive/MyDrive/FaithD2T-main"
    if os.path.exists(drive_dir):
        print("\nFiles in", drive_dir, "(top 200):")
        try:
            for i,fn in enumerate(sorted(os.listdir(drive_dir))):
                print(i, fn)
                if i >= 199:
                    break
        except Exception as e:
            print("Could not list drive dir:", e)
    raise FileNotFoundError(f"Place {fname} in one of the printed locations and re-run.")

# Example files used by evalPart2 (update names as per your original script)
file_imgs_name = "FullLinearGraphPicture.txt"
file_features_name = "Second100_GraphFeaturesFiltered_MiniGPT4Summary.txt"

# Find & load image list
found_img_path = find_file(file_imgs_name)
print("Loading image list from:", found_img_path)
with open(found_img_path, "r", encoding="utf-8", errors="ignore") as f:
    LinesImg = f.readlines()

GraphImageArr = []
img = []
index = -1
for limg in LinesImg:
    limg = limg.rstrip("\n")
    if limg.startswith("Sample") or limg.startswith("END"):
        if index != -1:
            GraphImageArr.append(img)
        img = []
        index += 1
        continue
    else:
        if limg.startswith("http://img.youtube.com"):
            continue
        if "mediaviewer_v3" in limg:
            continue
        if limg.strip() == "":
            continue
        img.append(limg.strip())
if img:
    GraphImageArr.append(img)
print("Length of GraphImage Array:", len(GraphImageArr))

# Find & load features file
found_feat_path = find_file(file_features_name)
print("Loading feature list from:", found_feat_path)
with open(found_feat_path, "r", encoding='ISO-8859-1', errors="ignore") as fileFeatures:
    LinesF = fileFeatures.readlines()

Feature = []
listFeature = []
for line in LinesF:
    if line != "\n":
        if line.startswith("Sample"):
            listFeature = []
        elif line.startswith("END"):
            Feature.append(listFeature)
        else:
            listFeature.append(line.strip())

print("Total Sample Features:", len(Feature))

# ------------------ EVAL LOOP (adapt logic to your evalPart2 needs) ------------------
saveResultsFile = "Second100_Mapping_GraphFeaturesFiltered_MiniGPT4_evalPart2.txt"
filePart2Result = open(saveResultsFile, "a", encoding="utf-8")

# NOTE: original code used offset indexing (e.g., +43100). Keep same if intended.
index_offset = 43100

for id in range(len(Feature)):
    listF = Feature[id]
    # ensure index exists in GraphImageArr
    graph_idx = id + index_offset
    if graph_idx < 0 or graph_idx >= len(GraphImageArr):
        print(f"Skipping sample {graph_idx}: index out of range for GraphImageArr (len={len(GraphImageArr)})")
        filePart2Result.write("Sample:\n" + str(graph_idx) + "\n")
        filePart2Result.write("MISSING_IMAGES\nEND\n")
        continue

    imgs = GraphImageArr[graph_idx]
    filePart2Result.write("Sample:\n" + str(graph_idx) + "\n")
    for eachF in listF:
        filePart2Result.write(str(eachF) + "::")
        ans = []
        for eachImg in imgs:
            try:
                ques = "Is the feature -'{}' 'Salient' or 'Not Salient' or 'Hallucinated'?".format(eachF)
                question = "Question:" + ques + " Short answer:"
                # fetch image robustly
                try:
                    image = Image.open(requests.get(eachImg, stream=True, timeout=10).raw).convert('RGB')
                except Exception as e:
                    print(f"Could not load image {eachImg}: {e}")
                    continue

                inputs = processor(image, question, return_tensors="pt")
                # move tensors to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                out = model.generate(**inputs, max_new_tokens=50)
                generated_text = processor.tokenizer.decode(out[0], skip_special_tokens=True)

                print(question)
                print(generated_text)
                ans.append(generated_text)
                if generated_text.strip().startswith("Salient"):
                    break
            except Exception as e:
                print("Error during model inference for image:", eachImg, "error:", e)
                continue

        # decide finalRes (same logic as evalPart1)
        finalRes = "Hallucinated"
        if any(a.strip().startswith("Salient") for a in ans):
            finalRes = "Salient"
        elif any(a.strip().startswith("Not Salient") for a in ans):
            finalRes = "Not Salient"

        print("Final Result for feature:", finalRes)
        filePart2Result.write(str(finalRes) + "\n")

    filePart2Result.write("END\n")

filePart2Result.close()
print("Done. Results appended to", saveResultsFile)