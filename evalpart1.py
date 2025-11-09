# evalPart1_fixed.py
'''
cache_dir = input("Please enter the location of cache dir:")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig

import requests
from PIL import Image
import torch
'''

# ------------------ REPLACEMENT BLOCK (use this exact block) ------------------
cache_dir = input("Please enter the location of cache dir:")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
import os

# Replace the old HF-download block with exactly this (no extra lines)
local_peft_dir = "/content/drive/MyDrive/MCADS2/saved_peft/Part1-blip2"
if not os.path.exists(local_peft_dir):
    raise FileNotFoundError(f"Local PEFT folder not found: {local_peft_dir}")

config = PeftConfig.from_pretrained(local_peft_dir)
print("Loaded PeftConfig from local folder:", local_peft_dir)

base_model_name = getattr(config, "base_model_name_or_path", None)
if base_model_name is None:
    base_model_name = "Salesforce/blip2-flan-t5-xl"
print("Using base model:", base_model_name)

base = Blip2ForConditionalGeneration.from_pretrained(
    base_model_name,
    cache_dir=cache_dir,
    load_in_8bit=True, # Added this line to match training setup
    device_map="auto"
)

offload_dir = "/tmp/offload_faithd2t"
os.makedirs(offload_dir, exist_ok=True)

adapter_name = None  # set to "adapter_model_rekeyed.bin" if you created a rekeyed file

if adapter_name:
    model = PeftModel.from_pretrained(base, local_peft_dir, adapter_name=adapter_name, device_map="auto", offload_folder=offload_dir)
else:
    model = PeftModel.from_pretrained(base, local_peft_dir, device_map="auto", offload_folder=offload_dir)

model.eval()

'''
peft_model_id = "Part1-blip2-saved-modelFlanT5-XL"
config = PeftConfig.from_pretrained(peft_model_id)

# load base model (8-bit quantization requires bitsandbytes installed)
model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path,
                                                      load_in_8bit=True,
                                                      cache_dir=cache_dir,
                                                      device_map="auto")

# load LoRA weights
model = PeftModel.from_pretrained(model, peft_model_id, cache_dir=cache_dir, device_map="auto")
model.eval()
'''

processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir=cache_dir)
'''
# read image list (existing code)
GraphImageArr = []
with open('FullLinearGraphPicture.txt','r') as fileImg:
    LinesImg = fileImg.readlines()

img=[]
index = -1
for limg in LinesImg:
    if limg.startswith("Sample") or limg.startswith("END"):
        if index != -1:
            GraphImageArr.append(img)
        img=[]
        index=index+1
        continue
    else:
        if limg.startswith("http://img.youtube.com"):
            continue
        if "mediaviewer_v3" in limg:
            continue
        img.append(limg.strip())

print("Length of GraphImage Array:"+str(len(GraphImageArr)))
'''
# --- Robust loading of FullLinearGraphPicture.txt (replace the old open(...) block) ---
import os

# 1) Try to find the file in a few likely spots
fname = "FullLinearGraphPicture.txt"
candidates = [
    fname,
    os.path.join("/content", fname),
    os.path.join("/content/drive/MyDrive/MCADS2/Real-estate House Dataset", fname),
    os.path.join("/content/drive/MyDrive", fname),
     os.path.join("/content/drive/MyDrive/MCADS2", fname),
    os.path.join(os.getcwd(), fname),
]

found_path = None
for p in candidates:
    if os.path.exists(p):
        found_path = p
        break

if found_path is None:
    # show helpful diagnostics and stop so you can place the file in one of the shown locations
    print("ERROR: Could not find", fname)
    print("Checked these locations:")
    for p in candidates:
        print(" -", p, " ->", "FOUND" if os.path.exists(p) else "missing")
    # also show current directory contents to help you locate the file
    print("\nFiles in current directory (top 200):")
    try:
        for i,fn in enumerate(sorted(os.listdir("."))):
            print(i, fn)
            if i >= 199:
                break
    except Exception as e:
        print("Could not list cwd:", e)
    # show Drive folder contents if available
    drive_dir = "/content/drive/MyDrive/MCADS2/Real-estate House Dataset"
    if os.path.exists(drive_dir):
        print("\nFiles in", drive_dir, "(top 200):")
        try:
            for i,fn in enumerate(sorted(os.listdir(drive_dir))):
                print(i, fn)
                if i >= 199:
                    break
        except Exception as e:
            print("Could not list drive dir:", e)
    # Helpful instruction and abort
    raise FileNotFoundError(f"Place {fname} in one of the printed locations and re-run. If you want, you can upload it to /content/drive/MyDrive/FaithD2T-main/")

# 2) Load file lines
print("Loading image list file from:", found_path)
with open(found_path, "r", encoding="utf-8", errors="ignore") as fileImg:
    LinesImg = fileImg.readlines()

# 3) Parse file into GraphImageArr (same format logic as before)
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

# if last block didn't end with END marker, append the last one
if img:
    GraphImageArr.append(img)

print("Length of GraphImage Array:", len(GraphImageArr))


# features reading code unchanged (ensure file exists)
fileFeatures = open("Second100_GraphFeaturesFiltered_MiniGPT4Summary.txt","r",encoding='ISO-8859-1')
LinesF = fileFeatures.readlines()
Feature=[]
for line in LinesF:
    if line != "\n":
        if line.startswith("Sample"):
            listFeature = []
        elif line.startswith("END"):
            Feature.append(listFeature)
        else:
            listFeature.append(line.strip())
fileFeatures.close()

print("Total Sample Features: "+str(len(Feature)))

# prepare model device (model already loaded with device_map, but ensure model is on device)
# For PEFT loaded with device_map="auto" the parameters are sharded; we still call eval.
model.to(device)

saveResultsFile = "Second100_Mapping_GraphFeaturesFiltered_MiniGPT4.txt"
filePart1Result = open(saveResultsFile,"a")

for id in range(len(Feature)):
    listF = Feature[id]
    # NOTE: original code used GraphImageArr[id+43100] -- ensure index in range
    imgs = GraphImageArr[id+43100]  # keep original indexing if intended
    filePart1Result.write("Sample:\n" + str(id+43100) + "\n")
    for eachF in listF:
        filePart1Result.write(str(eachF) + "::")
        ans=[]
        for eachImg in imgs:
            ques="Is the feature -'{}' 'Salient' or 'Not Salient' or 'Hallucinated'?".format(eachF)
            question = "Question:" + ques + " Short answer:"
            image = Image.open(requests.get(eachImg, stream=True).raw).convert('RGB')

            inputs = processor(image, question, return_tensors="pt")
            # send tensors to device (processor returns tensors on CPU by default)
            inputs = {k: v.to(device) for k,v in inputs.items()}

            out = model.generate(**inputs, max_new_tokens=50)
            # decode using tokenizer
            generated_text = processor.tokenizer.decode(out[0], skip_special_tokens=True)

            print(question)
            print(generated_text)
            ans.append(generated_text)
            if generated_text.strip().startswith("Salient"):
                break

        # decide finalRes (original logic)
        finalRes="Hallucinated"
        if any(a.strip().startswith("Salient") for a in ans):
            finalRes="Salient"
        elif any(a.strip().startswith("Not Salient") for a in ans):
            finalRes="Not Salient"

        print("Final Result:"+str(finalRes))
        filePart1Result.write(str(finalRes) + "\n")
    filePart1Result.write("END\n")

filePart1Result.close()