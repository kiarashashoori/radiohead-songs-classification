import librosa
from transformers import AutoFeatureExtractor
import torch
import os
from parameters import Parameters as p
from transformers import ASTConfig,ASTForAudioClassification

classes = os.listdir(p.dataset_path+'train/')
label2id = {label: i for i, label in enumerate(classes)}
id2label = {i: label for i, label in enumerate(classes)}


audio, sr = librosa.load(
    "test.mp3",
    sr=16000,
    mono=True
)



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ASTForAudioClassification.from_pretrained(
    p.model_path
)
model.to(device)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}


with torch.no_grad():
    outputs = model(**inputs)

predicted = outputs.logits.argmax(-1).item()

print(id2label[predicted])