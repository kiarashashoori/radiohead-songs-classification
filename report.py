from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import ASTConfig,ASTForAudioClassification
from transformers import AutoFeatureExtractor
from parameters import Parameters as p
import numpy as np
# from train import Train
import torch
import os
import librosa
import matplotlib.pyplot as plt



class ModelReport():
    def __init__(self):
        self.classes = os.listdir(p.dataset_path+'train/')
        self.label2id = {label: i for i, label in enumerate(self.classes)}
        id2label = {i: label for i, label in enumerate(self.classes)}


        # audio, sr = librosa.load(
        #     "test.mp3",
        #     sr=16000,
        #     mono=True
        # )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        

        self.model = ASTForAudioClassification.from_pretrained(
            p.model_path
        )
        self.model.to(self.device)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
    
    def get_predictions(self):
        y_true = []
        y_pred = []

        val_path = p.dataset_path + "val/"

        for class_name in os.listdir(val_path):

            class_path = os.path.join(val_path, class_name)

            for file in os.listdir(class_path):

                if not file.endswith(p.music_formats):
                    continue

                file_path = os.path.join(class_path, file)

                # actual label
                y_true.append(self.label2id[class_name])

                # load audio
                audio, sr = librosa.load(
                    file_path,
                    sr=16000,
                    mono=True
                )

                # preprocess
                inputs = self.feature_extractor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                )

                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                }

                # inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                prediction = outputs.logits.argmax(-1).item()

                y_pred.append(prediction)

        return y_pred,y_true
    def generate_results(self):
        y_pred,y_true = self.get_predictions()
        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.classes
        )

        fig, ax = plt.subplots(figsize=(len(self.classes), len(self.classes)))
        disp.plot(ax=ax, xticks_rotation=90)

        plt.tight_layout()
        plt.savefig("result/confusion_matrix.png")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred
        )

        x = np.arange(len(self.classes))

        plt.figure(figsize=(15, 6))
        plt.bar(x - 0.25, precision, 0.25, label="Precision")
        plt.bar(x, recall, 0.25, label="Recall")
        plt.bar(x + 0.25, f1, 0.25, label="F1")

        plt.xticks(x, self.classes, rotation=90)
        plt.legend()

        plt.tight_layout()
        plt.savefig("result/class_metrics.png")
        acc = accuracy_score(y_true,y_pred)
        print(f'accuracy : {acc}')

report = ModelReport()
report.generate_results()


