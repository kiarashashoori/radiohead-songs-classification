from transformers import ASTConfig,ASTForAudioClassification
import os
from parameters import Parameters as p
import numpy as np
from transformers import Trainer
from transformers import AutoFeatureExtractor
import librosa
from transformers import TrainingArguments
from datasets import Dataset
from audiomentations import Compose, AddGaussianSNR, GainTransition, Gain, ClippingDistortion, TimeStretch, PitchShift

class Train():
    def __init__(self):
        pass
    def load_data(self,split):
        data = []

        base_path = os.path.join(p.dataset_path, split)

        for label in os.listdir(base_path):
            label_path = os.path.join(base_path, label)

            for file in os.listdir(label_path):
                if file.endswith(".mp3") or file.endswith(".m4a"):
                    data.append({
                        "path": os.path.join(label_path, file),
                        "label": self.label2id[label]
                    })

        return data

    def preprocess(self,example , augment = False):
        audio, sr = librosa.load(example["path"], sr=16000,mono=True)
        if augment:
            audio_augmentations = Compose([
            AddGaussianSNR(min_snr_db=10, max_snr_db=20),
            Gain(min_gain_db=-6, max_gain_db=6),
            GainTransition(min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3, duration_unit="fraction"),
            ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.2),
            PitchShift(min_semitones=-4, max_semitones=4),
            ], p=0.8, shuffle=True)
            audio = audio_augmentations(samples=audio, sample_rate=16000)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000
        )

        return {
            "input_values": inputs["input_values"][0],
            "labels": example["label"]
        }
    
    def training(self):

        self.classes = os.listdir(p.dataset_path+'train/')
        self.label2id = {label: i for i, label in enumerate(self.classes)}
        self.id2label = {i: label for i, label in enumerate(self.classes)}


        config = ASTConfig.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        config.num_labels = len(self.classes)
        config.label2id = self.label2id
        config.id2label = self.id2label


        model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            config=config,
            ignore_mismatched_sizes=True
        )

        model.to('cuda')

        train_data = self.load_data('train')
        val_data = self.load_data('val')
        train_ds = Dataset.from_list(train_data)
        val_ds = Dataset.from_list(val_data)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        train_ds = train_ds.map( lambda x: self.preprocess(x, augment=True))
        val_ds = val_ds.map( lambda x: self.preprocess(x, augment=False))


        train_ds = train_ds.remove_columns(["path"])
        val_ds = val_ds.remove_columns(["path"])
        training_args = TrainingArguments(
                output_dir="./ast-radiohead",
            learning_rate=1e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True
        )



        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        trainer.train()
if __name__ == '__main__':
    train = Train()
    train.training()