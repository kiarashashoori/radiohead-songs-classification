import torch
from transformers import ASTConfig,ASTForAudioClassification
from parameters import Parameters as p

class Optimizer():
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ASTForAudioClassification.from_pretrained(
            p.model_path
        )
        self.model.to(device)
    def quantization(self):
        torch.save(
            torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            ).state_dict(),
            "optimizer.pt"
        )
    
optimizer = Optimizer()
optimizer.quantization()


