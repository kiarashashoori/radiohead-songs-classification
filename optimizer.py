import torch
from transformers import ASTConfig,ASTForAudioClassification
from transformers import AutoFeatureExtractor
from parameters import Parameters as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Train
from datasets import Dataset
from torch.utils.data import DataLoader


class Optimizer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ASTForAudioClassification.from_pretrained(
            p.model_path
        )
        self.model.to(self.device)
    def quantization(self):
        torch.save(
            torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            ).state_dict(),
            "optimizer.pt"
        )
    def knowledge_distillation(self):
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        student_config = ASTConfig.from_pretrained(
            p.model_path
        )


        student_config.num_hidden_layers = 6
        student_config.hidden_size = 384
        student_config.num_attention_heads = 6
        student_config.intermediate_size = 1536

        student_config.num_labels = p.classes_num
        student_config.label2id = p.label2id
        student_config.id2label = p.id2label
        

        student = ASTForAudioClassification(student_config)
        student.to(self.device)

        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

        kl_loss = nn.KLDivLoss(reduction="batchmean")

        T = 3
        alpha = 0.5

        train_class = Train()
        train_class.label2id = p.label2id
        train_data = train_class.load_data('train')
        train_ds = Dataset.from_list(train_data)
        train_class.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        train_ds = train_ds.map( lambda x: train_class.preprocess(x, augment=True),
                                batched=False)
        train_ds.set_format(type="torch")
        
        dataloader = DataLoader(train_ds,batch_size=32,shuffle=True)


        student.train()
        num_epoch = 100
        for epoch in range(num_epoch):
            print('training started')
            total_loss = 0
            best_loss = None

            for batch in dataloader:

                inputs = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.no_grad():
                    teacher_outputs = self.model(inputs)
                    teacher_logits = teacher_outputs.logits

                student_outputs = student(inputs)
                student_logits = student_outputs.logits

                hard_loss = F.cross_entropy(student_logits, labels)

                soft_loss = kl_loss(
                    F.log_softmax(student_logits / T,dim=1),
                    F.softmax(teacher_logits / T,dim=1)
                )

                loss = alpha * hard_loss + (1 - alpha) * soft_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss
            print(f'epoch : {epoch+1}  , loss = {total_loss}')
            if (best_loss == None or best_loss >= total_loss):
                save_path = f"ast_student/ast-student-epoch-best"
                student.save_pretrained(save_path)
            save_path = f"ast_student/ast-student-epoch-{epoch+1}"
            student.save_pretrained(save_path)

optimizer = Optimizer()
optimizer.knowledge_distillation()


