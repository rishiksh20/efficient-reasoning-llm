import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class VerifierModel(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token
        score = self.classifier(cls_embedding)
        return score.squeeze(-1)


def train_verifier(model, tokenizer, candidates, labels, epochs=3, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()
    model.to("cuda")

    for epoch in range(epochs):
        total_loss = 0
        for text, label in zip(candidates, labels):
            enc = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to("cuda")
            target = torch.tensor([label], dtype=torch.float32).to("cuda")

            score = model(enc.input_ids, enc.attention_mask)
            loss = loss_fn(score, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"[Verifier] Epoch {epoch+1} - Loss: {total_loss / len(candidates):.4f}")

    torch.save(model.state_dict(), "verifier_model.pt")
    print("[Verifier] Model trained and saved.")
