import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import ConceptualCaptions
from model import ViLBERT

dataset = ConceptualCaptions(image_paths, captions)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ViLBERT(num_labels=2)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for images, input_ids, attention_mask in dataloader:
        optimizer.zero_grad()
        logits = model(input_ids.squeeze(1), attention_mask.squeeze(1), images)
        loss = criterion(logits, labels)  # labels need to be defined or passed
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}')