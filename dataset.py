
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_data(image_path, caption):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)

    tokens = tokenizer.encode_plus(caption, max_length=128, truncation=True, padding='max_length', return_tensors="pt")
    return image, tokens['input_ids'], tokens['attention_mask']

class ConceptualCaptions(Dataset):
    def __init__(self, image_paths, captions):
        self.image_paths = image_paths
        self.captions = captions

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return preprocess_data(self.image_paths[idx], self.captions[idx])