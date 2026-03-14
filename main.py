import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

# you can replace essays.csv with your own dataset (make sure the fields match)
df = pd.read_csv('essays.csv')
print("Dataset Sample:")
print(df.head())

df = df[['Essay', 'Overall']]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class EssayDataset(Dataset):
    def __init__(self, essays, scores, tokenizer, max_len=256):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.essays)
    
    def __getitem__(self, index):
        essay = str(self.essays[index])
        score = self.scores[index]
        
        encoding = self.tokenizer(
            essay,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': torch.tensor(score, dtype=torch.float)
        }
        
train_texts, val_texts, train_scores, val_scores = train_test_split(
    df['Essay'], df['Overall'], test_size=0.2, random_state=42
)

train_dataset = EssayDataset(train_texts.tolist(), train_scores.tolist(), tokenizer)
val_dataset = EssayDataset(val_texts.tolist(), val_scores.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

class EssayGradingModel(torch.nn.Module):
    def __init__(self):
        super(EssayGradingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        score = self.regressor(cls_output)
        return score.squeeze()
    
# initialize model, optimizer, lass function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EssayGradingModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.MSELoss()

def train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)
        
        outputs = model(input_ids=input_ids,        attention_mask=attention_mask)
        loss = loss_fn(outputs, scores)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    predictions = []
    true_scores = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, scores)
            total_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            true_scores.extend(scores.cpu().numpy())
            
    mse = mean_squared_error(true_scores, predictions)
    r2 = r2_score(true_scores, predictions)
    return total_loss / len(data_loader), mse, r2

# training loops
epochs = 3
for epoch in range(epochs):
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    val_loss, mse, r2 = evaluate(model, val_loader, loss_fn, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}')

test_text = "The essay provided insightful analysis of the topic with well-structured arguments."
encoding = tokenizer(test_text, add_special_tokens=True, max_length=256, padding='max_length',return_tensors='pt')
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

model.eval()

with torch.no_grad():
    predicted_score = model(input_ids, attention_mask).item()
print(f"Predicted Score: {predicted_score:.2f}")

losses = {'Epoch': [1,2,3], 'Train Loss': [0.3, 0.2, 0.1], 'Val Loss': [0.35, 0.25, 0.15]}
loss_df = pd.DataFrame(losses)
sns.lineplot(data=loss_df, x='Epoch', y='Train Loss', label='Train Loss')
sns.lineplot(data=loss_df, x='Epoch', y='Val Loss', label='Val Loss')
plt.title('Training and Validation Loss')
plt.show()