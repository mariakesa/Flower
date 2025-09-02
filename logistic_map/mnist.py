# Re-run after kernel reset: repeat all definitions and rerun training
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
class LogisticIteratorLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, k=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.r_unbounded = nn.Parameter(torch.tensor(4.0).logit())  # bounded later
        self.k = k
        self.norm = nn.LayerNorm(hidden_dim)

    def r(self):
        return 3.5 + 0.45 * torch.sigmoid(self.r_unbounded)

    def logistic_iter(self, x):
        for _ in range(self.k):
            x = torch.clamp(x, 1e-4, 1 - 1e-4)
            x = self.r() * x * (1 - x)
        return x

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))  # ensure in [0,1]
        x = self.norm(x)
        x = self.logistic_iter(x)
        return x

class FlowerMNISTClassifier(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, k=2):
        super().__init__()
        self.flower1 = LogisticIteratorLayer(input_dim, hidden_dim, k=k)
        self.flower2 = LogisticIteratorLayer(hidden_dim, hidden_dim, k=k)
        self.output = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.flower1(x)
        x = self.flower2(x)
        return self.output(x)

# ─────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=256, shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowerMNISTClassifier(k=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_log = []
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            train_log.append((batch_idx, loss.item(), model.flower1.r().item()))
            if batch_idx >= 400:
                break
train_log

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total

# Run evaluation and print test accuracy
test_accuracy = evaluate(model, test_loader)
print(f"\n🌼 Test Accuracy: {test_accuracy * 100:.2f}%")