from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
import io
import uvicorn
import torch
import torch.nn as nn
from torchvision import transforms

class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = CheckImage()
model.load_state_dict(torch.load('garbage.pth', map_location=device))
model.to(device)
model.eval()

garbage_app = FastAPI()

@garbage_app.post('/predict/')
async def garbage(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=404, detail='файл не гайден')
    open_img = Image.open(io.BytesIO(data))
    img_tensor = transform(open_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        result = pred.argmax(dim=1).item()
    return {'class': classes[result]}

if __name__ == '__main__':
    uvicorn.run(garbage_app, host='127.0.0.1', port=8000)