import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import urllib.request
url,filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url,filename)
except: urllib.request.urlretrieve(url,filename)
model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
model.eval()
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output = model(input_batch)
print(output[0])
print(torch.nn.functional.softmax(output[0],dim=0))