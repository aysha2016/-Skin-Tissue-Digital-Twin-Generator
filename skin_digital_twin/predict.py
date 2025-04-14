import torch
from torchvision import transforms
from PIL import Image
from model import UNet

model = UNet(in_channels=3, out_channels=3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

img = Image.open("sample_input.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

out_img = transforms.ToPILImage()(output.squeeze(0))
out_img.save("output_prediction.png")