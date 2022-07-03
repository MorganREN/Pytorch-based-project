from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# 1. How to use Transform

img_path = '../practice_data/train/ants_image/0013035.jpg'
img = Image.open(img_path)

writer = SummaryWriter('logs')

tensor_train = transforms.ToTensor()  # Create a specific tool using transform
tensor_img = tensor_train(img)

writer.add_image('Tensor_img', tensor_img)

writer.close()

