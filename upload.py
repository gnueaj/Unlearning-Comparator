import torch
import torchvision
import torchvision.transforms as transforms
import os

# Constants
UMAP_DATA_SIZE = 2000  # 저장할 이미지 개수
UNLEARN_SEED = 42      # 랜덤 시드

# CIFAR-10 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,  # True면 학습셋, False면 테스트셋
    download=True, 
    transform=transform
)

# 랜덤 시드 설정
generator = torch.Generator()
generator.manual_seed(UNLEARN_SEED)

# 랜덤 인덱스 생성
indices = torch.randperm(len(dataset), generator=generator)[:UMAP_DATA_SIZE]

# 저장 디렉토리 생성
os.makedirs('cifar10_images', exist_ok=True)

# 이미지 저장
for i, idx in enumerate(indices):
    image, label = dataset[idx]
    image_path = os.path.join('cifar10_images', f'{idx.item()}.png')
    torchvision.utils.save_image(image, image_path)
    if i % 100 == 0:
        print(f'Saved {i} images...')

print(f'Successfully saved {UMAP_DATA_SIZE} images to cifar10_images/')