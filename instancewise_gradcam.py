import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

def get_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
])

# CIFAR-10 훈련 데이터셋 로드
testset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# CIFAR-10 클래스 이름들
class_names = testset.classes

# 특정 인덱스의 이미지 선택
idx = 39938  # 원하는 인덱스로 변경하세요
image, label = testset[idx]
input_tensor = image.unsqueeze(0)  # 배치 차원 추가

# 두 개의 모델 로드
model1 = get_resnet18(num_classes=10)
model2 = get_resnet18(num_classes=10)

# 사전 학습된 가중치 로드 (모델 경로를 실제 파일 위치로 변경하세요)
model1.load_state_dict(torch.load('/home/jaeung/mu-dashboard/backend/uploaded_models/0000.pth', weights_only=True))
model2.load_state_dict(torch.load('/home/jaeung/mu-dashboard/backend/uploaded_models/resnet18_CIFAR10_without_class_6_30epochs_0.01lr.pth', weights_only=True))

model1.eval()
model2.eval()

# Grad-CAM 사용을 위한 라이브러리 임포트
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 대상 레이어 설정 (ResNet-18의 마지막 컨볼루션 레이어)
target_layer1 = model1.layer4[-1]
target_layer2 = model2.layer4[-1]

cam1 = GradCAM(model=model1, target_layers=[target_layer1])
cam2 = GradCAM(model=model2, target_layers=[target_layer2])

# 모델 예측 얻기
with torch.no_grad():
    outputs1 = model1(input_tensor)
    outputs2 = model2(input_tensor)
    predicted_class1 = outputs1.argmax(dim=1).item()
    predicted_class2 = outputs2.argmax(dim=1).item()

# 실제 레이블과 모델의 예측 클래스 이름 얻기
true_class_name = class_names[label]
predicted_class_name1 = class_names[predicted_class1]
predicted_class_name2 = class_names[predicted_class2]

# Grad-CAM을 위한 타겟 설정
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
targets1 = [ClassifierOutputTarget(predicted_class1)]
targets2 = [ClassifierOutputTarget(predicted_class2)]

# CAM 생성
grayscale_cam1 = cam1(input_tensor, targets=targets1)[0, :]
grayscale_cam2 = cam2(input_tensor, targets=targets2)[0, :]

# 두 히트맵의 최대값을 비교하여 큰 값을 기준으로 선택
combined_max = max(grayscale_cam1.max(), grayscale_cam2.max())

# 각 히트맵을 큰 값으로 정규화
normalized_cam1 = grayscale_cam1 / combined_max
normalized_cam2 = grayscale_cam2 / combined_max


# 이미지를 원래 형태로 복원하는 함수
def unnormalize(img):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img.numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# 원본 이미지 얻기
original_image = unnormalize(image)

# 히트맵을 원본 이미지에 투명하게 겹치기 (alpha 값 조절)
alpha = 0.2  # 투명도 (0~1 사이 값)
cam_image1 = show_cam_on_image(original_image, normalized_cam1, use_rgb=True, image_weight=1 - alpha)
cam_image2 = show_cam_on_image(original_image, normalized_cam2, use_rgb=True, image_weight=1 - alpha)

# 결과 시각화 및 저장
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title(f'Original Image\nTrue Class: {true_class_name}')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam_image1)
plt.title(f'Model 1 Grad-CAM\nPrediction: {predicted_class_name1}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cam_image2)
plt.title(f'Model 2 Grad-CAM\nPrediction: {predicted_class_name2}')
plt.axis('off')

plt.tight_layout()
plt.savefig('gradcam_comparison.png')
# plt.show()
