import torch
import torch.nn as nn
from torchvision import models
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch_cka import CKA

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_resnet18(num_classes=10):
    model = models.resnet18(weights='DEFAULT')
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def load_resnet18_model(weights_path):
    model = get_resnet18().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def load_cifar10_data(batch_size=256, class_to_separate=6):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    class_6_indices = [i for i, (_, label) in enumerate(testset) if label == class_to_separate]
    other_indices = [i for i, (_, label) in enumerate(testset) if label != class_to_separate]
    
    class_6_subset = torch.utils.data.Subset(testset, class_6_indices)
    other_subset = torch.utils.data.Subset(testset, other_indices)
    
    class_6_loader = torch.utils.data.DataLoader(class_6_subset, batch_size=batch_size, shuffle=False)
    other_loader = torch.utils.data.DataLoader(other_subset, batch_size=batch_size, shuffle=False)
    
    return class_6_loader, other_loader

weights_path1 = '/home/jaeung/mu-dashboard/backend/trained_models/0000.pth'
weights_path2 = '/home/jaeung/mu-dashboard/backend/unlearned_models/resnet18_CIFAR10_GA_forget_class_6_5epochs_0.0001lr.pth'

model1 = load_resnet18_model(weights_path1)
model2 = load_resnet18_model(weights_path2)

class_6_loader, other_loader = load_cifar10_data()

detailed_layers1 = [
    'conv1',
    # 'bn1',
    # 'relu',
    # 'maxpool',
    # 'layer1',
    # 'layer2',
    # 'layer3',
    # 'layer4',
    'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
    'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2',
    'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2',
    'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2',
    # 'avgpool',
    'fc'
]

detailed_layers2 = [
    'conv1',
    # 'bn1',
    # 'relu',
    # 'maxpool',
    # 'layer1',
    # 'layer2',
    # 'layer3',
    # 'layer4',
    'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
    'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2',
    'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2',
    'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2',
    # 'avgpool',
    'fc'
]

def calculate_and_visualize_cka(loader):
    cka = CKA(model1, 
              model2, 
              model1_name="ResNet18_1", 
              model2_name="ResNet18_2",
              model1_layers=detailed_layers1, 
              model2_layers=detailed_layers2, 
              device=device)
    cka.compare(loader)
    results = cka.export()
    return results['CKA']

cka_matrix_class6 = calculate_and_visualize_cka(class_6_loader)
cka_matrix_others = calculate_and_visualize_cka(other_loader)

print(cka_matrix_class6)
print(cka_matrix_others)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

im1 = ax1.imshow(cka_matrix_class6, cmap='magma', interpolation='none', origin='upper')
ax1.set_title("CKA similarity (Class 6)", fontsize=16)
ax1.set_xlabel('After Unlearning', fontsize=12)
ax1.set_ylabel('Before Unlearning', fontsize=12)
ax1.set_xticks(range(len(detailed_layers2)))
ax1.set_yticks(range(len(detailed_layers1)))
ax1.set_xticklabels(detailed_layers2, rotation=90, ha='right', fontsize=8)
ax1.set_yticklabels(detailed_layers1, fontsize=8)

im2 = ax2.imshow(cka_matrix_others, cmap='magma', interpolation='none', origin='upper')
ax2.set_title("CKA similarity (Other Classes)", fontsize=16)
ax2.set_xlabel('After Unlearning', fontsize=12)
ax2.set_ylabel('Before Unlearning', fontsize=12)
ax2.set_xticks(range(len(detailed_layers2)))
ax2.set_yticks(range(len(detailed_layers1)))
ax2.set_xticklabels(detailed_layers2, rotation=90, ha='right', fontsize=8)
ax2.set_yticklabels(detailed_layers1, fontsize=8)

for ax, matrix in [(ax1, cka_matrix_class6), (ax2, cka_matrix_others)]:
    for i in range(len(detailed_layers1)):
        for j in range(len(detailed_layers2)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", 
                    ha="center", va="center", color="w", fontsize=10)

cbar1 = fig.colorbar(im1, ax=ax1)
cbar1.set_label("CKA similarity", fontsize=12)
cbar2 = fig.colorbar(im2, ax=ax2)
cbar2.set_label("CKA similarity", fontsize=12)

plt.tight_layout()
plt.savefig('/home/jaeung/mu-dashboard/cka_class6_exp.png', dpi=300, bbox_inches='tight')