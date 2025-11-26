import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

coarse_labels = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
    'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
    'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
]

fine_labels_map = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
}

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)


def get_label_name(index):
    return dataset.classes[index]


def plot_hierarchy_samples(selected_superclasses):
    fig, axes = plt.subplots(len(selected_superclasses), 6, figsize=(15, 2.5 * len(selected_superclasses)))

    for i, super_cls in enumerate(selected_superclasses):
        sub_classes = fine_labels_map[super_cls]

        ax_text = axes[i, 0]
        ax_text.text(0.5, 0.5, f"SUPER-CLASS:\n{super_cls.upper()}",
                     ha='center', va='center', fontsize=12, fontweight='bold')
        ax_text.axis('off')

        for j, sub_cls_name in enumerate(sub_classes):
            img_tensor = None
            for img, label_idx in dataset:
                if dataset.classes[label_idx] == sub_cls_name:
                    img_tensor = img
                    break

            ax_img = axes[i, j + 1]
            if img_tensor is not None:
                ax_img.imshow(img_tensor.permute(1, 2, 0))
                ax_img.set_title(f"Sub-class:\n{sub_cls_name}", fontsize=10)
            ax_img.axis('off')

    plt.tight_layout()
    plt.savefig("cifar100_hierarchy_demo.pdf", bbox_inches='tight')  # Uloží jako PDF pro LaTeX
    plt.show()


selected = ['fish', 'vehicles 1', 'large carnivores', 'flowers']
plot_hierarchy_samples(selected)