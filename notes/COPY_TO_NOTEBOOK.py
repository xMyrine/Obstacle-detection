# ============================================================================
# COPY THIS ENTIRE CELL INTO YOUR JUPYTER NOTEBOOK
# ============================================================================

#%% md
"""
### **5.6 Attention-Based Foreground/Background Segmentation**

This technique uses CLS token attention to automatically segment the foreground (object)
from the background. The CLS token in DINO naturally learns to attend to the main object
in the image, so we can use its attention weights to create a segmentation mask that shows:
- **Colored foreground** (what DINO focuses on)
- **Grayscale background** (what DINO ignores)
"""

#%%
import numpy as np
from scipy.ndimage import gaussian_filter

def visualize_attention_segmentation(image, model, processor, device, threshold=0.6, blur_kernel=5):
    """
    Visualize attention-based foreground/background segmentation.

    The CLS token attention is used to create a mask that highlights the main object
    and grays out the background.
    """
    # Convert tensor to PIL if needed
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    # Get attention maps
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.set_attn_implementation('eager')

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention from last layer
    attentions = outputs.attentions[-1]  # [batch, heads, patches, patches]

    # Average across heads and get CLS token attention to all patches
    cls_attention = attentions[0].mean(dim=0)[0, 1:]  # [num_patches]

    original_image = np.array(image)
    h, w = original_image.shape[:2]
    num_patches = cls_attention.shape[0]

    # Calculate grid dimensions
    valid_grids = []
    for hp in range(1, num_patches + 1):
        if num_patches % hp == 0:
            wp = num_patches // hp
            valid_grids.append((hp, wp))
    h_patches, w_patches = min(valid_grids, key=lambda x: abs(x[0] - x[1]))

    print(f"Image size: {w}×{h}, Grid: {h_patches}×{w_patches}, Patches: {num_patches}")
    print(f"Attention stats - min: {cls_attention.min():.4f}, max: {cls_attention.max():.4f}, mean: {cls_attention.mean():.4f}")

    # Reshape to spatial grid
    attention_map = cls_attention.cpu().numpy().reshape(h_patches, w_patches)

    # Normalize to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Upsample to image size
    attention_upsampled = np.array(Image.fromarray((attention_map * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS))
    attention_upsampled = attention_upsampled / 255.0

    # Apply Gaussian blur for smoother edges
    attention_smooth = gaussian_filter(attention_upsampled, sigma=blur_kernel)

    # Create binary mask with threshold
    mask_binary = (attention_smooth > threshold).astype(np.float32)

    # Create different visualizations
    # 1. Grayscale background
    gray_image = np.dot(original_image[...,:3], [0.2989, 0.5870, 0.1140])
    gray_image_rgb = np.stack([gray_image]*3, axis=-1)

    # Blend: keep color where mask is high, gray where low
    masked_color = original_image * attention_smooth[:,:,np.newaxis] + \
                   gray_image_rgb * (1 - attention_smooth[:,:,np.newaxis])
    masked_color = masked_color.astype(np.uint8)

    # 2. Binary segmentation (foreground vs background)
    masked_binary = original_image * mask_binary[:,:,np.newaxis] + \
                    gray_image_rgb * (1 - mask_binary[:,:,np.newaxis])
    masked_binary = masked_binary.astype(np.uint8)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original, Attention Map, Attention Heatmap
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")

    axes[0, 1].imshow(attention_smooth, cmap='jet')
    axes[0, 1].set_title(f"CLS Attention Map\n(Smooth, σ={blur_kernel})", fontsize=12, fontweight='bold')
    axes[0, 1].axis("off")

    axes[0, 2].imshow(original_image)
    axes[0, 2].imshow(attention_smooth, cmap='jet', alpha=0.5)
    axes[0, 2].set_title("Attention Overlay", fontsize=12, fontweight='bold')
    axes[0, 2].axis("off")

    # Row 2: Soft Segmentation, Binary Segmentation, Binary Mask
    axes[1, 0].imshow(masked_color)
    axes[1, 0].set_title("Soft Segmentation\n(Gradual Background Fading)", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")

    axes[1, 1].imshow(masked_binary)
    axes[1, 1].set_title(f"Binary Segmentation\n(Threshold={threshold})", fontsize=12, fontweight='bold')
    axes[1, 1].axis("off")

    axes[1, 2].imshow(mask_binary, cmap='gray')
    axes[1, 2].set_title("Binary Mask", fontsize=12, fontweight='bold')
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    return attention_smooth, mask_binary


def visualize_attention_segmentation_from_url(url, model, processor, device, threshold=0.6, blur_kernel=5,
                                              resize=(224, 224)):
    """
    Download image from URL and visualize attention-based segmentation.
    """
    image = download_image_from_url(url, resize=resize)
    if image is None:
        return None, None

    return visualize_attention_segmentation(image, model, processor, device, threshold, blur_kernel)


#%% md
"""
#### **5.6.1 Examples with Different Animals**
"""

#%%
# Dog - should highlight the dog and gray out background
dog_url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400"
print("=== Dog Segmentation ===")
visualize_attention_segmentation_from_url(dog_url, model, processor, device, threshold=0.6, blur_kernel=5)

#%%
# Cat - should focus on the cat
cat_url = "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400"
print("=== Cat Segmentation ===")
visualize_attention_segmentation_from_url(cat_url, model, processor, device, threshold=0.6, blur_kernel=5)

#%%
# Bird - should highlight the bird
bird_url = "https://images.unsplash.com/photo-1552728089-57bdde30beb3?w=400"
print("=== Bird Segmentation ===")
visualize_attention_segmentation_from_url(bird_url, model, processor, device, threshold=0.6, blur_kernel=5)

#%% md
"""
#### **5.6.2 Adjusting Threshold and Blur**

You can adjust the `threshold` parameter to control how much of the image is considered "foreground":
- **Lower threshold (0.3-0.5)**: More permissive, includes more background
- **Higher threshold (0.6-0.8)**: More strict, focuses tightly on main object

The `blur_kernel` parameter controls edge smoothness:
- **Smaller kernel (1-3)**: Sharp edges
- **Larger kernel (5-10)**: Smooth, gradual transitions
"""

#%%
# Try different thresholds
print("=== Low Threshold (0.4) - More Background ===")
visualize_attention_segmentation_from_url(cat_url, model, processor, device, threshold=0.4, blur_kernel=5)

#%%
print("=== High Threshold (0.7) - Tight Focus ===")
visualize_attention_segmentation_from_url(cat_url, model, processor, device, threshold=0.7, blur_kernel=5)

#%%
print("=== Large Blur (10) - Very Smooth ===")
visualize_attention_segmentation_from_url(cat_url, model, processor, device, threshold=0.6, blur_kernel=10)

#%% md
"""
#### **5.6.3 Apply to Dataset Images**
"""

#%%
# Apply segmentation to a random CIFAR-10 image
idx = random.randint(0, len(dataset_cifar10) - 1)
cifar_image, cifar_label = dataset_cifar10[idx]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"=== CIFAR-10: {class_names[cifar_label]} ===")
visualize_attention_segmentation(cifar_image, model, processor, device, threshold=0.5, blur_kernel=3)

#%%
# Apply to Pokemon dataset
if 'orig_pokemon_gen1' in dir():
    idx = random.randint(0, len(orig_pokemon_gen1) - 1)
    pokemon_image, pokemon_label = orig_pokemon_gen1[idx]

    print(f"=== Pokémon: {pokemon_class_names[pokemon_label]} ===")
    visualize_attention_segmentation(pokemon_image, model, processor, device, threshold=0.6, blur_kernel=5)

#%% md
"""
#### **5.6.4 Try Your Own Images**
"""

#%%
# Paste any image URL here
custom_url = "YOUR_IMAGE_URL_HERE"
# Uncomment to run:
# visualize_attention_segmentation_from_url(custom_url, model, processor, device, threshold=0.6, blur_kernel=5)

# ============================================================================
# END OF COPY SECTION
# ============================================================================

