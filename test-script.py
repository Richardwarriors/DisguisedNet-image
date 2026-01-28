
import requests
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np

# Replace with your ngrok URL from Colab
API_URL = "https://ruthfully-waterlocked-james.ngrok-free.dev/segment"  # e.g., "https://abc123.ngrok.io/segment"

# Path to your test image
IMAGE_PATH = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/20151127_123952_jpg.rf.e1b8c127be8f0edf9fbb28084e727d2b.jpg"


def test_segmentation_api(image_path, api_url):
    """Test the segmentation API with a local image"""
    
    print(f"��� Sending image to API: {api_url}")
    
    # Read the image file
    with open(image_path, 'rb') as f:
        files = {'file': ('image.jpg', f, 'image/jpeg')}
        
        # Send POST request
        response = requests.post(api_url, files=files)
    
    # Check response
    if response.status_code == 200:
        print("✅ API request successful!")
        
        # Parse JSON response
        result = response.json()
        
        print(f"\n��� Results:")
        print(f"   - Found {len(result['masks'])} food items")
        print(f"   - Food names: {result['food_names']}")
        print(f"   - Confidences: {[f'{c:.2f}' for c in result['confidences']]}")
        
        # Visualize the masks
        visualize_results(image_path, result)
        
        return result
    else:
        print(f"❌ API request failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def visualize_results(image_path, result):
    """Visualize the original image and segmentation masks"""
    
    # Load original image
    original_img = Image.open(image_path)
    
    # Create subplot for visualization
    num_items = len(result['masks']) + 1  # +1 for original image
    cols = min(3, num_items)
    rows = (num_items + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_items == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    # Show original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Show each mask
    for i, (mask_b64, name, conf) in enumerate(zip(
        result['masks'], 
        result['food_names'], 
        result['confidences']
    )):
        # Decode base64 mask
        mask_data = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_data))
        
        # Display mask
        axes[i + 1].imshow(mask_img, cmap='gray')
        axes[i + 1].set_title(f"{name}\nConfidence: {conf:.2%}", 
                             fontsize=12, fontweight='bold')
        axes[i + 1].axis('off')
    
    # Hide unused subplots
    for i in range(num_items, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n��� Visualization saved as 'segmentation_results.png'")
    plt.show()

def visualize_overlay(image_path, result):
    """Create overlay visualization with masks on original image"""
    
    original_img = Image.open(image_path).convert('RGBA')
    width, height = original_img.size
    
    # Create a blank overlay
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Color palette for different foods
    colors = [
        (255, 0, 0, 100),    # Red
        (0, 255, 0, 100),    # Green
        (0, 0, 255, 100),    # Blue
        (255, 255, 0, 100),  # Yellow
        (255, 0, 255, 100),  # Magenta
        (0, 255, 255, 100),  # Cyan
    ]
    
    for i, (mask_b64, name) in enumerate(zip(result['masks'], result['food_names'])):
        # Decode mask
        mask_data = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_data)).convert('L')
        mask_img = mask_img.resize((width, height))
        
        # Create colored overlay for this mask
        color = colors[i % len(colors)]
        colored_mask = Image.new('RGBA', (width, height), color)
        
        # Apply mask as alpha channel
        overlay = Image.composite(colored_mask, overlay, mask_img)
    
    # Combine original and overlay
    result_img = Image.alpha_composite(original_img, overlay)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img)
    plt.title("Segmentation Overlay", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Add legend
    legend_text = "\n".join([f"• {name}" for name in result['food_names']])
    plt.text(10, 30, legend_text, fontsize=12, color='white', 
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('segmentation_overlay.png', dpi=150, bbox_inches='tight')
    print(f"��� Overlay saved as 'segmentation_overlay.png'")
    plt.show()

if __name__ == "__main__":
    # Test the API
    result = test_segmentation_api(IMAGE_PATH, API_URL)
    
    if result:
        # Create overlay visualization
        visualize_overlay(IMAGE_PATH, result)
        
        print("\n✨ Testing complete!")
