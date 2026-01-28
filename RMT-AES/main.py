import sys, random, os
import imageio.v2 as imageio
import numpy as np
from cryp import RMT, AES
from Neuracrypt import NeuraCrypt
from PIL import Image
from tqdm import tqdm

def save_image_from_array(img_array, save_path):
    """
    Save a numpy array as an image using PIL.

    Args:
        img_array (numpy.ndarray): The image data in numpy array format.
        save_path (str): The path to save the image.
    """
    img = Image.fromarray(img_array.astype('uint8'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    img.save(save_path, 'JPEG')

def load_image_to_array(image_path):
    """
    Load an image from a file path and return it as a numpy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image data in numpy array format.
    """
    with Image.open(image_path) as img:
        img_array = np.array(img, dtype=np.float32)
    return img_array

class ImageDisguisingApp:
    def __init__(self, method, block_size, noise_level, dataset_directory, output_directory):
        self.method = method
        self.block_size = block_size
        self.noise_level = noise_level
        self.dataset_directory = dataset_directory
        self.output_directory = output_directory

        if self.method == 'RMT':
            self.encoder = RMT
        elif self.method == 'AES':
            self.encoder = AES
        elif self.method == 'NeuraCrypt':
            self.encoder = NeuraCrypt
        else:
            raise ValueError("Method must be either 'RMT', 'AES', or 'NeuraCrypt'")

    def encrypt_images(self):
        self.image_files = []
        self.image_paths = []

        for root, _, files in os.walk(self.dataset_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.bmp')):
                    self.image_files.append(file)
                    self.image_paths.append(os.path.join(root, file))

        encrypted_images = []
        original_images = []

        for image_path in tqdm(self.image_paths, desc="Encrypting Images"):
            image = load_image_to_array(image_path)

            adjusted_row = (image.shape[0] + self.block_size - 1) // self.block_size * self.block_size
            adjusted_col = (image.shape[1] + self.block_size - 1) // self.block_size * self.block_size

            pad_row = adjusted_row - image.shape[0]
            pad_col = adjusted_col - image.shape[1]

            if len(image.shape) == 3:
                image_padded = np.pad(image, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')
            else:
                image_padded = np.pad(image, ((0, pad_row), (0, pad_col)), mode='edge')

            original_images.append(image_padded)

            if self.method == 'RMT':
                encoder_instance = self.encoder(
                    image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                    block_size=self.block_size
                )
                noise = self.noise_level != 0
                encrypted_img_array = encoder_instance.Encode(image_padded, noise=noise, noise_level=self.noise_level)
            elif self.method == 'AES':
                encoder_instance = self.encoder(
                    image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                    block_size=(self.block_size, self.block_size)
                )
                noise = self.noise_level != 0
                encrypted_img_array = encoder_instance.Encode(image_padded, noise=noise, noise_level=self.noise_level)
            elif self.method == 'NeuraCrypt':
                encoder_instance = self.encoder(
                    image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                    patch_size=self.block_size
                )
                encrypted_img_array = encoder_instance.forward(image_padded).detach().numpy()

            encrypted_images.append(encrypted_img_array)

            relative_path = os.path.relpath(image_path, self.dataset_directory)
            os.makedirs(self.dataset_directory, exist_ok=True)
            encrypted_image_path = os.path.join(self.output_directory, relative_path)
            save_image_from_array(encrypted_img_array, encrypted_image_path)

        print("Encryption done for all images in the directory!")
        self.original_images = original_images
        self.encrypted_images = encrypted_images

    def attack_images(self, known_pairs):
        if not hasattr(self, 'original_images') or not hasattr(self, 'encrypted_images'):
            raise ValueError("Images must be encrypted first before attacking.")

        index = random.sample(range(len(self.original_images)), known_pairs)
        rec = []

        if len(self.original_images[0].shape) == 3:
            RMT_Mat = self.encoder.Estimate(np.array(self.original_images)[index, :, :, :], np.array(self.encrypted_images)[index, :, :, :])
        else:
            RMT_Mat = self.encoder.Estimate(np.array(self.original_images)[index, :, :], np.array(self.encrypted_images)[index, :, :])

        for i in tqdm(range(len(self.encrypted_images)), desc="Attacking Images"):
            encoded_img = self.encrypted_images[i]
            recover = self.encoder.Recover(encoded_img, RMT_Mat)
            rec.append(recover)

            print(f"Recovery error for image {i}: {np.linalg.norm(self.encoder.normalize(self.original_images[i]) - recover)}")

        print("Attack done!")
        return rec

if __name__ == "__main__":
    # method = 'RMT'  # or 'AES'
    # block_size = 4  # Example block size
    # noise_level = 0  # Example noise level
    # dataset_directory = './40X_all'  # Directory containing the dataset
    # output_directory = './Breast/rmt-4-0'  # Directory to save encrypted images
    # app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    # app.encrypt_images()

    # method = 'RMT'  # or 'AES'
    # block_size = 4  # Example block size
    # noise_level = 4  # Example noise level
    # dataset_directory = './40X_all'  # Directory containing the dataset
    # output_directory = './Breast/rmt-4-4'  # Directory to save encrypted images
    # app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    # app.encrypt_images()

    # method = 'AES'  # or 'AES'
    # block_size = 4  # Example block size
    # noise_level = 4  # Example noise level
    # dataset_directory = './40X_all'  # Directory containing the dataset
    # output_directory = './Breast/aes-4-4'  # Directory to save encrypted images
    # app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    # app.encrypt_images()

    # method = 'AES'  # or 'AES'
    # block_size = 4  # Example block size
    # noise_level = 4  # Example noise level
    # dataset_directory = './40X_all'  # Directory containing the dataset
    # output_directory = './Breast/aes-4-4'  # Directory to save encrypted images
    # app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    # app.encrypt_images()    


    method = 'NeuraCrypt'  # or 'AES'
    block_size = 4  # Example block size
    noise_level = 0  # Example noise level
    dataset_directory = './40X_all'  # Directory containing the dataset
    output_directory = './Breast/neu-4-0'  # Directory to save encrypted images
    app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    app.encrypt_images()    
