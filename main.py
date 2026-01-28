import sys, random, os
import pickle
import imageio.v2 as imageio
import numpy as np
from cryp import RMT, AES
from Neuracrypt import NeuraCrypt
from PIL import Image
from tqdm import tqdm
from torchvision import datasets


def download_mnist_pytorch(output_dir='mnist_data'):
    """Download MNIST using PyTorch and save as images."""
    
    # Download datasets
    train_dataset = datasets.MNIST(root='./temp', train=True, download=True)
    test_dataset = datasets.MNIST(root='./temp', train=False, download=True)
    
    # Save training images
    print("Saving training images...")
    for idx, (img, label) in enumerate(train_dataset):
        os.makedirs(f'{output_dir}/train/{label}', exist_ok=True)
        img.save(f'{output_dir}/train/{label}/{idx}.png')
    
    # Save test images
    print("Saving test images...")
    for idx, (img, label) in enumerate(test_dataset):
        os.makedirs(f'{output_dir}/test/{label}', exist_ok=True)
        img.save(f'{output_dir}/test/{label}/{idx}.png')
    
    print(f"MNIST saved to {output_dir}/")


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
    def __init__(self, method, block_size, noise_level, dataset_directory, output_directory, shuffle=False):
        self.method = method
        self.block_size = block_size
        self.noise_level = noise_level
        self.dataset_directory = dataset_directory
        self.output_directory = output_directory
        self.shuffle = shuffle

        if self.method == 'RMT':
            self.encoder = RMT
        elif self.method == 'AES':
            self.encoder = AES
            self.encoder_instance = None
        elif self.method == 'NeuraCrypt':
            self.encoder = NeuraCrypt
        else:
            raise ValueError("Method must be either 'RMT', 'AES', or 'NeuraCrypt'")
        
    def encrypt_image_label_pairs(self, label_directory, label_output_directory):
        """Encrypt images and their corresponding labels with the same shuffle"""
        self.image_files = []
        self.image_paths = []
        
        for root, _, files in os.walk(self.dataset_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.bmp')):
                    self.image_files.append(file)
                    self.image_paths.append(os.path.join(root, file))
        
        for idx, image_path in enumerate(tqdm(self.image_paths, desc="Encrypting Image-Label Pairs")):
            # Generate a unique seed for this image-label pair
            seed = hash(image_path) % (2**32)  # or use idx, or random.randint(0, 2**32-1)
            
            # Encrypt image
            image = load_image_to_array(image_path)
            encrypted_img = self._encrypt_single_image(image, seed)
            
            # Save encrypted image
            relative_path = os.path.relpath(image_path, self.dataset_directory)
            encrypted_image_path = os.path.join(self.output_directory, relative_path)
            save_image_from_array(encrypted_img, encrypted_image_path)
            
            # Encrypt corresponding label
            label_path = os.path.join(label_directory, os.path.relpath(image_path, self.dataset_directory))
            if os.path.exists(label_path):
                label = load_image_to_array(label_path)
                encrypted_label = self._encrypt_single_image(label, seed)
                
                # Save encrypted label
                encrypted_label_path = os.path.join(label_output_directory, relative_path)
                save_image_from_array(encrypted_label, encrypted_label_path)
        
        print("Encryption done for all image-label pairs!")

    def _encrypt_single_image(self, image, shuffling_seed):
        """Helper method to encrypt a single image with a given seed"""
        adjusted_row = (image.shape[0] + self.block_size - 1) // self.block_size * self.block_size
        adjusted_col = (image.shape[1] + self.block_size - 1) // self.block_size * self.block_size
        
        pad_row = adjusted_row - image.shape[0]
        pad_col = adjusted_col - image.shape[1]
        
        if len(image.shape) == 3:
            image_padded = np.pad(image, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')
        else:
            image_padded = np.pad(image, ((0, pad_row), (0, pad_col)), mode='edge')
        
        if self.method == 'RMT':
            encoder_instance = self.encoder(
                image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                block_size=self.block_size,
                Shuffle=self.shuffle
            )
            print(f"RMT Shuffle setting: {encoder_instance.shuffle}")
            encrypted_img = encoder_instance.Encode(image_padded, noise=self.noise_level != 0, 
                                                noise_level=self.noise_level, 
                                                shuffling_seed=shuffling_seed)
        elif self.method == 'AES':
            encoder_instance = self.encoder(
                image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                block_size=(self.block_size, self.block_size),
                Shuffle=self.shuffle
            )
            encrypted_img = encoder_instance.Encode(image_padded, noise=self.noise_level != 0, 
                                                noise_level=self.noise_level,
                                                shuffling_seed=shuffling_seed)
        
        return encrypted_img

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
                    block_size=self.block_size,
                    Shuffle=self.shuffle
                )
                noise = self.noise_level != 0
                encrypted_img_array = encoder_instance.Encode(image_padded, noise=noise, noise_level=self.noise_level)
            elif self.method == 'AES':
                # Create encoder ONCE on first image, reuse for all others with SAME key
                if self.encoder_instance is None:
                    self.encoder_instance = self.encoder(
                        image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                        block_size=(self.block_size, self.block_size),
                        One_cipher=True,
                        Shuffle=self.shuffle
                    )
                    print(f"AES encoder created with One_cipher=True - using SAME key for ALL images")
                noise = self.noise_level != 0
                encrypted_img_array = self.encoder_instance.Encode(image_padded, noise=noise, noise_level=self.noise_level)
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


    def attack_images(self, known_pairs, original_dataset_dir=None, encrypted_dataset_dir=None, output_dir='recovered'):
        """
        Perform attack to recover images.

        If original_dataset_dir and encrypted_dataset_dir are provided, load matching files
        from those directories (by relative path) and use them as original/encrypted datasets.
        Otherwise uses self.original_images and self.encrypted_images populated by encrypt_images().

        known_pairs: number of known plaintext-ciphertext pairs to sample for Estimate.
        """
        # If user supplied dataset directories, load images from them (matching relative paths)
        if original_dataset_dir and encrypted_dataset_dir:
            if not os.path.isdir(original_dataset_dir) or not os.path.isdir(encrypted_dataset_dir):
                raise ValueError("Both original_dataset_dir and encrypted_dataset_dir must be valid directories.")

           
            def collect_files(root_dir, strip_prefix=None):
                files = []
                for root, _, filenames in os.walk(root_dir):
                    for f in filenames:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            full = os.path.join(root, f)
                            rel = os.path.relpath(full, root_dir).replace("\\", "/")
                            # if requested, also add an alternate key with prefix stripped from filename
                            if strip_prefix and os.path.basename(rel).startswith(strip_prefix):
                                dirname = os.path.dirname(rel)
                                stripped_name = os.path.basename(rel)[len(strip_prefix):]
                                alt_rel = os.path.join(dirname, stripped_name).replace("\\", "/")
                                files.append((alt_rel, full))
                            files.append((rel, full))
                return dict(files)
    
            orig_map = collect_files(original_dataset_dir)
            # strip the "encrypted_" prefix from filenames in the encrypted dir so keys match originals
            enc_map = collect_files(encrypted_dataset_dir, strip_prefix='encrypted_')


            # intersect relative paths to ensure same structure
            common_rels = sorted(set(orig_map.keys()).intersection(set(enc_map.keys())))
            if len(common_rels) == 0:
                raise ValueError("No matching files found between provided directories.")
            if len(common_rels) < known_pairs:
                raise ValueError(f"Not enough matching files ({len(common_rels)}) for known_pairs={known_pairs}.")

            # load arrays in deterministic order
            original_images = []
            encrypted_images = []
            self.image_files = []
            self.image_paths = []
            for rel in common_rels:
                orig_path = orig_map[rel]
                enc_path = enc_map[rel]
                original_images.append(load_image_to_array(orig_path))
                encrypted_images.append(load_image_to_array(enc_path))
                self.image_files.append(os.path.basename(rel))
                self.image_paths.append(rel)

            self.original_images = original_images
            self.encrypted_images = encrypted_images

        # if not present, ensure we have images from previous encrypt_images()
        if not hasattr(self, 'original_images') or not hasattr(self, 'encrypted_images'):
            raise ValueError("Images must be available. Call encrypt_images() first or provide dataset dirs to attack_images().")

        print("Starting attack...")
        print(len(self.original_images), len(self.encrypted_images))
        if known_pairs <= 0 or known_pairs > len(self.original_images):
            raise ValueError("known_pairs must be >0 and <= number of available images.")

        index = random.sample(range(len(self.original_images)), known_pairs)
        rec = []

        # create an encoder instance to call instance methods (Estimate, Recover, normalize)
        sample_img = self.original_images[0]
        if len(sample_img.shape) == 3:
            image_size = (sample_img.shape[0], sample_img.shape[1], sample_img.shape[2])
        else:
            image_size = (sample_img.shape[0], sample_img.shape[1])


        if self.method == 'RMT':
            encoder_instance = self.encoder(
                image_size=image_size,
                block_size=self.block_size,
                Shuffle=self.shuffle
            )

            # call instance Estimate (bound) with selected known pairs
            if len(sample_img.shape) == 3:
                RMT_Mat = encoder_instance.Estimate(np.array(self.original_images)[index, :, :, :], np.array(self.encrypted_images)[index, :, :, :])
            else:
                RMT_Mat = encoder_instance.Estimate(np.array(self.original_images)[index, :, :], np.array(self.encrypted_images)[index, :, :])


            # prepare base directory to save recovered images
            if original_dataset_dir:
                recovered_base = os.path.join(original_dataset_dir, output_dir)
            else:
                recovered_base = os.path.join(self.dataset_directory, output_dir)
            os.makedirs(recovered_base, exist_ok=True)

            for i in tqdm(range(len(self.encrypted_images)), desc="Attacking Images"):
                encoded_img = self.encrypted_images[i]
                recover = encoder_instance.Recover(encoded_img, RMT_Mat)
                rec.append(recover)

                # # test print
                # for idx, mat in enumerate(recover):
                #     if idx < 5:
                #         print(f"Recovered Block {idx} RMT matrix:\n{mat}")

                # compute and print recovery error
                try:
                    orig_norm = encoder_instance.normalize(self.original_images[i])
                    err = np.linalg.norm(orig_norm - recover)
                except Exception:
                    err = np.nan
                print(f"Recovery error for image {i}: {err}")

                # convert recovered image to uint8 for saving
                r = recover.copy()
                if np.nanmax(r) <= 1.0:
                    r = (r * 255.0)
                r = np.clip(r, 0, 255).astype(np.uint8)

                # choose file name based on available paths
                name = f"recovered_{i}.jpg"
                if hasattr(self, "image_paths") and i < len(self.image_paths):
                    base = os.path.basename(self.image_paths[i])
                    name = f"recovered_{i}_" + os.path.splitext(base)[0] + ".jpg"

                # determine relative subfolder for this image so we can preserve structure
                rel = None
                if hasattr(self, "image_paths") and i < len(self.image_paths):
                    img_path = self.image_paths[i]
                    if not os.path.isabs(img_path):
                        rel = img_path.replace('\\', '/')
                    else:
                        base_for_rel = original_dataset_dir if original_dataset_dir else self.dataset_directory
                        try:
                            rel = os.path.relpath(img_path, base_for_rel).replace('\\', '/')
                        except Exception:
                            rel = os.path.basename(img_path)
                else:
                    rel = name

                subfolder = os.path.dirname(rel)
                if subfolder:
                    target_dir = os.path.join(recovered_base, subfolder)
                    os.makedirs(target_dir, exist_ok=True)
                    recovered_path = os.path.join(target_dir, name)
                else:
                    recovered_path = os.path.join(recovered_base, name)

                save_image_from_array(r, recovered_path)
                print(f"Saved recovered image: {recovered_path}")

            print("Attack done!")
            return rec

        elif self.method == 'AES':
            # --- AES Codebook Attack (logic inlined) ---
            def build_codebook(pairs):
                codebook = {}
                for e_bytes, o_bytes in pairs:
                    for i in range(0, len(e_bytes), 16):
                        e_block = e_bytes[i:i+16]
                        o_block = o_bytes[i:i+16]
                        codebook[e_block] = o_block
                return codebook

            def codebook_attack(codebook, encrypted_bytes):
                reconstructed = bytearray()
                hits = 0
                total_checks = 0
                for i in range(0, len(encrypted_bytes), 16):
                    block = encrypted_bytes[i:i+16]
                    total_checks += 1
                    if block in codebook:
                        reconstructed.extend(codebook[block])
                        hits += 1
                    else:
                        reconstructed.extend(b"\x00" * 16)
                return bytes(reconstructed), hits, total_checks

            # Build codebook from known pairs
            codebook_pairs = []
            for idx in index:
                enc_img = self.encrypted_images[idx]
                orig_img = self.original_images[idx]
                enc_bytes = enc_img.astype(np.uint8).tobytes()
                orig_bytes = orig_img.astype(np.uint8).tobytes()
                codebook_pairs.append((enc_bytes, orig_bytes))
            codebook = build_codebook(codebook_pairs)

            # prepare base directory to save recovered images
            if original_dataset_dir:
                recovered_base = os.path.join(original_dataset_dir, output_dir)
            else:
                recovered_base = os.path.join(self.dataset_directory, output_dir)
            os.makedirs(recovered_base, exist_ok=True)

            total_hits = 0
            total_checks = 0

            for i in tqdm(range(len(self.encrypted_images)), desc="Attacking Images (AES Codebook)"):
                enc_img = self.encrypted_images[i]
                enc_bytes = enc_img.astype(np.uint8).tobytes()
                rec_bytes, hits, checks = codebook_attack(codebook, enc_bytes)
                total_hits += hits
                total_checks += checks
                
                recover = np.frombuffer(rec_bytes, dtype=np.uint8).reshape(enc_img.shape)
                rec.append(recover)

                # compute and print recovery error
                try:
                    orig_norm = self.original_images[i].astype(np.uint8)
                    err = np.linalg.norm(orig_norm - recover)
                except Exception:
                    err = np.nan
                
                # hit_rate = (hits / checks * 100) if checks > 0 else 0
                # print(f"Image {i} - Recovery error: {err:.2f}, Hit rate: {hit_rate:.2f}% ({hits}/{checks})")

                r = recover.copy()
                if np.nanmax(r) <= 1.0:
                    r = (r * 255.0)
                r = np.clip(r, 0, 255).astype(np.uint8)

                name = f"recovered_{i}.jpg"
                if hasattr(self, "image_paths") and i < len(self.image_paths):
                    base = os.path.basename(self.image_paths[i])
                    name = f"recovered_{i}_" + os.path.splitext(base)[0] + ".jpg"

                rel = None
                if hasattr(self, "image_paths") and i < len(self.image_paths):
                    img_path = self.image_paths[i]
                    if not os.path.isabs(img_path):
                        rel = img_path.replace('\\', '/')
                    else:
                        base_for_rel = original_dataset_dir if original_dataset_dir else self.dataset_directory
                        try:
                            rel = os.path.relpath(img_path, base_for_rel).replace('\\', '/')
                        except Exception:
                            rel = os.path.basename(img_path)
                else:
                    rel = name

                subfolder = os.path.dirname(rel)
                if subfolder:
                    target_dir = os.path.join(recovered_base, subfolder)
                    os.makedirs(target_dir, exist_ok=True)
                    recovered_path = os.path.join(target_dir, name)
                else:
                    recovered_path = os.path.join(recovered_base, name)

                save_image_from_array(r, recovered_path)
                print(f"Saved recovered image: {recovered_path}")

            overall_hit_rate = (total_hits / total_checks * 100) if total_checks > 0 else 0
            print(f"\nOverall Codebook Hit Rate: {overall_hit_rate:.2f}% ({total_hits}/{total_checks})")
            print("AES Codebook Attack done!")
            return rec
        else:
            raise NotImplementedError("Attack is implemented for RMT/AES only.")
        
    def test_attack_images(self, num_images=2, original_images=None, encrypted_images=None, method='AES'):
        """
            run attack_images on images from an array
        
        """
        if original_images is None or encrypted_images is None:
            raise ValueError("original_images and encrypted_images must be provided for testing.")

        # save the images to temp directories
        temp_orig_dir = "temp_originals"
        temp_enc_dir = "temp_encrypted"
        os.makedirs(temp_orig_dir, exist_ok=True)
        os.makedirs(temp_enc_dir, exist_ok=True)
        for i in range(num_images):
            orig_path = os.path.join(temp_orig_dir, f"{i}.jpg")
            enc_path = os.path.join(temp_enc_dir, f"{i}.jpg")
            save_image_from_array(original_images[i], orig_path)
            save_image_from_array(encrypted_images[i], enc_path)
        
        recovered_images = self.attack_images(known_pairs=num_images, original_dataset_dir=temp_orig_dir, encrypted_dataset_dir=temp_enc_dir, output_dir='temp_recovered')
        # print out the original, encrypted and recovered images for comparison
        for i in range(num_images):
            print(f"Original Image {i}:\n", original_images[i])
            print(f"Encrypted Image {i}:\n", encrypted_images[i])
            print(f"Recovered Image {i}:\n", recovered_images[i])
            # # check that recovered is close to original
            diff = np.linalg.norm(original_images[i] - recovered_images[i])
            print(f"Difference norm between original and recovered image {i}: {diff}")
    
    def _collect_files(self, root_dir):
        """
        Collect all image files from a directory tree, returning a dict of relative_path -> full_path.
        """
        files = {}
        for root, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, root_dir).replace("\\", "/")
                    files[rel] = full
        return files
    
    def build_aes_codebook_from_folder(self, original_dir, encrypted_dir, known_pairs=None, codebook_path=None):
        """
        Build an AES codebook from matching images in original_dir and encrypted_dir.
        
        Args:
            original_dir (str): Directory containing original (plaintext) images.
            encrypted_dir (str): Directory containing encrypted images.
            known_pairs (int, optional): Number of known plaintext-ciphertext image pairs to use for codebook.
                                        If None, uses all available matching files.
            codebook_path (str, optional): Path to save the codebook as a pickle file.
        
        Returns:
            dict: The codebook mapping encrypted 16-byte blocks to original 16-byte blocks.
        """
        if self.method != 'AES':
            raise ValueError("Codebook method is only available for AES encryption.")
        
        orig_map = self._collect_files(original_dir)
        enc_map = self._collect_files(encrypted_dir)
        common_rels = sorted(set(orig_map.keys()).intersection(set(enc_map.keys())))
        
        if len(common_rels) == 0:
            raise ValueError("No matching files found between original and encrypted directories.")
        
        # Select subset of images if known_pairs is specified
        if known_pairs is not None:
            if known_pairs <= 0 or known_pairs > len(common_rels):
                raise ValueError(f"known_pairs must be > 0 and <= {len(common_rels)} (available matching files).")
            selected_rels = random.sample(common_rels, known_pairs)
            print(f"Building codebook from {known_pairs} known image pairs (out of {len(common_rels)} available)...")
        else:
            selected_rels = common_rels
            print(f"Building codebook from all {len(common_rels)} image pairs...")
        
        codebook = {}
        
        for rel in tqdm(selected_rels, desc="Building AES codebook"):
            orig_img = load_image_to_array(orig_map[rel])
            enc_img = load_image_to_array(enc_map[rel])
            orig_bytes = orig_img.astype(np.uint8).tobytes()
            enc_bytes = enc_img.astype(np.uint8).tobytes()
            
            for i in range(0, len(enc_bytes), 16):
                e_block = enc_bytes[i:i+16]
                o_block = orig_bytes[i:i+16]
                codebook[e_block] = o_block
        
        print(f"Codebook created with {len(codebook)} unique 16-byte blocks.")
        
        if codebook_path:
            with open(codebook_path, "wb") as f:
                pickle.dump(codebook, f)
            print(f"Codebook saved to {codebook_path}")
        
        return codebook
    
    def load_aes_codebook(self, codebook_path):
        """
        Load a previously saved AES codebook from a pickle file.
        
        Args:
            codebook_path (str): Path to the codebook pickle file.
        
        Returns:
            dict: The loaded codebook.
        """
        with open(codebook_path, "rb") as f:
            codebook = pickle.load(f)
        print(f"Codebook loaded from {codebook_path} with {len(codebook)} blocks.")
        return codebook
    
    def recover_images_with_codebook(self, encrypted_dir, codebook, output_dir="recovered_with_codebook"):
        """
        Recover images in encrypted_dir using a precomputed AES codebook.
        
        Args:
            encrypted_dir (str): Directory containing encrypted images to recover.
            codebook (dict): The AES codebook mapping encrypted blocks to original blocks.
            output_dir (str): Directory to save recovered images.
        
        Returns:
            list: List of recovered image arrays.
        """
        if self.method != 'AES':
            raise ValueError("Codebook method is only available for AES encryption.")
        
        enc_map = self._collect_files(encrypted_dir)
        os.makedirs(output_dir, exist_ok=True)
        recovered_images = []
        
        total_hits = 0
        total_checks = 0
        
        print(f"Recovering {len(enc_map)} images using codebook...")
        for rel, enc_path in tqdm(enc_map.items(), desc="Recovering images with codebook"):
            enc_img = load_image_to_array(enc_path)
            enc_bytes = enc_img.astype(np.uint8).tobytes()
            rec_bytes = bytearray()
            
            hits = 0
            checks = 0
            
            for i in range(0, len(enc_bytes), 16):
                block = enc_bytes[i:i+16]
                checks += 1
                if block in codebook:
                    rec_bytes.extend(codebook[block])
                    hits += 1
                else:
                    rec_bytes.extend(block)  # Use encrypted bytes as fallback instead of zeros
            
            total_hits += hits
            total_checks += checks
            
            recover = np.frombuffer(bytes(rec_bytes), dtype=np.uint8).reshape(enc_img.shape)
            recovered_images.append(recover)
            
            # hit_rate = (hits / checks * 100) if checks > 0 else 0
            # print(f"Image {rel} - Hit rate: {hit_rate:.2f}% ({hits}/{checks})")
            
            # Save recovered image
            out_path = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_image_from_array(recover, out_path)
        
        overall_hit_rate = (total_hits / total_checks * 100) if total_checks > 0 else 0
        print(f"\nOverall Codebook Hit Rate: {overall_hit_rate:.2f}% ({total_hits}/{total_checks})")
        print(f"Recovery complete. Images saved to {output_dir}")
        return recovered_images


    

if __name__ == "__main__":
    
    # Example: Build AES codebook from one folder and recover images from another
    # ============================================================================
    # Uncomment and adjust paths as needed:
    #
    # # Step 1: Initialize the app with AES
    # app = ImageDisguisingApp('AES', block_size=4, noise_level=0, dataset_directory='', output_directory='', shuffle=False)
    #
    # # Step 2: Build codebook from a training set with specific number of known pairs
    # train_original_dir = "path/to/train/originals"
    # train_encrypted_dir = "path/to/train/encrypted"
    # codebook = app.build_aes_codebook_from_folder(train_original_dir, train_encrypted_dir, 
    #                                               known_pairs=10, codebook_path="aes_codebook.pkl")
    #
    # # Step 3: Recover images in test set using the codebook
    # test_encrypted_dir = "path/to/test/encrypted"
    # recovered_images = app.recover_images_with_codebook(test_encrypted_dir, codebook, output_dir="recovered_test_images")
    #
    # # Alternative: Load a previously saved codebook
    # # codebook = app.load_aes_codebook("aes_codebook.pkl")
    # # recovered_images = app.recover_images_with_codebook(test_encrypted_dir, codebook, output_dir="recovered_test_images")
    #
    # # To test different numbers of known pairs for attack effectiveness:
    # # for num_pairs in [1, 5, 10, 20, 50]:
    # #     codebook = app.build_aes_codebook_from_folder(train_original_dir, train_encrypted_dir, 
    # #                                                   known_pairs=num_pairs, 
    # #                                                   codebook_path=f"aes_codebook_{num_pairs}_pairs.pkl")
    # #     recovered_images = app.recover_images_with_codebook(test_encrypted_dir, codebook, 
    # #                                                         output_dir=f"recovered_with_{num_pairs}_pairs")
    
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

    '''
    updated by jason
    '''
    # folders = ['BG', 'D', 'N', 'P', 'S', 'V']
    # encrypted_method = ['AES', 'RMT']
    # block_sizes = [4, 32]

    # for method in encrypted_method:
    #     for block_size in block_sizes:
    #         for folder in folders:

    #             # method = 'AES'  # or 'AES'
    #             # block_size = 32  # Example block size
    #             noise_level = block_size  # Example noise level
    #             dataset_directory = f"../../MData/256-OG/{folder}"  # Directory containing the dataset
    #             output_directory = f"../../MData/encrypted-noise/{method}-B{block_size}N{noise_level}/{folder}"  # Directory to save encrypted images
    #             print(f"Processing folder: {folder} with method: {method}, block size: {block_size}, noise level: {noise_level}")
                
    #             app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    #             app.encrypt_images()





    # block_size = [4, 32]
    # known_pairs = [1, 5, 10, 15, 20, 25, 30, 35]
    # for bs in block_size:
        
    #     print(f"Processing Breast with RMT, block size: {bs}")
    #     dataset_directory = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-OG"  # Directory containing the dataset
    #     output_directory = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-RMT-B{bs}N{bs}NS"  # Directory to save encrypted images
        
        
    #     app = ImageDisguisingApp('RMT', bs, bs, dataset_directory, output_directory)
    #     # app.encrypt_images()

    #     # original_images = [load_image_to_array(f"../../DisNet-classfication/DisNet-classfication/Data/Breast/OG-test/b/SOB_B_A-14-22549AB-400-002.png"), load_image_to_array(f"../../DisNet-classfication/DisNet-classfication/Data/Breast/OG/b/SOB_B_A-14-22549AB-400-003.png")]
    #     # encrypted_images = [load_image_to_array(f"../../DisNet-classfication/DisNet-classfication/Data/Breast/RMT-B4N4-Attack/b/SOB_B_A-14-22549AB-400-002.png"), load_image_to_array(f"../../DisNet-classfication/DisNet-classfication/Data/Breast/RMT-B4N4-Attack/b/SOB_B_A-14-22549AB-400-003.png")]

    #     # app.original_images = original_images
    #     # app.encrypted_images = encrypted_images
    #     for kp in known_pairs:
    #         recovered_images = app.attack_images(known_pairs=kp, original_dataset_dir="../../DisNet-classfication/DisNet-classfication/Data/Breast/256-OG", encrypted_dataset_dir=f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-RMT-B{bs}N{bs}NS", output_dir=f"recovered-RMT-B{bs}N{bs}NS-{kp}-pairs")
    
    
    runs = [1, 2, 3, 4, 5]
    block_size = [4]
    known_pairs = [1, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]
    # known_pairs = [300]
    for bs in block_size:
        for run in runs:
            dataset_directory = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-OG-Split2/split{run}"  # Directory containing the dataset
            output_directory = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-AES-B4N4NS/split{run}"  # Directory to save encrypted images
            app = ImageDisguisingApp('AES', bs, bs, dataset_directory, output_directory, shuffle=False)
            app.encrypt_images()

            
            for kp in known_pairs:
                # recovered_images = app.attack_images(known_pairs=kp, original_dataset_dir=dataset_directory, encrypted_dataset_dir=output_directory, output_dir=f"r-AES-B{bs}N{bs}NS-split{run}-{kp}-pairs")
                codebook = app.build_aes_codebook_from_folder(original_dir=f"{dataset_directory}/train", encrypted_dir=f"{output_directory}/train", known_pairs=kp)
                recovered_images = app.recover_images_with_codebook(encrypted_dir=f"{output_directory}/test", codebook=codebook, output_dir=f"{output_directory}/r-AES-B{bs}N{bs}NS-split{run}-{kp}-pairs")
                
    

    """
    runs = [1, 2, 3, 4, 5]
    block_size = [4]
    known_pairs = [1, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100]

    app = ImageDisguisingApp('AES', 4, 4, './mnist_data', './mnist_data_aes', shuffle=False)
    
    app.encrypt_images()
    for kp in known_pairs:
        codebook = app.build_aes_codebook_from_folder(original_dir=f"./mnist_data/train", encrypted_dir=f"./mnist_data_aes/train", known_pairs=kp)
        recovered_images = app.recover_images_with_codebook(encrypted_dir=f"./mnist_data_aes/train", codebook=codebook, output_dir=f"./train_aes-mnist/r-AES-B4NS--{kp}-pairs")

    for kp in known_pairs:
        codebook = app.build_aes_codebook_from_folder(original_dir=f"./mnist_data/train", encrypted_dir=f"./mnist_data_aes/train", known_pairs=kp)
        recovered_images2 = app.recover_images_with_codebook(encrypted_dir=f"./mnist_data_aes/test", codebook=codebook, output_dir=f"./test_aes-mnist/r-AES-B4NS--{kp}-pairs")
       
            # for kp in known_pairs:
            #     # recovered_images = app.attack_images(known_pairs=kp, original_dataset_dir=dataset_directory, encrypted_dataset_dir=output_directory, output_dir=f"r-AES-B{bs}N{bs}NS-split{run}-{kp}-pairs")
            #     codebook = app.build_aes_codebook_from_folder(original_dir=f"{dataset_directory}/train", encrypted_dir=f"{output_directory}/train", known_pairs=kp, codebook_path=f"{output_directory}/AES-CB-B{bs}N{bs}NS-split{run}-{kp}-pairs.pkl")
            #     recovered_images = app.recover_images_with_codebook(encrypted_dir=f"{output_directory}/test", codebook=codebook, output_dir=f"{output_directory}/r-AES-B{bs}N{bs}NS-split{run}-{kp}-pairs")
    
    """
    
    # # Test AES attack
    # dataset_directory = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-OG-Split2/split1"  # Directory containing the dataset
    # output_directory = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-RMT-B4N0NS/split1"  # Directory to save encrypted images
    # app = ImageDisguisingApp('AES', 4, 0, '', '', shuffle=False)

    # original_imgs_aes = []
    # for i in range(5):
    #     # create a simple pattern image
    #     img = np.ones((32, 32, 3), dtype=np.uint8)
        
    #     # make every quarter of the image a different value
    #     img[0:16, 0:16, :] *= 10 * i      # Top-left
    #     img[0:16, 16:32, :] *= 20 + 20 * i  # Top-right
    #     img[16:32, 0:16, :] *= 30 + 30 * i # Bottom-left
    #     img[16:32, 16:32, :] *= 40 + 40 * i # Bottom-right


    #     original_imgs_aes.append(img)

    # # Use a single encoder instance for all images
    # # encoder_instance = AES(image_size=(32, 32, 3), block_size=(4,4), Shuffle=False)
    # encrypted_imgs_aes = []
    # for img in original_imgs_aes:
    #     encoder_instance = app.encoder(
    #                 image_size=(img.shape[0], img.shape[1], img.shape[2]) if len(img.shape) == 3 else (img.shape[0], img.shape[1]),
    #                 block_size=(app.block_size, app.block_size),
    #                 Shuffle=app.shuffle
    #             )

    #     encrypted_img = encoder_instance.Encode(img, noise=False, noise_level=0)
    #     encrypted_imgs_aes.append(encrypted_img)
        

    # app.test_attack_images(num_images=5, original_images=original_imgs_aes, encrypted_images=encrypted_imgs_aes, method='AES')


    # # Testing RMT attack with small 4x4 images
    # dataset_directory = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-OG-Split2/split1"  # Directory containing the dataset
    # output_directory = f"../../DisNet-classfication/DisNet-classfication/Data/Breast/256-RMT-B4N0NS/split1"  # Directory to save encrypted images
    # app = ImageDisguisingApp('RMT', 4, 0, '', '', shuffle=False)

    # original_imgs = [np.random.randint(0, 256, (4, 4), dtype=np.uint8) for _ in range(4)]
    # encoder = RMT(image_size=(4, 4), block_size=4, Shuffle=False)
    # encrypted_imgs = [encoder.Encode(img, noise=False, noise_level=0) for img in original_imgs]

    # app.original_images = original_imgs
    # app.encrypted_images = encrypted_imgs
    # recovered = app.attack_images(known_pairs=4)

    # for i in range(4):
    #     print("Original:\n", original_imgs[i])
    #     print("Encrypted:\n", encrypted_imgs[i])
    #     print("Recovered:\n", recovered[i])
    #     print("Recovery error:", np.linalg.norm(original_imgs[i].astype(float) - recovered[i]))



    # encrypted_method = ['AES', 'RMT']
    # block_sizes = [4, 32]

    # for method in encrypted_method:
    #     for block_size in block_sizes:
    #         # method = 'AES'  # or 'AES'
    #         # block_size = 32  # Example block size
    #         noise_level = block_size  # Example noise level
    #         dataset_directory = f"../../DisNet-classfication/DisNet-classfication/Data/segmentation-data/woundpatch/256-OG-woundpatch/images"  # Directory containing the dataset
    #         output_directory = f"../../DisNet-classfication/DisNet-classfication/Data/segmentation-data/woundpatch/{method}-B{block_size}N{noise_level}NS/images"  # Directory to save encrypted images
    #         label_directory = f"../../DisNet-classfication/DisNet-classfication/Data/segmentation-data/woundpatch/256-OG-woundpatch/labels"  # Directory containing the labels
    #         label_output_directory = f"../../DisNet-classfication/DisNet-classfication/Data/segmentation-data/woundpatch/{method}-B{block_size}N{noise_level}NS/labels"  # Directory to save encrypted labels
    #         print(f"Processing OG with method: {method}, block size: {block_size}, noise level: {noise_level}")
            
    #         app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory, shuffle=False)
    #         app.encrypt_image_label_pairs(label_directory, label_output_directory)

    #         dataset_directory2 = f"../../DisNet-classfication/DisNet-classfication/Data/segmentation-data/CVC/256-OG-cvc/images"  # Directory containing the dataset
    #         output_directory2 = f"../../DisNet-classfication/DisNet-classfication/Data/segmentation-data/CVC/{method}-B{block_size}N{noise_level}NS/images"  # Directory to save encrypted images
    #         label_directory2 = f"../../DisNet-classfication/DisNet-classfication/Data/segmentation-data/CVC/256-OG-cvc/labels"  # Directory containing the labels
    #         label_output_directory2 = f"../../DisNet-classfication/DisNet-classfication/Data/segmentation-data/CVC/{method}-B{block_size}N{noise_level}NS/labels"  # Directory to save encrypted labels
    #         print(f"Processing OG with method: {method}, block size: {block_size}, noise level: {noise_level}")


    #         app2 = ImageDisguisingApp(method, block_size, noise_level, dataset_directory2, output_directory2, shuffle=False)
    #         app2.encrypt_image_label_pairs(label_directory2, label_output_directory2)

    #         app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    #         app.encrypt_images()
    
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # 
    # for method in encrypted_method:
    #     for block_size in block_sizes:
    #         # method = 'AES'  # or 'AES'
    #         # block_size = 32  # Example block size
    #         noise_level = block_size  # Example noise level
    #         dataset_directory = f"../../DisNet-classfication/DisNet-classfication/Data/woundpatch/OG/images"  # Directory containing the dataset
    #         output_directory = f"../../DisNet-classfication/DisNet-classfication/Data/woundpatch/{method}-B{block_size}N{noise_level}NS/images"  # Directory to save encrypted images
    #         label_directory = f"../../DisNet-classfication/DisNet-classfication/Data/woundpatch/OG/labels"  # Directory containing the labels
    #         label_output_directory = f"../../DisNet-classfication/DisNet-classfication/Data/woundpatch/{method}-B{block_size}N{noise_level}NS/labels"  # Directory to save encrypted labels
    #         print(f"Processing OG with method: {method}, block size: {block_size}, noise level: {noise_level}")
            
    #         app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    #         app.encrypt_image_label_pairs(label_directory, label_output_directory)