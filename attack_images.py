"""
attack_images.py
Implements attacks on encrypted images, including RMT and AES codebook attacks.
"""
import numpy as np

# --- AES Codebook Attack ---
def build_codebook(pairs):
    """
    Build a codebook from pairs of (encrypted, original) images.
    Args:
        pairs: list of tuples (encrypted_bytes, original_bytes)
    Returns:
        dict: codebook mapping 16-byte encrypted blocks to 16-byte original blocks
    """
    codebook = {}
    for e_bytes, o_bytes in pairs:
        for i in range(0, len(e_bytes), 16):
            e_block = e_bytes[i:i+16]
            o_block = o_bytes[i:i+16]
            codebook[e_block] = o_block
    return codebook

def codebook_attack(codebook, encrypted_bytes, scaling_fn=None):
    """
    Reconstruct an image using the codebook attack.
    Args:
        codebook: dict mapping 16-byte encrypted blocks to 16-byte original blocks
        encrypted_bytes: bytes of the encrypted image
        scaling_fn: optional function to scale down the reconstructed image
    Returns:
        bytes: reconstructed image
    """
    reconstructed = bytearray()
    for i in range(0, len(encrypted_bytes), 16):
        block = encrypted_bytes[i:i+16]
        if block in codebook:
            reconstructed.extend(codebook[block])
        else:
            reconstructed.extend(b"\x00" * 16)
    if scaling_fn:
        return scaling_fn(reconstructed)
    return bytes(reconstructed)

# --- RMT Attack (Stub) ---
def rmt_attack(encrypted_image, params=None):
    """
    Dummy RMT attack implementation. Replace with actual logic.
    Args:
        encrypted_image: bytes or np.ndarray
        params: optional parameters
    Returns:
        np.ndarray or bytes: reconstructed image
    """
    # TODO: Implement actual RMT attack logic
    return encrypted_image

# --- Tests ---
def test_codebook_attack():
    # Create dummy data
    orig = b"A" * 32 + b"B" * 32
    enc = b"X" * 32 + b"Y" * 32
    pairs = [(enc, orig)]
    codebook = build_codebook(pairs)
    # Test perfect match
    result = codebook_attack(codebook, enc)
    assert result == orig, f"Expected {orig}, got {result}"
    # Test partial match (unknown block)
    enc2 = b"X" * 32 + b"Z" * 32
    result2 = codebook_attack(codebook, enc2)
    assert result2[:32] == b"A" * 32 and result2[32:] == b"\x00" * 32
    print("test_codebook_attack passed.")

def test_rmt_attack():
    # Dummy test: output should match input for stub
    arr = np.arange(16, dtype=np.uint8)
    result = rmt_attack(arr)
    assert np.array_equal(result, arr)
    print("test_rmt_attack passed.")

if __name__ == "__main__":
    test_codebook_attack()
    test_rmt_attack()
