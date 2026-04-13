import socket
import struct
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageOps
import numpy as np
import os
from io import BytesIO
import uuid
import math
from functools import lru_cache

# Configuration
MODEL_NAME = "google/siglip2-so400m-patch16-naflex"
MODEL_REVISION = "cc24074f717b612951c2dead130904ab9b65a81e"
DTYPE = "float32"
SOCKET_PATH = os.getenv("SOCKET_PATH", "/tmp/embedding.sock")
DEVICE = "cpu"

@lru_cache(maxsize=256)
def get_image_size_for_max_num_patches(
    image_height: int, image_width: int, patch_size: int, max_num_patches: int, eps: float = 1e-5
) -> tuple[int, int]:
    """
    This is a direct Python replication of the SigLIP-2 model's preprocessing logic.

    Its sole purpose is to determine the exact target resolution (height, width)
    that the SigLIP model would use for a given image. The model works by dividing
    an image into a grid of small "patches" (e.g., 16x16 pixels). This function
    calculates a new image size that maintains the original aspect ratio while ensuring
    the total number of patches does not exceed `max_num_patches`.

    Args:
        image_height (int): The original height of the image.
        image_width (int): The original width of the image.
        patch_size (int): The size of one square patch (e.g., 16).
        max_num_patches (int): The maximum allowed number of patches.

    Returns:
        tuple[int, int]: The target (height, width) for resizing.
    """
    # Helper function to calculate a scaled dimension, ensuring it's a multiple of the patch size.
    def get_scaled_image_size(scale: float, size: int, patch_size: int) -> int:
        scaled_size = size * scale
        # The ceiling division ensures the new size is a multiple of the patch size,
        # which is a requirement for the model's patch embedding.
        scaled_size = math.ceil(scaled_size / patch_size) * patch_size
        scaled_size = max(patch_size, scaled_size)
        return int(scaled_size)

    # Use a binary search to efficiently find the optimal scaling factor.
    # The goal is to find the largest possible scale where the number of patches
    # is less than or equal to the maximum allowed.
    scale_min, scale_max = eps / 10, 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_height = get_scaled_image_size(scale, image_height, patch_size)
        target_width = get_scaled_image_size(scale, image_width, patch_size)
        num_patches = (target_height / patch_size) * (target_width / patch_size)

        if num_patches <= max_num_patches:
            scale_min = scale  # If we have room, try a larger scale.
        else:
            scale_max = scale  # If we have too many patches, we must scale down.

    # After the search, scale_min holds the highest possible scale that meets the constraint.
    scale = scale_min
    final_target_height = get_scaled_image_size(scale, image_height, patch_size)
    final_target_width = get_scaled_image_size(scale, image_width, patch_size)
    return final_target_height, final_target_width

def create_exact_siglip_variant(image: Image.Image, max_num_patches: int = 1024, patch_size: int = 16) -> Image.Image:
    """
    Creates a thumbnail by resizing an image to the *exact* dimensions required
    by the SigLIP-2 vision model's preprocessing logic.

    This is different from a standard thumbnail. Instead of resizing to a fixed
    bounding box (like 512x512), it resizes the image so that when it's broken
    down into 16x16 pixel patches, the total patch count doesn't exceed 1024.
    This ensures that the saved image is identical to what the model would "see".

    Args:
        image (Image.Image): The input PIL Image object.
        max_num_patches (int): The maximum number of patches allowed by the model.
        patch_size (int): The dimension of each patch (e.g., 16 for a 16x16 patch).

    Returns:
        Image.Image: The resized PIL Image, ready for saving.
    """
    original_height, original_width = image.height, image.width

    # Call the replicated SigLIP logic to get the precise target dimensions.
    target_height, target_width = get_image_size_for_max_num_patches(
        image_height=original_height,
        image_width=original_width,
        patch_size=patch_size,
        max_num_patches=max_num_patches
    )

    # Perform the actual resize operation using the calculated target dimensions.
    # LANCZOS is used again as it's a high-quality resampling filter suitable for this task.
    siglip_image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
    return siglip_image


def get_image_embedding(image: Image.Image, model, processor, device) -> np.ndarray:
    """
    Generate embedding for a single image.
    """
    try:
        with torch.no_grad():
            inputs = processor(
                images=image,
                return_tensors="pt",
            ).to(device)

            image_features = model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            # Ensure the output is float32 for numpy compatibility, as bfloat16 is not well-supported
            return image_features.cpu().to(torch.float32).numpy()
    except Exception as e:
        print(f"Error during model inference: {e}")
        return np.array([])


def get_text_embedding(text: str, model, processor, device) -> np.ndarray:
    """
    Generate embedding for a single text string.
    """
    try:
        with torch.no_grad():
            inputs = processor(
                text=text,
                return_tensors="pt",
            ).to(device)

            text_features = model.get_text_features(**inputs)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            # Ensure the output is float32 for numpy compatibility, as bfloat16 is not well-supported
            return text_features.cpu().to(torch.float32).numpy()
    except Exception as e:
        print(f"Error during model inference: {e}")
        return np.array([])


def handle_connection(conn, model, processor):
    try:
        print("Waiting for new connection...")
        while True:
            # 1. Read message type
            print("Reading message type...")
            raw_msgtype = conn.recv(1)
            if not raw_msgtype:
                print("Connection closed by client.")
                break
            msgtype = struct.unpack('!B', raw_msgtype)[0]
            print(f"Received message type: {msgtype}")

            # 2. Receive the length of the payload
            print("Reading payload length...")
            raw_msglen = conn.recv(4)
            if not raw_msglen:
                print("Connection closed while reading payload length.")
                break
            msglen = struct.unpack('!I', raw_msglen)[0]
            print(f"Payload length: {msglen}")
            
            # 3. Receive the data
            print("Reading payload data...")
            data = bytearray()
            while len(data) < msglen:
                packet = conn.recv(msglen - len(data))
                if not packet:
                    print("Connection closed while reading payload data.")
                    break
                data.extend(packet)
            
            if len(data) < msglen:
                continue

            embedding = None
            
            # 4. Conditional Processing
            if msgtype == 0x01: # Text
                text_query = data.decode('utf-8')
                print(f"Processing text query: '{text_query}'")
                embedding = get_text_embedding(text_query, model, processor, DEVICE)[0]
                print("Text embedding generated.")
            
            elif msgtype == 0x02: # Image
                try:
                    print("Processing image data...")
                    # For debugging, save the incoming image data to a file
                    debug_dir = "/tmp/siglip-input"
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                    filename = f"{uuid.uuid4()}.jpg"
                    filepath = os.path.join(debug_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(data)

                    img_raw = Image.open(BytesIO(data)).convert("RGB")
                    img_resized = create_exact_siglip_variant(img_raw)
                    embedding = get_image_embedding(img_resized, model, processor, DEVICE)[0]
                except IOError as e:
                    print(f"Error opening image from binary data: {e}")

            # Send response back as before...
            if embedding is not None and embedding.size > 0:
                print(f"Sending embedding of size {embedding.size}.")
                conn.sendall(struct.pack('!I', embedding.size))
                conn.sendall(embedding.tobytes())
            else:
                print("Sending error response (0 size).")
                conn.sendall(struct.pack('!I', 0))

    finally:
        print("Closing connection.")
        conn.close()

def main():
    # Determine the torch dtype from the string argument
    torch_dtype = torch.float32
    if DTYPE == "bfloat16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print(f"Using bfloat16 precision on GPU.")
        else:
            # Check for CPU bfloat16 support
            torch_dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float32
            print(f"Using {DTYPE if torch_dtype == torch.bfloat16 else 'float32'} precision on CPU.")

    # Load the model and processor
    print(f"Loading model '{MODEL_NAME}' onto {DEVICE} with dtype {torch_dtype}...")
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype).to(DEVICE)
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        do_resize=False,
        max_num_patches=1024  # This MUST match the value used for generation
    )
    model.eval()
    print(f"Model loaded on {DEVICE} with dtype {model.dtype}.")

    # Make sure the socket does not already exist
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(128)
    print(f"SigLIP2 embedding server listening on {SOCKET_PATH}")

    try:
        while True:
            connection, client_address = server.accept()
            handle_connection(connection, model, processor)
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

if __name__ == '__main__':
    main()
