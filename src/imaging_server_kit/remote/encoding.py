import io
import base64
import numpy as np
import tifffile


def decode_contents(contents: str) -> np.ndarray:
    """Decodes base64 encoded image contents into a NumPy array.

    Args:
        contents (str): The base64 encoded string of the image.

    Returns:
        np.ndarray: The decoded image represented as a NumPy array.
    """
    return tifffile.imread(io.BytesIO(base64.b64decode(contents)))


def encode_contents(arr: np.ndarray) -> str:
    """Encodes a NumPy array image into a base64 string.

    Args:
        image (np.ndarray): The image to encode.

    Returns:
        str: The base64 encoded string representation of the image.
    """
    img_byte_arr = io.BytesIO()
    tifffile.imwrite(img_byte_arr, arr)

    return base64.b64encode(img_byte_arr.getvalue()).decode()


# def encode_contents_bytes(image: np.ndarray) -> bytes:
#     """Encodes a NumPy array image into bytes.

#     Args:
#         image (np.ndarray): The image to encode.

#     Returns:
#         bytes: The bytes representation of the image.
#     """
#     img_byte_arr = io.BytesIO()
#     tifffile.imwrite(img_byte_arr, image)

#     return base64.b64encode(img_byte_arr.getvalue())
