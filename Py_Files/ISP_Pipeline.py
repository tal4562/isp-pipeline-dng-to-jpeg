import rawpy
import numpy as np


# --------------------------------------------------
# Step 1: Black-level correction & normalization
# --------------------------------------------------
def process_raw_to_linear(path):
    '''
    Step One: convert DNG file into linear light image to mimic sensor capture
    :param path: DNG image path
    :return: Linear light image as float 32
    '''
    with rawpy.imread(path) as raw:
        # 1.1: Load raw image
        raw_image = raw.raw_image_visible.copy().astype(np.float32)

        # 1.2: Extract camera's parameters - black / white levels and the white balance
        black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white_level = np.float32(raw.white_level)
        wb = np.asarray(raw.camera_whitebalance, dtype=np.float32)[:3]

        # 1.3: Extract and normalize the Color Correction Matrix (CCM)
        ccm = raw.color_matrix[:3, :3].astype(np.float32)
        ccm = ccm / ccm.sum(axis=1, keepdims=True)

        # 1.4: BGGR pattern normalization
        # B  G1
        # G2 R
        b_bl, g1_bl, g2_bl, r_bl = black_levels

        # Blue
        raw_image[0::2, 0::2] = (raw_image[0::2, 0::2] - b_bl) / (white_level - b_bl)
        # First Green
        raw_image[0::2, 1::2] = (raw_image[0::2, 1::2] - g1_bl) / (white_level - g1_bl)
        # Second Green
        raw_image[1::2, 0::2] = (raw_image[1::2, 0::2] - g2_bl) / (white_level - g2_bl)
        # Red
        raw_image[1::2, 1::2] = (raw_image[1::2, 1::2] - r_bl) / (white_level - r_bl)

    # 1.5: Clamp lower bound to 0.0
    raw_image = np.maximum(raw_image, 0.0)

    return raw_image, wb, ccm


# --------------------------------------------------
# Step 2: BGGR nearest-neighbor demosaicing
# --------------------------------------------------
def nn_interpolation_numpy(mosaic_img: np.ndarray) -> np.ndarray:
    '''
    Step Two: Nearest Neighbour interpolation
    Converts Bayer's mosiac BGGR image into a color image
    :param mosaic_img: Bayer's mosiac dng image of the form BGGR
    :return: Demosiac color image
    '''
    # 2.1: Extract shape
    rows, cols = mosaic_img.shape

    # 2.2: Pad to deal with odd lengths
    padded = np.pad(mosaic_img,((0, rows % 2), (0, cols % 2)),mode="reflect")

    # 2.3: Create output color array
    h, w = padded.shape
    demosaic = np.zeros((h, w, 3), dtype=np.float32)

    # 2.4: Blue: replicate each blue sample across a 2x2 block.
    b = padded[0::2, 0::2]
    demosaic[0::2, 0::2, 2] = b
    demosaic[0::2, 1::2, 2] = b
    demosaic[1::2, 0::2, 2] = b
    demosaic[1::2, 1::2, 2] = b

    # 2.5: Green: top-right and bottom-left samples.
    g1 = padded[0::2, 1::2]
    demosaic[0::2, 0::2, 1] = g1
    demosaic[0::2, 1::2, 1] = g1

    g2 = padded[1::2, 0::2]
    demosaic[1::2, 0::2, 1] = g2
    demosaic[1::2, 1::2, 1] = g2

    # 2.6: Red: replicate each red sample across a 2x2 block.
    r = padded[1::2, 1::2]
    demosaic[0::2, 0::2, 0] = r
    demosaic[0::2, 1::2, 0] = r
    demosaic[1::2, 0::2, 0] = r
    demosaic[1::2, 1::2, 0] = r

    # 2.7: Clip back to the original dimension
    return demosaic[:rows, :cols, :]


# --------------------------------------------------
# Step 3: White Balance
# --------------------------------------------------
def white_balance(demosiac_img: np.ndarray, wb: np.ndarray) -> np.ndarray:
    '''
    Step Three: Apply channel-specific white balance gains to a demosaiced
    linear RGB image.
    :param demosiac_img: Demosiac color image
    :param wb: White Balance levels
    :return: White balanced image
    '''
    # 3.1: Normalize the green gain to 1.0
    wb = wb / wb[1]
    # 3.2: apply the white balance to each channel
    balanced = demosiac_img * wb.reshape(1, 1, 3)
    return balanced


# --------------------------------------------------
# Step 4: Color Correction Matrix
# --------------------------------------------------
def color_correction_matrix(white_balanced_img: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    '''
    Step Four: Color Correction
    :param white_balanced_img: White balanced linear RGB image in float 32
    :param ccm: Color Corrretion Matrix of size 3x3
    :return: Color corrected image in float 32
    '''
    # 4.1: Extract dimensions
    h, w, _ = white_balanced_img.shape
    # 4.2: Reshape image from (H, W, 3) to a flat list of 3-element pixel vectors (N, 3)
    pixels = white_balanced_img.reshape(-1, 3)
    # 4.3: Apply the 3x3 CCM via matrix multiplication.
    ccm_corrected = np.dot(pixels, ccm.T).reshape(h, w, 3)
    return ccm_corrected


# --------------------------------------------------
# Step 5: Exposure and Linear RGB to sRGB encoding
# --------------------------------------------------
def linear_to_srgb(linear_rgb:  np.ndarray, exposure = 1.0):
    '''
    Converts linear light RGB to non-linear sRGB display space.
    :param linear_rgb: Linear float32 array in range [0.0, 1.0]
    :param exposure: Linear multiplier to adjust image brightness, default value is 1.0
    :return: sRGB image in float32
    '''
    linear_rgb = np.maximum(linear_rgb * exposure, 0.0)
    return np.where(linear_rgb <= 0.0031308,12.92 * linear_rgb,1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055)


# --------------------------------------------------
# Step 6: Uint 8 and clipping
# --------------------------------------------------
def clip_and_uint(linear_srgb: np.ndarray) -> np.ndarray:
    # Final Clipping (Done EXACTLY ONCE at the very end)
    srgb = np.clip(linear_srgb, 0.0, 1.0)
    srgb_8bit = (srgb * 255.0).astype(np.uint8)
    return srgb_8bit
