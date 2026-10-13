import numpy as np

def rgb_to_ycbcr_numpy(rgb_img_uint8):
    # 1. Convert 8-bit uint8 to float32 to allow negative values and fractions
    rgb = rgb_img_uint8.astype(np.float32)

    # 2. Define ITU-R BT.601 conversion matrix
    transform_matrix = np.array([
        [0.299000, 0.587000, 0.114000],
        [-0.168736, -0.331264, 0.500000],
        [0.500000, -0.418688, -0.081312]
    ])

    # 3. Reshape image from (H, W, 3) to (N, 3) for efficient matrix multiplication
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3)

    # 4. Dot product: pixels @ matrix.T
    # (Note: Transposing the matrix aligns standard row vector multiplication)
    ycbcr_pixels = np.dot(pixels, transform_matrix.T)

    # 5. Add the +128 DC offset to Cb and Cr channels (columns 1 and 2)
    ycbcr_pixels[:, 1:] += 128.0

    # 6. Clip values strictly to the valid 8-bit range [0, 255] and cast to uint8
    ycbcr_pixels = np.clip(ycbcr_pixels, 0.0, 255.0)
    ycbcr_img = ycbcr_pixels.reshape(h, w, 3).astype(np.uint8)

    return ycbcr_img



def chroma_subsample_420(ycbcr_img):
    """
    Performs 4:2:0 chroma subsampling on a YCbCr image.
    Y remains full size. Cb and Cr are averaged over 2x2 blocks and downsampled.
    """
    h, w, _ = ycbcr_img.shape

    # 1. Y channel stays full resolution
    y_channel = ycbcr_img[:, :, 0]

    # Extract Cb and Cr planes
    cb_channel = ycbcr_img[:, :, 1].astype(np.float32)
    cr_channel = ycbcr_img[:, :, 2].astype(np.float32)

    # 2. 2x2 Block Averaging (Mean Pooling) for Cb and Cr
    # Reshape into blocks of (H/2, 2, W/2, 2) and take the mean across axes 1 and 3
    cb_subsampled = cb_channel.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    cr_subsampled = cr_channel.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    # Cast back to uint8
    cb_subsampled = np.clip(cb_subsampled, 0, 255).astype(np.uint8)
    cr_subsampled = np.clip(cr_subsampled, 0, 255).astype(np.uint8)

    return y_channel, cb_subsampled, cr_subsampled


def chroma_upsample_420(Cb_sub, Cr_sub, target_h, target_w):
    Cb_up = np.repeat(np.repeat(Cb_sub, 2, axis=0), 2, axis=1)
    Cr_up = np.repeat(np.repeat(Cr_sub, 2, axis=0), 2, axis=1)

    # Strictly enforce target dimensions to avoid 1-pixel off-by-one green borders
    return Cb_up[:target_h, :target_w], Cr_up[:target_h, :target_w]


def ycbcr_to_rgb_numpy(ycbcr_img_uint8):
    # 1. Convert 8-bit uint8 to float32 to prevent clipping during math
    ycbcr = ycbcr_img_uint8.astype(np.float32)

    # 2. Reshape image from (H, W, 3) to (N, 3) for matrix multiplication
    h, w, _ = ycbcr.shape
    pixels = ycbcr.reshape(-1, 3)

    # 3. Subtract the +128 DC offset from Cb and Cr channels back
    pixels[:, 1:] -= 128.0

    # 4. Define ITU-R BT.601 inverse conversion matrix
    inverse_matrix = np.array([
        [1.000000,  0.000000,  1.402000],
        [1.000000, -0.344136, -0.714136],
        [1.000000,  1.772000,  0.000000]
    ])

    # 5. Dot product: pixels @ inverse_matrix.T
    rgb_pixels = np.dot(pixels, inverse_matrix.T)

    # 6. Clip values strictly to the valid 8-bit range [0, 255] and cast back to uint8
    rgb_pixels = np.clip(rgb_pixels, 0.0, 255.0)
    rgb_img = rgb_pixels.reshape(h, w, 3).astype(np.uint8)

    return rgb_img