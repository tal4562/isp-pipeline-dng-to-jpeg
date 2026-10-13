import ISP_Pipeline as ip
import Chroma_Pipeline as cp
import JPEG_Pipeline as jp
import numpy as np
import matplotlib.pyplot as plt
import rawpy


# Standard 8x8 Luminance Table
Q_Y = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
], dtype=np.float32)

# Standard 8x8 Chrominance Table (shared by Cb and Cr)
Q_CbCr = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)


def main(path, quality, block_size):
    with rawpy.imread(path) as raw:
        rgb_full = raw.postprocess(half_size=False, use_camera_wb=True)

    raw_image, wb, ccm = ip.process_raw_to_linear(path)


    demosiac_img = ip.nn_interpolation_numpy(raw_image)
    white_balanced_image = ip.white_balance(demosiac_img, wb)
    ccm_image = ip.color_correction_matrix(white_balanced_image, ccm)
    srgb_float = ip.linear_to_srgb(ccm_image, 1.3)
    srgb_8int = ip.clip_and_uint(srgb_float)


    ycbcr_img = cp.rgb_to_ycbcr_numpy(srgb_8int)


    # 1. Split and subsample your color image
    Y, Cb_sub, Cr_sub = cp.chroma_subsample_420(ycbcr_img)

    Y = Y.astype(np.float32)
    Cb_sub = Cb_sub.astype(np.float32)
    Cr_sub = Cr_sub.astype(np.float32)


    # 2. Compress EACH channel separately into its own 1D JPEG stream
    jpeg_Y = jp.im2jpeg(Y, block_size, quality, Q_Y)
    jpeg_Cb = jp.im2jpeg(Cb_sub, block_size, quality, Q_CbCr)
    jpeg_Cr = jp.im2jpeg(Cr_sub, block_size, quality, Q_CbCr)


    # 3. Decode EACH channel back from its 1D JPEG stream
    Y_dec = jp.jpeg2im(jpeg_Y, block_size, quality, Q_Y)
    Cb_dec = jp.jpeg2im(jpeg_Cb, block_size, quality, Q_CbCr)
    Cr_dec = jp.jpeg2im(jpeg_Cr, block_size, quality, Q_CbCr)


    # 4. Upsample chroma channels back to full Y resolution
    target_h, target_w = Y_dec.shape
    Cb_up, Cr_up = cp.chroma_upsample_420(Cb_dec, Cr_dec, target_h, target_w)

    # 5. Stack the channels back into a single YCbCr image array
    ycbcr_dec = np.stack([Y_dec, Cb_up, Cr_up], axis=-1).astype(np.uint8)

    # 6. Convert YCbCr back to final RGB
    rgb_dec = cp.ycbcr_to_rgb_numpy(ycbcr_dec)

    # 1. Memory usage in bytes (automatically accounts for uint8 vs int32 data types)
    compressed_bytes = jpeg_Y.nbytes + jpeg_Cb.nbytes + jpeg_Cr.nbytes

    subsampled_bytes = Y.nbytes + Cb_sub.nbytes + Cr_sub.nbytes
    full_rgb_bytes = rgb_full.nbytes  # H x W x 3 bytes

    print(f"Subsampled color plane size: {subsampled_bytes / 1024:.2f} KB")
    print(f"Full original RGB image size: {full_rgb_bytes / 1024:.2f} KB")
    print(f"Compressed JPEG stream size: {compressed_bytes / 1024:.2f} KB")

    # 2. True Byte-Level Compression Ratios
    subsampled_ratio = np.round(100 * compressed_bytes / subsampled_bytes, 3)
    full_pipeline_ratio = np.round(100 * compressed_bytes / full_rgb_bytes, 3)

    print(f"Byte Ratio (vs Subsampled YCbCr): {subsampled_ratio}%")
    print(f"Byte Ratio (vs Full Original RGB): {full_pipeline_ratio}%")

    imag = main(path, 1, 8)

    with rawpy.imread(path) as raw:
        original2 = raw.postprocess(
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16
        )

    original2 = original2.astype(np.float32) / 65535.0

    # --------------------------------------------------
    # Step 6: Compare images
    # --------------------------------------------------

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].imshow(original2)
    ax[0].set_title("Original: rawpy")
    ax[0].axis("off")

    ax[1].imshow(imag)
    ax[1].set_title("My Custom ISP: WB + CCM + Tone Mapping")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    return rgb_dec





