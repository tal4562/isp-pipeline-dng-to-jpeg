import numpy as np
from skimage import transform
import scipy.fftpack as fftpack


def load_quantization_table(q_table, quality, block_size):
    q = q_table * quality  # JPEG normalizing array and zig-zag reordering
    if block_size != 8:
        q = transform.resize(q, (block_size, block_size), preserve_range=True)
    return q


def quantize(q_table, block, quality, block_size):
    q = load_quantization_table(q_table, quality, block_size)
    return np.array((block / q).round().astype(np.int32))


def dequantize(q_table, block, quality, block_size):
    q = load_quantization_table(q_table, quality, block_size)
    return block * q

def zigzag(M, N):
    # M is the number of rows
    # N is the number of columns
    a = np.arange(M * N).reshape(M, N)
    inx_list = np.array([])
    a = np.fliplr(a)
    for i in range(1 - M, N):
        dummy = np.diagonal(a, offset=i)
        if i % 2 == 1:
            dummy = dummy[::-1]
        inx_list = np.append(inx_list, dummy)
    inx_list = np.int_(inx_list)
    inx_list = inx_list[::-1]
    order = inx_list.tolist()
    return order


def block_to_zigzag(block, order):
    a = block.flatten()
    zigzag = []
    for i in range(len(order)):
        zigzag.append(a[order[i]])
    return zigzag


def zigzag_to_block(zigzag, order):
    n = len(zigzag)
    n_sqrt = int(np.sqrt(n))
    block = np.zeros(n).reshape(n_sqrt, n_sqrt)
    for i in range(n):
        r = order[i] // n_sqrt
        c = order[i] % n_sqrt
        block[r, c] = zigzag[i]

    return block


def dct_2d(image):
    return fftpack.dctn(image, norm='ortho')


def idct_2d(image):
    image = fftpack.idctn(image, norm='ortho')

    image[image > 127] = 127
    return image


def im2jpeg(img, block_size, quality, table):
    h, w = img.shape

    # Calculate needed padding
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    # Pad the image
    padded_image = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
    padded_h, padded_w = padded_image.shape

    # End-Of-Block (EOB) symbol
    eob = 256
    jpeg_list = list()

    # Header: Prepend original dimensions
    jpeg_list.append(h)
    jpeg_list.append(w)

    # Find the zigzag order
    order = zigzag(block_size, block_size)

    # FIX: Loop over the PADDED image dimensions (padded_h / padded_w)
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            # 1. Extract block and center range: [0, 255] --> [-128, 127]
            block = np.copy(padded_image[i:i + block_size, j:j + block_size])
            block = block.astype(np.int32)
            block = block - 128

            # 2. Perform DCT
            block_dct = dct_2d(block)

            # 3. Quantize
            block_dct_q = quantize(table, block_dct, quality, block_size)

            # 4. Zigzag reorder
            block_dct_q_zig = block_to_zigzag(block_dct_q, order)

            # 5. Omit trailing zeros
            block_dct_q_zig_ommited = list(np.trim_zeros(block_dct_q_zig, 'b'))

            # 6. Add EOB at the end of the block
            block_dct_q_zig_ommited.append(eob)

            # 7. Extend the main list with this block's codes
            jpeg_list.extend(block_dct_q_zig_ommited)

    # Convert the list to a 1D array
    jpeg_array = np.array(jpeg_list, dtype=np.int32)
    return jpeg_array


def jpeg2im(jpeg_img, block_size, quality, table):
    h = jpeg_img[0]
    w = jpeg_img[1]
    jpeg_img = jpeg_img[2:]

    eob = 256
    order = zigzag(block_size, block_size)

    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    padded_h = h + pad_h
    padded_w = w + pad_w

    blocks_per_line = padded_w // block_size
    ends_of_blocks = np.where(jpeg_img == eob)[0]

    # Initialize with zeros to prevent random/green garbage artifacts
    npmat = np.zeros((padded_h, padded_w), dtype=np.float32)

    start = 0
    for block_index, end in enumerate(ends_of_blocks):
        if block_index >= blocks_per_line * (padded_h // block_size):
            break  # Prevent overflow if extra tokens exist

        zz_list = np.zeros(block_size * block_size, dtype=np.int32)
        num_elements = end - start
        zz_list[:num_elements] = jpeg_img[start:end]

        quant_matrix = zigzag_to_block(zz_list, order)
        dct_matrix = dequantize(table, quant_matrix, quality, block_size)
        block = idct_2d(dct_matrix)

        i = (block_index // blocks_per_line) * block_size
        j = (block_index % blocks_per_line) * block_size

        npmat[i:i + block_size, j:j + block_size] = block + 128
        start = end + 1

    # Clip and crop back to exact original dimensions
    npmat = np.clip(npmat, 0, 255)
    return npmat[:h, :w]

