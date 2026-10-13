# Bayer RAW Image Signal Processing (ISP) Pipeline

A Python-based Image Signal Processing (ISP) pipeline that transforms RAW DNG sensor data into a color-corrected, display-ready image, manages color spaces via a dedicated Chroma pipeline, and handles full JPEG compression and decoding.

This project implements the fundamental stages of a digital camera imaging pipeline, bridging raw hardware data capture with standard display formats.

Images were taken from the MIT-Adobe FiveK Dataset.

Camera of use: NIKON D70 30mm F/10.0 1/400 ISO200 6.0MP with BGGR mosaic pattern.

## Results & Performance

Here is a sample output comparison and memory footprint analysis from the pipeline execution *(Note: The output image appears slightly dark possibly due to lack of correct tone mapping, bugged code or incorrect CCM.)*
<img width="1000" height="1000" alt="Figure_1" src="https://github.com/user-attachments/assets/924f8aef-7826-4b15-929e-03c895630339" />


### Pipeline Data Footprint & Compression Metrics
- **Subsampled color plane size:** 35,874.38 KB
- **Full original RGB image size:** 17,937.19 KB
- **Compressed JPEG stream size:** 6,127.10 KB
- **Byte Ratio (vs Subsampled YCbCr):** 17.079%
- **Byte Ratio (vs Full Original RGB):** 34.159%

## Requirements

To run this project, make sure you have the following Python libraries installed:
```bash
pip install numpy matplotlib rawpy scikit-image scipy
```

## Pipeline Architecture

**RAW DNG → Black-Level & White-Level Normalization → Demosaicing → White Balance → Color Correction Matrix (CCM) → sRGB → Chroma Pipeline (YCbCr / Subsampling) → JPEG Encoding / Decoding**

## Key Features

- **RAW Sensor Processing:** Load DNG images, perform black-level correction, and apply white-level normalization.
- **Bayer Demosaicing:** Reconstruct full-resolution RGB images from single-channel Bayer-pattern sensor data using interpolation techniques.
- **White Balance:** Estimate and compensate for color casts using Gray World white-balance algorithms.
- **Color Correction:** Apply a Color Correction Matrix (CCM) to align camera RGB values with standard target color spaces.
- **Tone Mapping & sRGB Encoding:** Map dynamic ranges and convert linear sensor values into nonlinear sRGB values optimized for standard displays.
- **Chroma Pipeline:** Convert RGB data into luminance-chromance color spaces (YCbCr) and manage chroma subsampling to mimic human visual perception.
- **JPEG Encoder & Decoder:** Complete round-trip handling of JPEG compression, quantization, discrete cosine transform (DCT) steps, and pixel reconstruction.

## Quick Start

1. Install the required dependencies.
2. Place your input RAW DNG files in the input directory.
3. Configure settings in `Main.py` with your image path, note that exposure can be changed in line 44 of main.
4. Run `Main.py` to process the RAW file and generate the JPEG output.


## Implementation Note on Memory Management
The current sequential Python implementation retains intermediate arrays across pipeline steps in-memory. While ideal for debugging and step-by-step verification, future C++ optimizations will leverage in-place buffer reuse to minimize peak memory footprints during high-resolution batch processing.

## Future Plans & Optimization

- **C++ Porting:** Reimplementing the core pipeline stages in high-performance C++ using AVX vectorization and cache blocking to achieve real-time frame processing speeds.
- **Entropy Coding Extensions:** Integrating custom Run-Length Encoding (RLE) and Huffman entropy coding stages to further optimize lossless and lossy stream compression ratios beyond standard JPEG.

## Closing Notes

This project serves as a foundational study into core digital image processing architectures, bridging raw hardware sensor data with display-ready sRGB formatting and compression metrics.
