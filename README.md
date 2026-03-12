# Neural Style Transfer

A neural style transfer model to apply the style of one image to another. This project uses a pre-trained VGG network to separate the content and style of images and combine them.

## Features

*   **VGG-19 Based**: Utilizes the VGG-19 network for feature extraction.
*   **Content and Style Separation**: Effectively separates the content of a target image and the style of a source image.
*   **High-Quality Results**: Generates high-quality stylized images.

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch
*   Pillow

### Installation

```bash
git clone https://github.com/Thatimmorsit/style-transfer-net.git
cd style-transfer-net
pip install -r requirements.txt
```

### Usage

```bash
python style_transfer.py --content_image /path/to/content.jpg --style_image /path/to/style.jpg
```
