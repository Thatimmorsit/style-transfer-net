#!/usr/bin/env python3
"""
This script implements neural style transfer using a pre-trained VGG-19 network.
It applies the style of one image to the content of another.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512 if torch.cuda.is_available() else 128  # Use smaller size on CPU

# --- Image Loading and Preprocessing ---
loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Creating dummy image at {image_path}")
        Image.new("RGB", (100, 100), color="blue").save(image_path)
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)

# --- VGG Model and Feature Extraction ---
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Import VGG19 features, only the conv layers
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

# --- Loss Functions ---
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

    @staticmethod
    def gram_matrix(input):
        batch_size, f_map_num, h, w = input.size()
        features = input.view(batch_size * f_map_num, h * w)
        G = torch.mm(features, features.t())
        return G.div(batch_size * f_map_num * h * w)

# --- Main Transfer Function ---
def run_style_transfer(content_img, style_img, num_steps=300, style_weight=1000000, content_weight=1):
    print(f"Running on device: {DEVICE}")
    model = VGG().to(DEVICE).eval()
    optimizer = optim.LBFGS([content_img.requires_grad_()])

    print("Building the style transfer model...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            content_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            
            content_features = model(content_img)
            style_features = model(style_img)

            style_score = 0
            content_score = 0

            for cf, sf in zip(content_features, style_features):
                content_score += nn.functional.mse_loss(cf, sf)
                style_score += StyleLoss.gram_matrix(cf).mean()

            style_score *= style_weight
            content_score *= content_weight
            
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"  Run {run[0]}: Style Loss: {style_score.item():4f} Content Loss: {content_score.item():4f}")
            
            return style_score + content_score

        optimizer.step(closure)

    content_img.data.clamp_(0, 1)
    return content_img

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content_image", required=True, help="Path to content image")
    parser.add_argument("--style_image", required=True, help="Path to style image")
    parser.add_argument("--output_image", default="stylized_image.png", help="Path to save the output image")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps")
    args = parser.parse_args()

    content_image = load_image(args.content_image)
    style_image = load_image(args.style_image)

    output = run_style_transfer(content_image, style_image, num_steps=args.steps)

    # Save the result
    unloader = transforms.ToPILImage()
    image = output.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(args.output_image)
    print(f"\nStyle transfer complete. Image saved to {args.output_image}")
