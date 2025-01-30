# coding: utf-8

import os
import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
import scipy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def create_directories(dirs):
    """Create necessary directories"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def convert_mat_to_png(mat_dir, save_dir):
    """Convert .mat files to PNG format"""
    # Get all .mat files
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
    mat_files.sort(key=lambda x: int(x.split('-')[0]))  # Sort by number

    for mat_file in mat_files:
        # Read .mat file
        mat_path = os.path.join(mat_dir, mat_file)
        mat_data = sio.loadmat(mat_path)["data"].astype(np.float32)

        # Normalize to [0, 255]
        mat_data = (mat_data - mat_data.min()) / (mat_data.max() - mat_data.min()) * 255
        mat_data = mat_data.astype(np.uint8)

        # Convert to PNG and save
        img = Image.fromarray(mat_data)
        save_path = os.path.join(save_dir, mat_file.replace('.mat', '.png'))
        img.save(save_path)
        print(f"Converted: {mat_file} -> {os.path.basename(save_path)}")

def resize_and_save_labels(label_dir, save_dir, target_size=(256, 256)):
    """Resize and save label images"""
    # Get all PNG files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
    label_files.sort(key=lambda x: int(x.split('-')[0]))  # Sort by number

    for label_file in label_files:
        # Read label image
        label_path = os.path.join(label_dir, label_file)
        label_img = Image.open(label_path).convert('L')

        # Resize image
        resized_img = label_img.resize(target_size, Image.LANCZOS)

        # Save resized image
        save_path = os.path.join(save_dir, label_file)
        resized_img.save(save_path)
        print(f"Resized and saved: {label_file}")

class SpeckleDataset(Dataset):
    def __init__(self, train_dir, label_dir, transform=None):
        self.train_dir = train_dir
        self.label_dir = label_dir
        self.transform = transform

        # Get all training image filenames
        self.image_files = [f for f in os.listdir(train_dir) if f.endswith('.png')]
        self.image_files.sort(key=lambda x: int(x.split('-')[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Read training image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.train_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        # Read image and label
        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

class UNetWithAttention(nn.Module):
    def __init__(self):
        super(UNetWithAttention, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Center layer
        self.center = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.up_conv(1024, 512)
        self.dec3 = self.up_conv(512, 256)
        self.dec2 = self.up_conv(256, 128)
        self.dec1 = self.up_conv(128, 64)

        # Attention mechanism
        self.attention4 = self.attention_block(512)
        self.attention3 = self.attention_block(256)
        self.attention2 = self.attention_block(128)
        self.attention1 = self.attention_block(64)

        # Output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            self.conv_block(out_ch, out_ch)
        )

    def attention_block(self, ch):
        return nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))

        # Center layer
        center = self.center(nn.MaxPool2d(2)(e4))

        # Decoding path (with attention mechanism)
        d4 = self.dec4(center)
        d4 = d4 * self.attention4(e4)

        d3 = self.dec3(d4)
        d3 = d3 * self.attention3(e3)

        d2 = self.dec2(d3)
        d2 = d2 * self.attention2(e2)

        d1 = self.dec1(d2)
        d1 = d1 * self.attention1(e1)

        out = self.sigmoid(self.final(d1))
        return out

def visualize_results(model, val_loader, device, epoch, save_dir='results'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Get a batch of data
        inputs, labels = next(iter(val_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Select first sample for visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Display input image
        axes[0].imshow(inputs[0,0].cpu().numpy(), cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Display ground truth
        axes[1].imshow(labels[0,0].cpu().numpy(), cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # Display prediction
        axes[2].imshow(outputs[0,0].cpu().numpy(), cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        plt.suptitle(f'Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'result_epoch_{epoch}.png'))
        plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Calculate average loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Print training information
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')

        # Visualize current results
        visualize_results(model, val_loader, device, epoch+1)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def predict_single_image(model_path, mat_path, save_dir):
    """
    Predict a single .mat image and save the result

    Parameters:
    model_path: Path to the model file
    mat_path: Path to the input .mat file
    save_dir: Directory to save results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = UNetWithAttention().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Read .mat file
    mat_data = sio.loadmat(mat_path)["data"].astype(np.float32)

    # Preprocessing - same steps as during training
    # Normalize to [0, 255]
    mat_data = (mat_data - mat_data.min()) / (mat_data.max() - mat_data.min()) * 255
    mat_data = mat_data.astype(np.uint8)

    # Convert to PIL Image
    img = Image.fromarray(mat_data)

    # Apply same transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Transform image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.squeeze().cpu().numpy()

    # Post-processing - convert prediction to image
    prediction = (prediction * 255).astype(np.uint8)
    prediction_img = Image.fromarray(prediction)

    # Save prediction result
    # Get original filename (without path) and replace .mat with .png
    save_filename = os.path.basename(mat_path).replace('.mat', '.png')
    save_path = os.path.join(save_dir, save_filename)
    prediction_img.save(save_path)

    print(f"Prediction saved as: {save_path}")

    return save_path
    
#predict
model_path = 'best_model.pth'
mat_path = '/content/drive/MyDrive/data/Speckle_mat_Val/10-label-5.mat'
save_dir = 'predictions'

predicted_image_path = predict_single_image(model_path, mat_path, save_dir)