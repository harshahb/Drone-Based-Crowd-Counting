import os
import cv2
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import ssl
import argparse
import sys

# Create an argument parser
parser = argparse.ArgumentParser(description='Crowd Counting')

# Add a command-line argument to specify the mode (train or infer)
parser.add_argument('--mode', type=str, default='train', help='Mode: train or infer')

# Parse the command-line arguments
args = parser.parse_args()


# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
# Define the dataset class
class CrowdCountingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) if f.startswith('IMG_')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.data_dir, 'images', image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        # Load and return the crowd count from corresponding ground-truth .mat file
        mat_filename = f'GT_IMG_{image_filename[4:-4]}.mat'
        mat_path = os.path.join(self.data_dir, 'ground-truth', mat_filename)
        annotation_data = scipy.io.loadmat(mat_path)
        image_info = annotation_data['image_info']
        crowd_count = image_info[0, 0]['number'][0, 0][0, 0]  # Replace with actual variable name
        crowd_count = float(crowd_count)  # Convert to float
        crowd_count = torch.Tensor([crowd_count])  # Convert to a tensor with the correct shape

        return image, crowd_count

# Define the crowd counting model
class CrowdCountingModel(nn.Module):
    def __init__(self):
        super(CrowdCountingModel, self).__init__()
        self.cnn = models.resnet50(weights="IMAGENET1K_V2")
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)  # Output layer with one neuron for count
        )

    def forward(self, x):
        return self.cnn(x)

# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL image
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if args.mode == 'train':

    print('The model is Training...')
    # Load your crowd counting dataset
    dataset = CrowdCountingDataset(data_dir='train_data', transform=transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model and optimizer
    model = CrowdCountingModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (you can adjust num_epochs and other settings)
    num_epochs = 10

    # Training code here
    for epoch in range(num_epochs):
        for images, crowd_counts in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.MSELoss()(outputs, crowd_counts.float())
            loss.backward()
            optimizer.step()

    # Save model weights after training
    torch.save(model.state_dict(), 'crowd_counting_model_weights.pth')
    print('Training complete ')
    sys.exit()
    

elif args.mode == 'infer':
    # Load the pre-trained model
    model = CrowdCountingModel()
    model.load_state_dict(torch.load('crowd_counting_model_weights.pth'))
    model.eval()

else:
    print("Invalid mode. Use '--mode train' for training or '--mode infer' for inference.")

# Video capture from camera (0) or video file ("your_video_file.mp4")
cap = cv2.VideoCapture(0)

while True:
    # Preprocess the frame and perform inference
    ret, frame = cap.read()  # Read a frame from the video feed

    if ret:
       # Preprocess the frame and perform inference
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame)
        crowd_count = model(frame_tensor.unsqueeze(0)).item()

        # Convert the frame back to BGR format for displaying with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the crowd count on the frame
        cv2.putText(frame, f"Crowd Count: {crowd_count:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with crowd count
        cv2.imshow("Crowd Counting", frame)

        # Press 'q' to exit the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        # Handle the case when reading a frame fails
        break


# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
