# Drone-Based-Crowd-Counting
This project is a crowd counting system that employs a deep learning approach, specifically a ResNet-18-based Convolutional Neural Network (CNN). It is designed to count the number of people in a crowd or video stream in real-time. The system uses a dataset of images with corresponding crowd count labels for training.

**Objective:**
The main objective of this project is to count the number of people in a crowd or a video stream in real-time.

**Key Components:**

**Dataset:** The project uses a dataset of images and their corresponding crowd counts. Each image is associated with a ground-truth crowd count obtained from annotation data.

**CrowdCountingDataset Class:** This custom dataset class is responsible for loading images and their crowd count labels from the dataset. It also applies data transformations to the images.

**CrowdCountingModel Class:** This custom deep learning model is based on the ResNet-18 architecture. It is used to learn and predict crowd counts from input images.

**Data Preprocessing:** Images are loaded, converted to RGB format, and resized to a consistent size (224x224 pixels). These preprocessed images are then fed into the model.

**Training Loop:** The model is trained using the dataset to learn the relationship between the input images and the crowd counts. The Mean Squared Error (MSE) loss is used for training.

**Real-time Crowd Counting:** After training, the model is used to perform real-time crowd counting on video frames from either a camera feed or a video file. It preprocesses each frame, feeds it to the model, and displays the crowd count on the frame.

**Saving and Loading Model Weights:** The trained model weights are saved to a file (crowd_counting_model_weights.pth) so that they can be loaded and reused for future inference without retraining.

**Usage:**

The project can be used to count crowds in real-time from a live camera feed or a video file.
The model can also be used to predict crowd counts for static images.

**Note:**

_Ensure that you have the required dataset, such as images and their corresponding crowd count annotations, to train and test the model._
_You can adjust hyperparameters like batch size, learning rate, and the number of training epochs to fine-tune the model's performance._
_Overall, this project provides a foundation for building a crowd counting system using deep learning, which can be useful in various applications like crowd management, event planning, and more._
