"""
Module for creating the streamlit web app using the ML/DL model for the Bengali deepfake detection
"""

import sys
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.preprocessing import StandardScaler
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# Model Filepath
model_filepath = "audio_cnn_model.pth"

# Defining Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mfcc(data, sample_rate=16000, target_length_seconds=6):
    """
    Returns the MFCC feature for an audio file
    Args:
    filepath: Audio file path, string
    sample rate:  Audio sample rate, in Hz
    target_length_seconds: Number of seconds of data consider of feature extraction
    """
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=20).to(device)
    target_length_samples = target_length_seconds * sample_rate

    scaler = StandardScaler()
    
    data = data.to(device)  # Move data to GPU
    
    # Crop or pad the audio to the target length
    num_samples = data.shape[1]

    if num_samples > target_length_samples:
        start = (num_samples - target_length_samples) // 2
        data = data[:, start : start + target_length_samples]
    else:
        pad_total = target_length_samples - num_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        data = torch.nn.functional.pad(data, (pad_left, pad_right))

    # Perform MFCC on GPU
    mfcc = mfcc_transform(data)

    # Move MFCC back to CPU for scaling
    mfcc = mfcc.cpu().numpy()

    # Flatten the MFCC features across time and frequency for scaling
    flattened_mfcc = mfcc.reshape(-1, mfcc.shape[-1])
    scaled_mfcc = scaler.fit_transform(flattened_mfcc)

    # Reshape back to the original MFCC shape and convert to tensor
    scaled_mfcc = torch.tensor(scaled_mfcc.reshape(mfcc.shape), device=device)

    return scaled_mfcc

def extract_feature(audio_file, chunk_dur = 6):
    """
    Extract the MFCC from the whole audio file
    Returns MFCC features from 6 second chunks
    args: audio file
    """
    # read the time series and sample rate from the audio file
    ydat, samp_rate = torchaudio.load(audio_file) #ydat is returned in [chan, samples] format
    
	#No. of samples in each sliced chunk
    len_samp = int(samp_rate*chunk_dur)

    #check number of 6 second files in there
    nslices = int(np.floor(ydat.shape[1]/len_samp))

    print(f" Number of {chunk_dur} sec slices: {nslices}")
	
    feat_list = []

    if nslices == 0:
        feat = get_mfcc(ydat, samp_rate, target_length_seconds=chunk_dur)
        feat_list.append(feat)
    else:
        #create multiple MFCC features based on the number of audio slices
        #st.write(text=f"Processing {nslices} six second chunks.")
        for i in range(nslices):
            # Getting each slice of data
            yslice = ydat[:, i*len_samp:(i+1)*len_samp]
            
            feat = get_mfcc(yslice, samp_rate, target_length_seconds=chunk_dur)
            feat_list.append(feat)
            
    return torch.stack(feat_list)



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(128 * 2 * 60, 512)  # Adjust input size based on feature map size after pooling
        self.fc2 = nn.Linear(512, 2)  # Assuming binary classification (real vs fake)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def detect_deepfake(model_path, feat_list):

    """
    Returns the prediction for a set of mel spectrograms using the model

    args:
    model_path: model file path, string
    feat_list : list of spectrograms from an audio file
    """
    
    # Load the saved model
    model = torch.load(model_filepath)
    model.to(device)  # Move the model to the device (GPU if available)

    # setting the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Forward pass: Get predictions
        pred = model(feat_list)
    
    #print(f" predicted values: {pred}")
    # Assume this is a classification task and model output is logits
    pred_prob = torch.softmax(pred, dim=1)  # Convert logits to probabilities
    print(pred_prob)
    pred_class = torch.argmax(pred_prob, dim=1)  # Get the class with the highest probability
    print(pred_class)

    # Label 1 for bonafide and 0 for fake audio
    print("Predicted class:", pred_class.item())  # For a single prediction

    if pred_class.any()  == 0:
        return 0
    else:
        return 1
    

def main():

    # Title
    st.title("Welcome to the AudioShield: Leveraging AI to detect deepfake audio")
    st.header("Deep learning based detection system for Bengali language")
    
    image = Image.open("deepfake-logo.png")
    st.image(image)


    file_uploaded = st.file_uploader("Upload the audio file for detection", type=["flac","wav","mp3"])

    click = st.button("Predict")

    if click:

        feature_list = extract_feature(file_uploaded)
        output = detect_deepfake(model_filepath, feature_list)
        
        if output == 1:
            result = 'Genuine'
        else:
            result = 'Fake'
        st.write(f"The Audio file is {result}")


if __name__ == '__main__':
    
    # Run the streamlit function
    main()
    
    #file_uploaded = sys.argv[1]
    #feature_list = extract_feature(file_uploaded)
    #output = detect_deepfake(model_filepath, feature_list)
    #print(output)
