import streamlit as st
from PIL import Image

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding='same'
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs


@st.cache_resource
def load_model(model_path, num_classes=10):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(
        model_path, weights_only=True, map_location=torch.device('cpu')))
    lenet_model.eval()
    return lenet_model


model_path = '../../model/weights/lenet_model_[mnist_dataset].pt'
absolute_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), model_path))
model = load_model(absolute_path)


def inference(image, model):
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        wnew, hnew = image.size
        print(wnew, hnew)
    img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    img_new = img_transform(image)
    img_new = img_new.expand(1, 1, 28, 28)
    with torch.no_grad():
        preds = model(img_new)
    probabilities = nn.Softmax(dim=1)(preds)
    p_max, y_hat = torch.max(probabilities, dim=1)
    return round(p_max.item() * 100, 2), y_hat.item()


def run():
    st.title('Digit Recognition')
    st.subheader('Model: LeNet. Dataset: MNIST')

    option = st.selectbox(
        "How would you like to give input",
        ("Upload Image File", "Use Demo Image"),
    )
    if option == "Upload Image File":
        uploaded_file = st.file_uploader(
            "Please upload an image of a digit",
            type=['png', 'jpg'],
            accept_multiple_files=False
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            p_max, y_hat = inference(image, model)
            st.image(image)
            st.text(
                f'The uploaded image is of the digit {y_hat} with {p_max}% probability.')
    elif option == 'Use Demo Image':
        demo_img_path = '../test_images/mnist/demo_8.png'
        absolute_img_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), demo_img_path))
        image = Image.open(absolute_img_path)
        p_max, y_hat = inference(image, model)
        st.image(image)
        st.success(
            f'The uploaded image is of the digit {y_hat} with {p_max}% probability.')
