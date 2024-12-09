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
            in_channels=3,
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
        self.fc_1 = nn.Linear(16 * 35 * 35, 120)
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
def load_model(model_path, num_classes=5):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(
        model_path, weights_only=True, map_location=torch.device('cpu')))
    lenet_model.eval()
    return lenet_model


model_path = '../../model/weights/lenet_model_[cassava_leaf_disease].pt'
absolute_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), model_path))
model = load_model(absolute_path)


def inference(input_img, model):
    img_size = 150
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img_new = img_transform(input_img)
    img_new = torch.unsqueeze(img_new, 0)
    with torch.no_grad():
        predicts = model(img_new)
    probabilities = nn.Softmax(dim=1)(predicts)
    p_max, y_hat = torch.max(probabilities, dim=1)
    return round(p_max.item() * 100, 2), y_hat.item()


# def get_name_class(path):
#     return os.listdir(path)


# name_classes = get_name_class(
#     'module6/week1/model/image_classification/cassavaleafdata/train')
name_classes = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']


def run():
    st.title('Cassava Leaf Disease Classification')
    st.subheader('Model: LeNet. Dataset: Cassava Leaf Disease')

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
            st.image(image)
    elif option == 'Use Demo Image':
        demo_img_path = '../test_images/cassava_leaf_disease/test-cbsd-1.jpg'
        absolute_img_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), demo_img_path))
        image = Image.open(absolute_img_path)
        p_max, y_hat = inference(image, model)
        st.image(image)
        st.success(
            f'The uploaded image is of the ***{name_classes[y_hat]}*** with {p_max}% probability.')
