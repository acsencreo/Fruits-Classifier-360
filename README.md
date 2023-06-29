# Fruits-Classifier-360

## Overview
The Fruit Classifier 360 is a deep learning model based on the MobileNetV2 architecture. It is trained to classify images into various types of fruits based on the Fruits 360 dataset. The model can classify 24 different kinds of fruits and their different types.

## Dataset
The training dataset consists of a collection of labeled images of different fruits and their different types. The dataset is organized into separate folders for each class, with different proportions of test, train, and validation sets. The dataset can be obtained from [Kaggle - Fruits 360](https://www.kaggle.com/datasets/moltean/fruits).

## Model Training
The model training process involves the following steps:

1. Data Preprocessing: The images are preprocessed using common data augmentation techniques, such as random cropping, resizing, and normalization.

2. Model Architecture: The MobileNetV2 architecture, pre-trained on the ImageNet dataset, is used as the base model. The final classification layer is replaced to match the number of classes in the dataset.

3. Training: The model is trained using the labeled images from the dataset. The training process involves optimizing the model's parameters using a specified optimizer and loss function.

4. Evaluation: The trained model is evaluated on a separate validation set to assess its performance.

5. Model Export: Once trained, the model weights are saved for future use. The saved model is provided in the repository for quick testing of the classification model.

## Usage
To train the Fruit Classifier 360 model, follow these steps:

1. Import all the required libraries included in the first cell and install the necessary requirements provided in the `requirements.txt` file (if not already installed).

2. Organize your dataset into the appropriate folder structure, with separate folders for each class under the train, test, and validation directories.

3. Adjust the file paths in the training script `train.py` to match your dataset directory structure.

4. Run the training cell with the desired number of epochs. Observe the loss function decreasing and the accuracy improving with each epoch.

5. After training and saving your model, run the image classification cell. Provide an appropriate image source path and execute the cell to classify the image.

Note: If you are using the provided notebook in Google Colab, the required libraries are preinstalled, so you only need to import them.
