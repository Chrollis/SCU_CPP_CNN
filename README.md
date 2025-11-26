# Handwritten Digit Recognition with CNN in C++

A complete C++ implementation of Convolutional Neural Networks for MNIST digit classification, featuring LeNet-5 and VGG16 architectures with an interactive graphical interface.

## Features

- **LeNet-5 Implementation**: Custom C++ implementation achieving **>99.8% accuracy** on MNIST test data
- **VGG16 Architecture**: Complete VGG16 implementation (provided as reference)
- **Interactive GUI**: Drawing canvas with real-time digit recognition
- **Multiple Training Modes**: Configurable training with different learning rates
- **Model Persistence**: Save and load trained models
- **Image Processing**: Support for both drawing and image file input
- **Cross-Platform**: Uses Eigen for linear algebra and OpenCV for image processing

## Architecture

### Neural Network Components
- **Convolutional Layers**: Custom implementation with configurable kernel size, stride, and padding
- **Pooling Layers**: Max pooling and average pooling support
- **Fully Connected Layers**: Complete forward/backward propagation
- **Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Tanh
- **Weight Initialization**: Gaussian, Xavier, He initialization methods

### Models Implemented
- **LeNet-5**: Optimized for MNIST digit recognition
- **VGG16**: Deep architecture (provided untrained due to computational requirements)

## Dependencies

- **Eigen3**: Linear algebra operations
- **OpenCV**: Image processing and computer vision
- **EasyX**: Graphical user interface (Windows)
- **C++17** with filesystem support

## Quick Start

### Using Pre-trained Models

1. Download the latest release from the [Releases page](https://github.com/Chrollis/SCU_CPP_CNN/releases)
2. Extract the files, maintaining the folder structure
3. Run `graphics.exe` for the graphical interface

### Graphical Interface Usage

- **Draw**: Use the canvas to draw digits freehand
- **Detect**: Click "Detect" to recognize drawn digits
- **Import**: Use "Import" to load and recognize `target.jpg` from the same directory
- **Tools**: Undo, redo, brush, rubber, and clean functions available

### Training Models

Two training executables are provided:

```bash
# Standard training (learning rate = 0.0001)
Training.6e4.1.100.1e-4.1.exe

# Fine-tuning training (learning rate = 0.00001)  
Training.6e4.1.100.1e-5.1.exe
