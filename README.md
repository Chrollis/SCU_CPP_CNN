# Handwritten Digit Recognition with CNN in C++/Qt

A comprehensive C++ implementation of Convolutional Neural Networks for MNIST digit classification, featuring both LeNet-5 and VGG16 architectures with a modern Qt-based graphical interface.

## What's New in Qt Version

- **Modern Qt GUI**: Complete migration from EasyX to Qt framework
- **Enhanced User Experience**: Professional interface with menus, toolbars, and status bars
- **Advanced Features**: Real-time training progress, model management, and batch processing
- **Cross-Platform Support**: Built with Qt for potential multi-platform compatibility
- **Binary Model Format**: Custom `.cnn` binary format for efficient model storage

## Features

### Neural Network Core
- **LeNet-5 Implementation**: Custom C++ implementation achieving **>99.8% accuracy** on MNIST
- **VGG16 Architecture**: Complete VGG16 implementation (provided as reference)
- **From-Scratch Implementation**: Custom CNN layers (convolutional, pooling, fully connected)
- **Multiple Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Tanh
- **Advanced Initialization**: Gaussian, Xavier, He initialization methods

### User Interface
- **Interactive Canvas**: Draw digits with customizable brush colors
- **Real-time Recognition**: Instant digit detection with bounding boxes
- **Image Import**: Support for PNG, JPG, JPEG, BMP formats
- **Undo/Redo System**: 16-step history for canvas operations
- **Training Monitor**: Real-time progress bars and accuracy tracking

### Model Management
- **Binary Model Format**: Custom `.cnn` format with magic number validation
- **Model Persistence**: Save and load trained models
- **Training Configuration**: Adjustable batch size, epochs, learning rates
- **Multiple Training Modes**: Simple and detailed training with progress monitoring

## Installation

### Pre-built Executable

1. Download the latest release from the [Releases page](https://github.com/Chrollis/SCU_CPP_CNN/releases)
2. Extract all files, maintaining the folder structure
3. Run `Mini-Digits.exe` (Qt version) or `graphics.exe` (legacy EasyX version)

### Building from Source

#### Prerequisites

- **Qt 5.15+** or **Qt 6.x**
- **Eigen3** library for linear algebra
- **OpenCV 4.x** for image processing
- **C++17** compatible compiler
- **CMake** (recommended) or QMake

#### Build Steps

```bash
# Clone the repository
git clone https://github.com/Chrollis/SCU_CPP_CNN.git
cd SCU_CPP_CNN

# Create build directory
mkdir build && cd build

# Configure with CMake (example)
cmake .. -DCMAKE_PREFIX_PATH=/path/to/qt -DOpenCV_DIR=/path/to/opencv
make -j4
```

## Quick Start Guide

1. Setup Model
- Choose **Model Type**: LeNet-5 (recommended) or VGG16
- **Create New Model**: File → New Model or load existing `.cnn` file
- Configure training parameters in the Model panel
2. Prepare Data
- Set paths to MNIST dataset files:
 - Train Data: `train-images.idx3-ubyte`
 - Train Labels: `train-labels.idx1-ubyte`
 - Test Data: `t10k-images.idx3-ubyte`
 - Test Labels: `t10k-labels.idx1-ubyte`
3. Train Model
- Click **Start Training** or use Train menu
- Monitor real-time progress with accuracy and loss
- Stop training anytime with **Stop Training**
4. Recognize Digits
- **Draw** on canvas or **Import** image file
- Click **Recognize** to detect and label digits
- **Export** recognized digits for further analysis

## Configuration

The application automatically saves settings to `config.ini`

## Development

### Key Technical Features

- **Custom CNN Implementation**: All layers implemented from scratch
- **Efficient Linear Algebra**: Eigen3 for matrix operations
- **Image Processing**: OpenCV for digit extraction and preprocessing
- **Multi-threading**: OpenMP support for parallel computation
- **Binary Serialization**: Custom format for model persistence

### File Formats

- **Models**: `.cnn` (custom binary format with magic number 1128)
- **Images**: PNG, JPG, JPEG, BMP
- **MNIST Data**: IDX3/IDX1 binary formats

## License

This project was developed as part of **Sichuan University C++ coursework**, demonstrating complete CNN implementation from scratch.
