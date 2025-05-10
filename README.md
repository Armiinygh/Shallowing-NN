# Shallowing Neural Networks

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Last Commit](https://img.shields.io/github/last-commit/your-username/Shallowing-NN.svg)](https://github.com/your-username/Shallowing-NN/commits/main)
[![Issues](https://img.shields.io/github/issues/your-username/Shallowing-NN.svg)](https://github.com/your-username/Shallowing-NN/issues)

An implementation of Shallow and Deep Neural Networks for the seminar 'Understanding Deep Learning' in the Winter Semester 2025. This project explores and compares the capabilities and limitations of both shallow and deep neural networks using the MNIST dataset as a benchmark.

## Authors

- Seyedalireza Yaghoubi (syaghoubi@uni-osnabrueck.de)
- Lina Drapal (ldrapal@uni-osnabrueck.de)
- Elias Scheller (escheller@uni-osnabrueck.de)

## Project Structure

```plaintext
Shallowing-NN/
├── src/
│   ├── Config/
│   │   └── config.py         # Configuration parameters for both networks
│   ├── Plot/
│   │   └── plot.py          # Plotting utilities for visualizations
│   ├── Shallow_nn/
│   │   └── Shallow_nn.py    # Shallow neural network implementation
│   └── Deep_nn/
│       └── Deep_nn.py       # Deep neural network implementation
├── main.py                  # Entry point for running both networks
├── pyproject.toml          # Project dependencies and metadata
└── README.md               # Project documentation
```

## Project Description

This project implements and compares two neural network architectures for classifying handwritten digits from the MNIST dataset:

1. **Shallow Neural Network**
   - Single hidden layer architecture
   - Focus on understanding basic neural network principles
   - Simpler architecture for comparison baseline

2. **Deep Neural Network**
   - Multiple hidden layers
   - More complex architecture
   - Demonstrates the power of deep learning

### Key Features

1. **Network Architectures**
   - **Shallow Network:**
     - Input Layer: 784 neurons (28x28 flattened images)
     - Single Hidden Layer: Configurable size (default 256 neurons)
     - Output Layer: 10 neurons (digit classes 0-9)
     - ReLU activation function
     - Softmax output layer

   - **Deep Network:**
     - Input Layer: 784 neurons
     - Multiple Hidden Layers with configurable sizes
     - Output Layer: 10 neurons
     - Advanced activation functions
     - Dropout layers for regularization

2. **Training Pipeline**
   - Cross-entropy loss function
   - Multiple optimizer options (Adam, SGD)
   - Configurable batch size and epochs
   - Progress tracking with loss visualization
   - Separate training loops for shallow and deep networks

3. **Evaluation Metrics**
   - Test accuracy for both networks
   - Prediction confidence visualization
   - Correctness analysis
   - Loss tracking for both training and testing
   - Comparative analysis between networks

## Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/Shallowing-NN.git
cd Shallowing-NN
```

2. Install uv and project dependencies:
```bash
# Install uv
pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

## Usage

### Running the Models

1. To train and evaluate both models:
```bash
python main.py
```

2. The script will:
   - Download the MNIST dataset (first run only)
   - Train both shallow and deep networks
   - Generate performance visualizations for each network
   - Save the trained models

### Configuration

You can modify the model parameters in `src/Config/config.py`:

```python
# Common parameters
cfg.epochs = 5              # Number of training epochs
cfg.batch_size = 128        # Batch size
cfg.learning_rate = 0.00001 # Learning rate
cfg.optimization = "Adam"   # Optimizer choice (Adam or SGD)

# Shallow network parameters
cfg.shallow_output_size = 256  # Hidden layer size for shallow network

# Deep network parameters
cfg.deep_hidden_sizes = [512, 256, 128]  # Hidden layer sizes for deep network
cfg.dropout_rate = 0.2                   # Dropout rate for deep network
```

## Output and Visualizations

The project generates separate visualizations for each network type, saved in their respective directories:

### Shallow Network Visualizations (`src/Plot/shallow-nn/`)
1. **Training Loss** (`Train_loss_shallow.png`)
   - Shows loss progression during training
   - X-axis: Epochs
   - Y-axis: Loss value

2. **Test Loss** (`Test_loss_shallow.png`)
   - Shows loss on test dataset
   - X-axis: Epochs
   - Y-axis: Loss value

3. **Prediction Confidence** (`Confidence_shallow.png`)
   - Histogram of model's confidence in predictions
   - X-axis: Confidence value
   - Y-axis: Frequency

4. **Correctness Analysis** (`Correctness_shallow.png`)
   - Scatter plot showing relationship between confidence and correctness
   - X-axis: Sample index
   - Y-axis: Confidence value
   - Color: Correct (yellow) vs Incorrect (purple) predictions

### Deep Network Visualizations (`src/Plot/deep-nn/`)
Similar visualizations are generated for the deep network with corresponding filenames:
- `Train_loss_deep.png`
- `Test_loss_deep.png`
- `Confidence_deep.png`
- `Correctness_deep.png`

## Model Performance

### Shallow Network
- Training loss reduction from ~2.0 to ~0.89 over 5 epochs
- Test accuracy varies based on configuration
- Performance analysis available through generated visualizations

### Deep Network
- Typically achieves higher accuracy than shallow network
- More complex learning patterns
- Better feature extraction capabilities

## Future Improvements

1. **Architecture Enhancements**
   - Experiment with different activation functions (GELU, ELU)
   - Implement batch normalization layers
   - Add residual connections

2. **Training Optimizations**
   - Implement learning rate scheduling
   - Add early stopping mechanism
   - Implement cross-validation

3. **Code Structure**
   - Add comprehensive testing suite
   - Implement logging system
   - Add type hints and documentation

## References

### Shallow Neural Networks
1. Ba, J., & Caruana, R. (2014). [Do deep nets really need to be deep?](https://papers.nips.cc/paper/2014/hash/ea8fcd92d59581717e06eb187f10666d-Abstract.html) Advances in Neural Information Processing Systems, 27.
   - *Demonstrates that shallow networks can sometimes match deep network performance*

2. Lin, H. W., Tegmark, M., & Rolnick, D. (2017). [Why does deep and cheap learning work so well?](https://doi.org/10.1088/1751-8121/aa9a90) Journal of Statistical Physics, 168(6), 1223-1247.
   - *Analyzes the theoretical foundations of why neural networks work, including shallow architectures*

### MNIST and Image Classification
3. LeCun, Y., Cortes, C., & Burges, C. J. (2010). [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/).
   - *The original MNIST dataset paper, fundamental to understanding the benchmark*

4. Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). [Best practices for convolutional neural networks applied to visual document analysis](https://doi.org/10.1109/ICDAR.2003.1227801). ICDAR, 3(2003).
   - *Classic paper on neural network practices for image classification*

### Neural Network Visualization and Analysis
5. Zeiler, M. D., & Fergus, R. (2014). [Visualizing and understanding convolutional networks](https://doi.org/10.1007/978-3-319-10590-1_53). European Conference on Computer Vision (pp. 818-833).
   - *Important paper on visualizing and interpreting neural networks*

6. Montavon, G., Samek, W., & Müller, K. R. (2018). [Methods for interpreting and understanding deep neural networks](https://doi.org/10.1016/j.dsp.2017.10.011). Digital Signal Processing, 73, 1-15.
   - *Comprehensive review of neural network interpretation methods*

### Model Confidence and Calibration
7. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). [On calibration of modern neural networks](http://proceedings.mlr.press/v70/guo17a.html). International Conference on Machine Learning (pp. 1321-1330).
   - *Important paper on understanding neural network confidence*

8. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). [Simple and scalable predictive uncertainty estimation using deep ensembles](https://papers.nips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html). Advances in Neural Information Processing Systems, 30.
   - *Methods for estimating prediction uncertainty in neural networks*

### Optimization and Training
9. Glorot, X., & Bengio, Y. (2010). [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html). International Conference on Artificial Intelligence and Statistics (pp. 249-256).
   - *Fundamental paper on neural network initialization and training*

10. Smith, L. N. (2017). [Cyclical learning rates for training neural networks](https://doi.org/10.1109/WACV.2017.58). IEEE Winter Conference on Applications of Computer Vision (pp. 464-472).
    - *Modern approach to learning rate scheduling*

### Comparative Studies
11. Urban, G., Geras, K. J., Kahou, S. E., Aslan, O., Wang, S., Caruana, R., ... & Richardson, M. (2017). [Do deep convolutional nets really need to be deep and convolutional?](https://arxiv.org/abs/1603.05691) arXiv preprint arXiv:1603.05691.
    - *Analysis comparing shallow and deep architectures*

12. Arora, S., Bhaskara, A., Ge, R., & Ma, T. (2014). [Provable bounds for learning some deep representations](http://proceedings.mlr.press/v32/arora14.html). International Conference on Machine Learning (pp. 584-592).
    - *Theoretical analysis of representation learning in neural networks*

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

This project was developed as part of the "Understanding Deep Learning" seminar at the University of Osnabrück.
