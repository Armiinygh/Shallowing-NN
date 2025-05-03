# Shallowing Neural Networks

An implementation of Shallow Neural Networks for the seminar 'Understanding Deep Learning' in the Winter Semester 2025. This project explores the capabilities and limitations of shallow neural networks using the MNIST dataset as a benchmark.

## Authors

- Seyedalireza Yaghoubi (syaghoubi@uni-osnabrueck.de)
- Lina Drapal (ldrapal@uni-osnabrueck.de)
- Elias Scheller (escheller@uni-osnabrueck.de)

## Project Structure

The project follows a modular structure:

```plaintext
Shallowing-NN/
├── src/
│   ├── Config/
│   │   └── config.py         # Configuration parameters
│   ├── Plot/
│   │   └── plot.py          # Plotting utilities
│   └── Shallow_nn/
│       └── Shallow_nn.py    # Core neural network implementation
├── main.py                  # Entry point
├── pyproject.toml           # Project dependencies and metadata
└── README.md               # Project documentation
```

## Project Description

This project implements a shallow neural network architecture to classify handwritten digits from the MNIST dataset. The implementation includes:

- A feedforward neural network with one hidden layer
- Configurable network parameters (hidden layer size, learning rate, etc.)
- Training and evaluation pipelines
- Visualization tools for model performance

### Key Features

1. **Network Architecture**
   - Input Layer: 784 neurons (28x28 flattened images)
   - Hidden Layer: Configurable size (default 256 neurons)
   - Output Layer: 10 neurons (digit classes 0-9)
   - ReLU activation function
   - Softmax output layer

2. **Training Pipeline**
   - Cross-entropy loss function
   - Adam optimizer
   - Configurable batch size and epochs
   - Progress tracking with loss visualization

3. **Evaluation Metrics**
   - Test accuracy
   - Prediction confidence visualization
   - Correctness analysis
   - Loss tracking for both training and testing

## Installation

### Prerequisites
- Python 3.11 or higher
- Poetry (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/Shallowing-NN.git
cd Shallowing-NN
```

2. Install dependencies using Poetry:
```bash
poetry install
```

Alternatively, if you prefer using pip:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Running the Model

1. To train and evaluate the model with default settings:
```bash
python main.py
```

2. The model will:
   - Download the MNIST dataset (first run only)
   - Train for the specified number of epochs
   - Generate performance visualizations
   - Save the trained model

### Configuration

You can modify the model parameters in `src/Config/config.py`:

```python
cfg.epochs = 5              # Number of training epochs
cfg.batch_size = 128        # Batch size
cfg.learning_rate = 0.00001 # Learning rate
cfg.output_size = 256      # Hidden layer size
```

## Output and Visualizations

The project generates several visualizations to help understand the model's performance:

1. **Training Loss** (`src/Plot/Train_loss.png`)
   - Shows loss progression during training
   - X-axis: Epochs
   - Y-axis: Loss value

2. **Test Loss** (`src/Plot/Test_loss.png`)
   - Shows loss on test dataset
   - X-axis: Epochs
   - Y-axis: Loss value

3. **Prediction Confidence** (`src/Plot/Confidence.png`)
   - Histogram of model's confidence in predictions
   - X-axis: Confidence value
   - Y-axis: Frequency

4. **Correctness Analysis** (`src/Plot/Correctness.png`)
   - Scatter plot showing relationship between confidence and correctness
   - X-axis: Sample index
   - Y-axis: Confidence value
   - Color: Correct (yellow) vs Incorrect (purple) predictions

## Model Performance

The shallow neural network typically achieves:
- Training loss reduction from ~2.0 to ~0.89 over 5 epochs
- Test accuracy varies based on configuration
- Performance analysis available through generated visualizations

## Future Improvements

1. Architecture Enhancements:
   - Experiment with different activation functions
   - Try various hidden layer sizes
   - Implement dropout for regularization

2. Training Optimizations:
   - Learning rate scheduling
   - Early stopping
   - Cross-validation

3. Code Structure:
   - Modularize components into separate classes
   - Add comprehensive testing
   - Implement logging system

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
