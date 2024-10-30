# Bitcoin Price Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict Bitcoin prices based on historical data. The model is designed to analyze past Bitcoin prices and forecast future values, leveraging deep learning techniques for sequence prediction. 

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Bitcoin prices have shown a high degree of volatility, making it challenging to forecast future prices accurately. This project uses LSTM, a type of recurrent neural network (RNN) well-suited for sequential data, to attempt to predict Bitcoin's closing prices.

The model:
- Takes historical Bitcoin data as input.
- Preprocesses the data by scaling it.
- Feeds the data through an LSTM network.
- Outputs predictions, which are then inverse-transformed back to the original price scale.

## Features

- Uses LSTM layers for effective sequence prediction.
- Handles data preprocessing, including MinMax scaling.
- Visualizes actual vs. predicted prices for performance evaluation.
- Includes a mean squared error evaluation metric for model performance.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/lashmanv/Bitcoin-Price-Prediction-using-LSTM.git
    cd Bitcoin-Price-Prediction-using-LSTM
    ```

2. Install the required dependencies:

    ```bash
    pip install numpy pandas matplotlib tensorflow
    ```

3. Obtain an API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) and replace `"YOUR_API_KEY"` in the code with your actual API key.

## Usage

1. **Load Data**: The script loads daily historical Bitcoin prices from Alpha Vantage. 

2. **Data Preprocessing**: The code scales the data to the range of `[0, 1]` using MinMaxScaler, which helps the model converge more efficiently.

3. **Create Sequences**: The script generates sequences of 60 days for the LSTM to use for training and testing. You can adjust this length in the code.

4. **Train the Model**: The model is trained on 80% of the data, with the remaining 20% used for testing.

5. **Run the Prediction**:
   
    ```bash
    python bitcoin_lstm.py
    ```

6. **Visualize Results**: The script outputs a plot showing both the actual and predicted Bitcoin prices over time, allowing you to evaluate the model's performance.

## Model Architecture

The LSTM model includes:

- Three LSTM layers with 50 units each.
- Dropout layers between LSTM layers to reduce overfitting.
- A Dense layer for output prediction.

The model is compiled with `mean_squared_error` as the loss function and `Adam` as the optimizer.

## Results

The model's performance is visualized by plotting the predicted and actual Bitcoin prices. Additionally, a Mean Squared Error (MSE) score is printed to quantify the model's accuracy.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request if you have any ideas to improve the model or add new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Sample Output

The final plot output includes two lines:
- **Actual Bitcoin Price**: The real prices.
- **Predicted Bitcoin Price**: The prices predicted by the model.

For best results, experiment with different sequence lengths, batch sizes, and LSTM configurations to optimize model performance.
