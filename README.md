# Recurrent Neural Networks (RNNs)

## Introduction
In this lecture, the focus is on sequence modeling problems in deep learning. We revisit the basics of neural networks and introduce how to handle sequential data. We illustrate with a simple example: predicting the next position of a moving ball, highlighting the importance of historical data in sequence modeling.

Sequential data is ubiquitous, spanning audio waveforms, written language, medical readings, stock prices, and biological sequences. The lecture categorizes sequence modeling into three types:
- Single input, sequential output (e.g., image captioning).
- Sequential input, single output (e.g., sentiment analysis of a sentence).
- Sequential input, sequential output (e.g., language translation).

## Perceptron and Sequence Modeling
To build models for these tasks, we begin with a perceptron model, emphasizing intuition. The perceptron processes inputs via a linear combination and a non-linear activation function, extendable to layers of neurons.

However, this basic model lacks temporal understanding. To address this, we propose applying the model iteratively across time steps. Yet, this simplistic approach ignores time-step dependencies.

## Introducing Recurrent Neural Networks (RNNs)
To address these shortcomings, we introduce recurrent neural networks (RNNs). RNNs maintain a state variable \( H_t \), updated at each time step, capturing and propagating information from previous time steps. This memory enables the model to learn dependencies across sequences.

### RNN Structure
- **State Update**: \( H_t = \tanh(W_h \cdot H_{t-1} + W_x \cdot X_t + b) \), where \( W_h \) and \( W_x \) are weight matrices for the hidden state and input, respectively.
- **Output Generation**: Outputs at each time step depend on the current state \( H_t \).

### Training RNNs
- **Forward Pass**: Sequentially process inputs to update state and generate outputs.
- **Loss Calculation**: Compute losses per time step, aggregating for total sequence loss.
- **Backpropagation Through Time (BPTT)**: Update weights using gradients propagated backwards through time.

### Practical Implementation
- **Initialization**: Initialize hidden state.
- **Processing**: Iterate through input sequences, updating state and generating outputs.
- **Update**: Use outputs to predict subsequent sequence elements and update weights based on loss.

### Visualization
RNNs are visualized in two primary forms:
- **Loop Function**: Depicts the network as a loop, updating state iteratively.
- **Unrolling**: Shows network computations over time steps, detailing state and output computations.

### Choosing Activation Functions
While \( \tanh \) is standard for state updates, diverse activation functions can meet specific requirements. Complex RNN variants may incorporate multiple activations within a layer.

### Key Points
- **Recurrent Cell**: Updates state based on current and previous inputs.
- **State Propagation**: \( H_t \) retains memory of past inputs.
- **Sequential Processing**: RNNs iterate through time steps, updating states and generating outputs.
- **Loss Calculation and BPTT**: Compute and aggregate losses, updating weights via backpropagation.

## Conclusion
RNNs excel in capturing temporal dependencies via state variables updated across time steps. This foundational concept underpins advanced sequence modeling techniques.

## Implementing RNNs from Scratch in TensorFlow
To implement RNNs from scratch in TensorFlow:

```python
import tensorflow as tf

class SimpleRNNCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SimpleRNNCell, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.W_h = self.add_weight(shape=(self.units, self.units), initializer='random_normal', trainable=True)
        self.W_x = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs, states):
        prev_state = states[0]
        state = tf.tanh(tf.matmul(inputs, self.W_x) + tf.matmul(prev_state, self.W_h) + self.b)
        return state, [state]

# Example usage
rnn_cell = SimpleRNNCell(units=128)
inputs = tf.random.normal([32, 10, 8])  # Batch size, Time steps, Input size
rnn_layer = tf.keras.layers.RNN(rnn_cell)
output = rnn_layer(inputs)
print(output.shape)

## Additional Examples

### One-Hot Encoding
### Example of encoding words as vectors using TensorFlow's one-hot encoding functionality.

vocab_size = 10000  # Example vocabulary size
word_index = 5  # Example word index

one_hot_vector = tf.one_hot(word_index, vocab_size)
print(one_hot_vector.shape)

## Learned Embeddings
### Example demonstrating the use of embeddings for word representations in TensorFlow.
embedding_dim = 16
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

# Example input: batch of sequences with word indices
sequences = tf.constant([[1, 5, 9], [2, 6, 0]])
embedded_sequences = embedding_layer(sequences)
print(embedded_sequences.shape)

# Complete RNN Model for Sequence Prediction
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.SimpleRNN(units=128, return_sequences=True),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

