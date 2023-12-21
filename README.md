# mnist

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.

The hidden layer's purpose is to perform computations on the input data and extract features that are useful for the given task. Each neuron (or node) in the hidden layer receives inputs from the neurons in the previous layer (the input layer) and applies a transformation using weights and biases. These weights and biases are adjusted during the training process to minimize the difference between the predicted output and the actual output.

The hidden layer essentially learns the complex patterns and relationships within the input data. As the network is trained on labeled examples (e.g., images of digits with their respective labels), the hidden layer adjusts its weights and biases to progressively recognize features that help in distinguishing between different classes of digits. These features are not explicitly programmed but are learned automatically through the training process, allowing the network to make accurate predictions on unseen data.

The number of hidden layers and the number of neurons in each hidden layer are hyperparameters of the neural network architecture that can significantly impact the network's ability to learn and generalize patterns in the data. 


Our NN will have a simple two-layer architecture. 

Input layer  𝑎[0] :
    784 units corresponding to the 784 pixels in each 28x28 input image. 

Hidden layer  𝑎[1] :
    10 units with ReLU activation, 
    ReLU (Rectified Linear Unit) is an activation function commonly used in neural networks. It is defined mathematically as:

    f(x) = max(0,x)

    In simple terms, the ReLU activation function outputs the input value if it's positive and zero if it's negative. This function introduces non-linearity into the network, allowing it to learn complex relationships within the data. ReLU has become popular due to its simplicity and effectiveness in many deep learning models.
    
Output layer  𝑎[2] :
    10 units corresponding to the ten digit classes with softmax activation.
    Softmax is commonly used for multi-class classification problems as it provides a probability distribution over the classes.

    The softmax function takes a set of values as input and normalizes them into a probability distribution. It computes the probabilities of each class being the correct class.
    
    In the context of digit classification:
    The output layer has 10 units, each representing a digit class from 0 to 9.
    The softmax activation function will take the inputs (logits) from the preceding layer and produce a probability distribution over these 10 classes, indicating the likelihood or confidence of the input belonging to each digit class.
    The final prediction will typically be the class with the highest probability output by the softmax function.
    This setup allows the neural network to generate probabilities for each class, enabling it to make predictions by selecting the class with the highest probability as the predicted digit for a given input.