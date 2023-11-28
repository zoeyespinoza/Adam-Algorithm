# Adam-Algorithm 
Example Task

We will soon start training the LeNet network on GPU graphics cards, but first, let's consider one more optimization algorithm that will help improve the results.

Stochastic gradient descent (SGD) isn't the most effective algorithm for training a neural network. If the stride is too small, the training may take too long. If it's too large, it might not manage the minimum required training. The Adam algorithm makes stride selection automatic. It selects different parameters for different neurons, which speeds up model training.
To understand how this algorithm works, consider this visualization created by Emilien Dupont from Oxford University. It displays four algorithms: SGD **on the left, the Adam algorithm on the right, and between them are two algorithms similar to Adam (we won't be discussing them in-depth). Out of the four of them, Adam is the fastest way to find the minimum.

Let's write the Adam algorithm into Keras:

model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc']
) 
Set the algorithm class to configure the hyperparameters:
from tensorflow.keras.optimizers import Adam

optimizer = Adam()

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['acc'],
) 
The main configurable hyperparameter in the Adam algorithm is the learning rate. This is the part of the gradient descent where the algorithm starts. It's written as follows:

optimizer = Adam(lr=0.01) 
T
he default learning rate is 0.001. Reducing it can sometimes slow down learning, but that improves the overall quality of the model.

### Task 
Create and train a convolutional neural network for the clothing dataset. To do this, create three functions in the code:
load_train() for loading the training sample
create_model() for model creation
train_model() for launching the model
