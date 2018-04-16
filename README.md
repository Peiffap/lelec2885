# lelec2885

In this project the team has studied theoretically and practically a type of **Neural Networks** especially used for images applications: **Convolutional Neural Networks**.

Convolution Neural Networks are similar to ordinary Neural networks, but they take advantages that the input layer consists on images and its can be the full image with the larger size compared with regular networks. In fact, the image of the regular network can only of size 32*32*3 which corresponds to width, height and color channel. If we take the image with larger size, overfitting would happened, because it would contain more parameters than can be justified by the data, leading to poor performance.

Actually, there are three main layers in CNN: Convolution layer, Pooling layer (Max pooling) and Fully connected Layer. The convolution layer is the main layer in CNN, including the parameter ’filter’ which is actually the smaller number, but with the same number of depth with input image. Inside the convolution layer, the filter scan over or convolve every pixel and computing the dot product. Pooling layer would be our second layer and it is used to conduct downsampling operations inside which reducing the size of the image, that can be more easy to control the ’overfitting’ issue. The most common one is max pooling, it use the 2*2 filter with stride=2 to keep the maximum value of the image. The final layer goes to Fully connected layer that computing the class scores, and reduce the full image to the single vector of class scores.
To implement practically our CNN, we are going to resort to Keras library. Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano[1].

## Model Implementation

In this section we are going to explain our choices to implement the CNN. You can find the python code in [CNN.py](code/CNN.py).

First of all we have implemented 2D convolutional layer which is set up to take 32 windows, or ’filters’, of each image, each filter being 3x3 in size. We then run a second convolution on top of that with 64 3x3 windows. This topology is just what comes recommended within Keras’s own examples [1].

Next we apply a MaxPooling2D layer that takes the maximum of each 2x2 result to distill the results down into something more manageable. A dropout filter is then applied to prevent or to avoid overfitting. It consists of setting to zero the output of each hidden neuron with the probability given. The neurons which are ’dropped out’ in this way do not contribute to the forward pass and do not participate in backpropagation [5].

Next we flatten the 2D layer we have at this stage into a 1D layer. Then we feed that into a hidden, flat layer of 128 units. We then apply dropout again to further prevent overfitting. And finally, we feed that into our final 10 units where softmax is applied to choose our category of 0-9.

Then, the next part we are going to induce is the two main argument in the Keras Model, optimizer and loss function. Firstly, regarding the optimizer, it is about to optimize the model weight and reduce the cost function and are based based on Gradient descent (GD). There are several type of optimizer that we can choose, such as SGD which is Stochastic gradient descent optimizer, in order words random GD, and RMSprop, Adagrad, Adam ,etc We chose Adam because it is straightforward, efficient and only consume little memory.

The second argument is Loss Function, it is used to estimate the difference between predicted value and true value, named y pred and y true. There are also numerous types of loss function that we can choose. For this project, we chose categorical crossentropy because it expects a binary class, that what our image perform.