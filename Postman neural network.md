Cost Function

## Neural Network with Manual Backpropagation


This is my attempt at creating a neural network that can identify numbers successfully
using the MNIST dataset (reduced).


- Samson Zhang - How to build a neural network using just math


- Neural Networks and Deep Learning by michael nielsen - read through a couple of
chapters of the book


- 3blue1brown : insightful videos, but didn't really understand the math behind them in
depth. why are these functions necessary to distribute the numbers in depth.


The main issue with my understanding of the neural network was the way that it learnt.
Even after watching the 3Blue1Brown videos and the Michael Nielsen book (the first three
chapters) I had no idea what these mathematical functions meant, in the sense that why
they were necessary. I had to do some exploration on reddit and stackoverflow for it to

make a bit of sense to me.


I’m going to be fully honest. Some of the logic behind the code that I’ve written isn’t
completely my own. I wasn’t very proficient with NumPy before starting this task (basically,
I knew nothing about it), and I definitely had no idea how neural networks worked.


It first began to click when I watched the Harvard CS50 lecture on how to use the MNIST
database (although that lecture had used TensorFlow and some very complicated stuff.
But the teacher broke it down into very simple steps.


The way we filter a normal image into black and white is by breaking apart the image into
a million different pixels, and giving the computer a set of training data. Basically, if this
stands out more toward one side, represent that as a degree of blackness, and represent
the rest as a degree of whiteness. Oh, and this degree is represented in binary.

Backpropagation essentially means to feed the neural network what it has done again,
whilst subtly adding weights to the functions to “teach” (heavy double quotes) what it is
doing wrong and what it needs to be doing.


The above three pictures essentially represents all the math I’ve used for my project. The
matrix operations were handled by numpy, and the neural network training was handled
by mainly two python functions, ReLU and SoftMax.

# Step 1 : The Input Layer


Take each of the images and break it down into an array of 28*28 = 784 pixels or units.
Input that into the neural network, shape and shuffle this data.


A(0) = X ( has 784 x m )


Then we shuffle and shape into a random m x n array.


Z(1) = w(1)A(0) + b(1)


w refers to the weight, b refers to the bias.


A(1) = g(Z(1) ) (ReLU (Z(1))


A(Z) = softmax (Z(1)


This function is to ensure that the stepwise function that we are doing ( 1 if the pixel is
dark and 0 if the pixel is light is softened down into a value BETWEEN zero and one so that
we can define the degree of darkness and lightness. (Since we are dealing with purely

black and white images from the MNIST dataset, we don’t need to worry about colour or
hex codes and stuff.)


Now this will give us values that may be greater than one, so we apply sigmoid function to
squeeze it down between that range.

# Step 2: The Next Layer


The second layer is just a linear combination of the first layer. To avoid linear regression,
we apply functions. These are really cool mathematical constructs that ( to me atleast)
attempts to mimic on a basic level how our neurons fire in the brain. Of course, our brain
has multiple weird weights and biases and has been training its neurons since the day we
were born but this is how I understood it. I thought Michael Nielsen explained this very
well with the perceptron girlfriend analogy.


The second layer is a sigmoid function, followed by a ReLU.


Gradient Descent

We use the cost function to figure out how far from the actual necessary pinpoint accurate
model we are standing. And then we nudge the neural network toward better and better
levels of accuracy by using gradient descent. What is gradient descent? We are basically
jumping around with averages, hoping that each jump will get us closer to the bottom of
the graph ( in 3 dimensions). This graph is a plotting of three of four characteristics. A, Z, B
and W. A minima of any three of these is good enough for us to ensure that we are
reaching a level of accuracy, so we choose any three parameters and get started. This is
where the statistical learning comes in. We use calculus to compute derivatives of this cost

function.


Dz = A(Z) – Y


DZ is (10x m )


A(Z) ( 10 x m)


Y is 10 x m


Y is necessary to offset the values so that they are closer to zero, and we don’t get an out
of bounds accuracy.


Db(Z) = 1/m (summation of the Z values )


DZ is = W(z)(t) * Dz(z) ( all three are in 10 x m format)


DW ( 10* 784) is = 1/m Dz(1) ( 10 x m) * X(transpose) ( m * 784)


This is a basic overview to what we are trying to achieve.

# Learning Rate

A good learning rate should be tweaked according to both the amount of values that you
are giving to the algorithm to learn from. And also the dataset ( variance of the data . A
large learning rate ( alpha) will mean to me that you are overfitting the data, and a small
learning rate might be meaning that you are taking too small jumps toward the lowest
gradient descent.


I’m not exactly sure how to optimize the algorithm toward tweaking itself to choosing the
best amount of iterations and the alpha value so as to get the best accuracy. So far its just
been me tweaking the variables manually, to find what can improve.