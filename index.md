---
title: "Matrix Calculus Cheatsheet"
format:
  html:
    toc: false
    css: styles.css
    mainfont: sans-serif  # alt: system-ui 
    # minimal: true
  pdf: 
    documentclass: article
---

To train machine learning models, we use the standard SGD update rule (or some derived variant) defined as 

$$\theta^{\text {new }}=\theta^{\text {old }}-\alpha \nabla_\theta J(\theta)$$

However, of course, this requires us to compute $\nabla_\theta J(\theta)$, the gradient of the loss function with respect to our model weights/parameters. That is, for each parameter, we will need to calculate 

$$\theta_j^{\text {new }}=\theta_j^{\text {old }}-\alpha \frac{\partial J(\theta)}{\partial \theta_j^{\text {old }}}.$$

For the rest of the class, we will discuss how to calculate $\nabla_\theta J(\theta)$ in two different ways: 

1. By hand
2. Algorithmically: the backpropagation algorithm


### Chain rule 

The backbone of backprop is the chain rule which allows us to compute derivatives on compositions of functions. 

 
::: {#exm-default}

## Scalar-valued function 

$$z=3 y$$
$$y=x^2$$
$$\frac{d z}{d x}=\frac{d z}{d y} \frac{d y}{d x}=(3)(2 x)=6 x$$

:::

 
::: {#exm-default}

## Vector-valued function 

$$\boldsymbol{h}=f(\boldsymbol{z})$$
$$\boldsymbol{z}=\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b}$$
$$\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}}=\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{x}}=\ldots$$

:::
  

 
::: {#exm-default}

## Derivation of neural network gradient 

Say we have a one layer neural net where we take from input vector $\boldsymbol{x}$ and want to produce a singular output neuron called the score. Diagram: 

<p align='center'>
    <img alt="picture 1" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/16639f74e6e91122675e5be41335e78db68f8733c75a9953df3039fc3b8c282c.png" width="300" />  
</p>


key realization: let's break up the equations into simpler pieces use placeholder variables: 

$$\boldsymbol{h}=f(\boldsymbol{z})$$ where 
$$\boldsymbol{z}=\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b}$$

Now say we want $\frac{\partial s}{\partial \boldsymbol{b}}$ all we got to do is apply the chain rule on 

<p align='center'>
    <img alt="picture 2" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/bebf557b28605825afcc49133aba08ca5de7774f6312d2bdd11458aee9ac92fb.png" width="20%" />  
</p>

for which we get 

$$
\frac{\partial s}{\partial \boldsymbol{b}}=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{b}} = \boldsymbol{u}^T \operatorname{diag}\left(f^{\prime}(\boldsymbol{z})\right) \boldsymbol{I} = \boldsymbol{u}^T \odot f^{\prime}(\boldsymbol{z}).
$$

Now try calculating $\frac{\partial s}{\partial \boldsymbol{W}}$. Answer: 

$$\frac{\partial s}{\partial \boldsymbol{W}}=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}}$$

Aside: if we need to calculate both $\frac{\partial s}{\partial \boldsymbol{W}}$ and $\frac{\partial s}{\partial \boldsymbol{b}}$, then we are redudant: 

<p align='center'>
  <img alt="picture 3" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/408623306770ca64cd5cd902cd7e07640e5647ae5e592deb12b6dc992a7f40f0.png" width="20%" />  
</p>

to resolve this redundancy, we can define a variable $\boldsymbol{\delta}=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}}$. This memoization concept is key in backpropogation. 

:::
 



### Function shapes 

See [here](https://www.cs.princeton.edu/courses/archive/fall20/cos302/notes/cos302_f20_lecture12_vectordiff.pdf). 

 
:::{.callout-note appearance='simple'}


 
::: {.remark}

## **Derivative shape **


Let $$f: \mathbb{R}^{\text {(input shape) }} \rightarrow \mathbb{R}^{\text {(output shape) }}.$$ Then its derivative will be a function of shape $$\nabla f: \mathbb{R}^{(\text {input shape) }} \rightarrow \mathbb{R}^{(\text {output shape }) \times(\text { input shape) }}.$$

Each extra added dimension on the output corresponds to taking partial derivatives with respect to the all of the input dimension (think of the gradient, for example). Meaning, for each output, we need to take the derivative with respect to all inputs seperately. 
:::
:::

- Function of a scalar, returning a scalar: $\mathbb{R} \rightarrow \mathbb{R}$
  - Example: $f(x)=a x+b$
  - Derivative shape: $\frac{d f}{d x}: \mathbb{R} \rightarrow \mathbb{R}$ (derivative)
- Function of a scalar, returning a vector: $\mathbb{R} \rightarrow \mathbb{R}^n$
  - Example: $f(x)=x v$
  - Derivative shape: $\frac{d f}{d x}: \mathbb{R} \rightarrow \mathbb{R}^n$ (vector-valued derivative)
- Function of a vector, returning a scalar: $\mathbb{R}^n \rightarrow \mathbb{R}$
  - Example: $f(\boldsymbol{x})=\boldsymbol{v}^{\top} \boldsymbol{x}$
  - Derivative shape: $\nabla f(\boldsymbol{x}): \mathbb{R}^n \rightarrow \mathbb{R}^{1 \times n}$ (gradient)
- Function of a vector, returning a vector: $\mathbb{R}^n \rightarrow \mathbb{R}^n$
  - Example: $f(x)=M x$
  - Derivative shape:$\mathrm{J}(f(\boldsymbol{x})): \mathbb{R}^n \rightarrow \mathbb{R}^{n \times n}$ (jacobian)
- Function of a vector, returning a matrix: $\mathbb{R}^n \rightarrow \mathbb{R}^{n \times n}$
  - Example: $\boldsymbol{F}(\boldsymbol{x})=\boldsymbol{x} \boldsymbol{x}^{\top}$
  - Derivative shape: $\nabla \boldsymbol{F}(\boldsymbol{x}): \mathbb{R}^n \rightarrow \mathbb{R}^{n \times n \times n}$ (generalized Jacobian)


 
:::{.callout-note collapse='true'}
## Many more examples 

see cos 302 slides. 

:::
 

#### Neural network output shape 

You'll notice a flaw in our machinery. The above shapes are true in pure mathematics. But in neural network land, we want to do the following SGD update: 

$$\theta^{\text {new }}=\theta^{\text {old }}-\alpha \nabla_\theta J(\theta)$$

However, if our $\nabla_\theta J(\theta)$ shape is different from our old parameter shape $\theta^{\text {old }}$, we can't possibly do this update--the shapes don't match. So we define our new shape convention as **the shape of the gradient is the shape of the parameters $\theta$**. Example: $\frac{\partial s}{\partial \boldsymbol{W}}$ will be of shape $n \times m$: $$\left[\begin{array}{ccc}\frac{\partial s}{\partial W_{11}} & \cdots & \frac{\partial s}{\partial W_{1 m}} \\ \vdots & \ddots & \vdots \\ \frac{\partial s}{\partial W_{n 1}} & \cdots & \frac{\partial s}{\partial W_{n m}}\end{array}\right].$$ 

This leads us into reshaping and tranpose: two tools we'll need to get the SGD update to work shape-wise. 

 
::: {#exm-default}
Using our neural network example from above, we want to compute $\frac{\partial s}{\partial \boldsymbol{W}}$. We know that the shape should be the shape of $\boldsymbol{W}$ which is $n \times m$. So calculate our derivative as normal: 

$$\frac{\partial s}{\partial \boldsymbol{W}}=\boldsymbol{\delta} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}}$$

where $\boldsymbol{\delta}=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}}=\boldsymbol{u}^T \circ f^{\prime}(\boldsymbol{z})$ (which is of shape $1 \times n$). So our answer is: 


$$\frac{\partial s}{\partial \boldsymbol{W}}=\boldsymbol{\delta} 
\boldsymbol{x}
$$

But these dimensions don't give us our desired $n \times m$ shape. So let's use the tranpose to get an *outer product* (a hacky trick): 

$$\frac{\partial s}{\partial \boldsymbol{W}}=\boldsymbol{\delta}^T \boldsymbol{x}^T$$

Breaking down the shapes: 

<p align='center'>
  <img alt="picture 4" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/954b54e6b79129e8e88aa7782dfdd37e7d846106d52db7574654955859f5acda.png" width="55%" />  
</p>

This shape convention idea is all tied to neural networks because each derivative component in the outputted derivative is with respect to one edge between the current layer and the next (we need a component for each edge): 

<p align='center'>
  <img alt="picture 5" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/ca2563a333664bbbc61ccbd812e502a1e51288b196c5adecef5aaf1322f01ffe.png" width="30%" />  
</p>



:::
 


 



### Backpropogation 


How can we do everything we discussed above (matrix calculus), but in an **efficient** manner for many layer deep neural networks with millions of parameters? Backpropogation is an algorithm that accomplishes these two feats. 

The main ideas behind backprop include memoization of upstream gradients and computation graphs. 

 
::: {#def-default}

### Computation graph

A directed acyclic graph whose root node represents the final mathematical expression and each node represents intermediate subexpressions. "forward propogation". 

:::

 
::: {#exm-default}

### Forward propogation 

For: 

<p align='center'>
  <img alt="picture 6" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/7c5bbabd9d34ba61728fe93feee25749e577726a52e6eecb06fa2280e47315ca.png" width="18%" />  
</p>

we have: 

<p align='center'>
  <img alt="picture 8" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/f0107e3ea11aab3b0295990efc3f7c2a6c1aa3bf1d339dc1298f6a135523403c.png" width="400" />  
</p>

Two things:

- Nodes represent the operations (besides input/output nodes)
- Edges pass along result of the
operation

:::
 
 
::: {#exm-default}

### Back propogation 

Now calculate the gradients with respect to each node/edge needed of the final expression ($s$) with respect to the parameter you are updating (go backwards along the edges): 

<p align='center'>
  <img alt="picture 9" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/ee128e4f9a81683355d5c2af7f53354500d23e7e62f845c4881a650de1a43398.png" width="400" />  
</p>

Each node has a **local gradient**: The gradient of its output with respect to its input (i.e., the edge going into the operation/node and the edge going out of the operation/node). Each node receives (from the right hand side since we are going backwards) an "**upstream gradient**". The goal then is to calculate and pass along the correct **downstream gradient** where `[downstream gradient] = [upstream gradient] x [local gradient]`. 

<p align='center'>
  <img alt="picture 11" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/e07fb44a623864685bea0fcb7276c826d841f2e38a6c07826b3d64c7d5f6298a.png" width="400" />  
</p>


Then use the **chain rule** to piece together each downstream gradient until you arrive at the desired gradient: 

$$\frac{\partial s}{\partial \boldsymbol{b}}=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{b}}$$

*Remark*: notice that each downstream gradient is a derivative of the final expression $s$ (i.e., the global gradient). However, each local gradient which comprises the downstream global gradient is a derivative with respect to the local function.

:::


 
::: {#exm-default}

## Back propogation with multiple inputs 

Of course, neural network layers have many inputs for each hidden layer node. A simplified version of the neural network hidder layer evaluation expression is: 

$$z=\boldsymbol{W} x.$$

The concept in the previous example generalizes for multiple inputs; just calculate the downstream gradient for each respective input! Then follow along the path of the desired downstream gradient. 


<p align='center'>
  <img alt="picture 12" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/973fe8a56896a6b40672e7188baf7489075cd581d643ec6daca93f33469df2d1.png" width="400" />  
</p>

:::
 

Let's do another example of a simple function to solidy our understanding. 

 
::: {#exm-default}

### Forward pass computation graph construction 

Take the following function $f: \mathbb{R}^{3} \rightarrow \mathbb{R}$.

$$f(x, y, z)=(x+y) \max (y, z)$$

We want to construct a computation graph for the forward pass. 

1. Break down each "operation" into its own function and define that as a node. 

$$
\begin{aligned}
& a=x+y \\
& b=\max (y, z) \\
& f=a b
\end{aligned}
$$

2. Formulate the graph where the edges are the inputs and the nodes are the sub-operation functions. 

<p align='center'>
  <img alt="picture 13" src="https://cdn.jsdelivr.net/gh/minimatest/vscode-images@main/images/fcb95fa2716d5277612e376d8b5b31f1fde509e7385ef8f590086bc3b1ed7379.png" width="400" />  
</p>


:::







## Appendix A: Useful Identities {.appendix}

> Here we enumerate a bunch of helpful derivatives, simplifications, identities etc. in this section. 



::: {#def-default}

## Gradient 
 yo

:::

::: {#def-default}

## Jacobian 
 yo

:::
 

 
::: {#def-default}


## Various jacobian derivatives 


$$\frac{\partial}{\partial \boldsymbol{x}}(\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b})=\boldsymbol{W}$$

$$\frac{\partial}{\partial \boldsymbol{b}}(\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b})=\boldsymbol{I}$$

$$\frac{\partial}{\partial \boldsymbol{u}}\left(\boldsymbol{u}^T \boldsymbol{h}\right)=\boldsymbol{h}^T$$


$$\frac{\partial}{\partial \boldsymbol{z}}(f(\boldsymbol{z}))=\operatorname{diag}\left(f^{\prime}(\boldsymbol{z})\right)$$

:::

 
::: {#def-default}

### Sigmoid derivative

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

$$\frac{\partial}{\partial \theta_j} \sigma(z)=\sigma(z)[1-\sigma(z)]$$

:::


::: {#def-default}

### Log identities

Attaching a log before taking the derivative often helps us simplify the calculation since we can use the following identities: 

$$\log \left(\frac{x}{y}\right)=\log (x)-\log(y)$$

... 


:::


::: {#def-default}

### Log-exponential cancelation

You'll often come across the softmax normalization function: 

$$s\left(x_i\right)=\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$

which normalizes the distribution such that it sums to 1. It involves an `exp`. If we tack on a log before we take this gradient, these will cancel out. For example: 

$$
\frac{\partial}{\partial v_{c}} \log \exp \left(u_0^{\top} v_c\right)=\frac{\partial}{\partial v_c} u_0^{\top} v_c
$$ 




:::




 


## Appendix B: More matrix calculus examples {.appendix}


Taking gradients of loss functions with respect to matrix weight parameters is a key concept used in many machine learning classes--so you'll want to nail this concept down. To this end, we will provide many examples of taking gradients of loss functions with the hope of picking up useful patterns and identities (i.e., gradients of log). 

*To be updated eventually*. 

Example sources: 

- CS 229 
- CS 224N/CS231N 
- CS 221 exams 


## Appendix C: Useful resources {.appendix}

- COS 302 slides 
- Matrix cookbook 
- https://explained.ai/matrix-calculus/ 
