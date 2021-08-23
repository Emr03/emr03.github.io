---
title: The Structure of Adversarial Perturbations (Part I)
author: Elsa Riachi
categories: [Research]
tags: [adversarial robustness]
date: 2021-02-13
math: true
---

<div style="display:none">
Adversarial vulnerability is a fundamental limitation of deep neural networks which remains poorly understood. Recent work suggests that adversarial attacks exploit the fact that non-robust models rely on superficial statistics to form predictions.
</div>

<div style="display:none">
$$
\newcommand\testmacro[2]{\mathbf{F\alpha}(#1)^{#2}}
\def\norm#1{\left\|{#1}\right\|} % A norm with 1 argument
\newcommand\zeronorm[1]{\norm{#1}_0} % L0 norm
\newcommand\onenorm[1]{\norm{#1}_1} % L1 norm
\newcommand\twonorm[1]{\norm{#1}_2} % L2 norm
\def\<{\left\langle} % Angle brackets
\def\>{\right\rangle}
\newcommand\inner[1]{\langle #1 \rangle} % inner product
$$

</div>


## Introduction and Background
Adversarial vulnerability is a fundamental limitation of deep neural networks which remains poorly understood. Recent work suggests that adversarial attacks exploit the fact that non-robust models rely on superficial statistics to form predictions. Ilyas et al. discuss this hypothesis in [this article]() and [this paper](). In a nutshell, the authors propose that there may be patterns of pixels that are highly predictive of the image's class label but appear non-sensical to humans. These patterns are named *non-robust features*.

This hypothesis is reminiscent of results that show that neural networks are highly sensitive to changes in texture, or high-frequency components. Most importantly, the paper's hypothesis states that these patterns or *features* which appear non-sensical, allow the network to generalize to unseen examples. To support their claim, the authors demonstrate a surprising experimental result. Their method is briefly outlined below:

1. The authors train a neural network (let's call it network A) on the CIFAR10 training set.  
2. For each image-label pair (x, y) in the training set, a target label $$t$$ is chosen at random, and an adversarial perturbation
$$\delta$$ is computed such that network A classifies $$x + \delta$$ as an instance of class t.
3. The above procedure results in a newly created training set of adversarial input-target pairs $$(x + \delta, t)$$ denoted as $$D_{adv}$$.
4. A new network (let's call it network B) is trained on the adversarial training set $$D_{adv}$$.
5. Network B is then evaluated on the original CIFAR10 test set, and surprisingly it does well!

The fact that network B, trained on $$D_{adv}$$ generalizes to the standard test set appears to contradict the basic intuition that generalization is achieved by training on many representative samples from the data distribution. The authors use this observation to conclude that adversarial perturbations correspond to well-generalizing features that are predictive of the target label $$t$$. **However, these results don't tell us anything about the structure of these perturbations, nor do they explain why these perturbations lead to generalization to the standard test set. I specifically explore both of these questions.**


## What About Attacks on Autoencoders?
A natural approach to gain insight into the phenomenon described above is to replicate it using autoencoders. The intuition behind this approach is simple. Since an autoencoder is required to effectively compress an input image such that it may be reconstructed with low reconstruction error, an encoder must capture most of the discernible features within an image. Yet, autoencoders have also been shown to be vulnerable to adversarial attack. If an input image may be reconstructed from the encoder's representation, what makes the encoder vulnerable to adversarial attacks? How should we re-interpret the results of Ilyas et al. \cite{} to explain attacks on autoencoders?

Targeted attacks on autoencoders can be formulated as the constrained optimization problem shown below:

$$ \begin{align}
\boldsymbol{\delta}^* &= \mathop{\mathrm{arg} min} \twonorm{E(\textbf{x}_t) - E(\textbf{x}_s + \boldsymbol{\delta})}^2 \\
\label{eq:encoder_attack} \\
& \twonorm{\boldsymbol{\delta}}^2 \leq \epsilon
\end{align}$$

Where a norm-bounded perturbation $$\boldsymbol{\delta}$$ is added to a source image $$\textbf{x}_s$$ so as to produce a similar representation to that of a randomly selected target image $$\textbf{x}_t$$.  Denoting the encoder by $$E(.)$$ and the decoder by $$D(.)$$, the success of the attack is determined by the squared error between the target $$\textbf{x}_t$$ and the reconstruction $$D \circ E(\textbf{x}_s + \delta)$$. Examples of targeted adversarial attacks on images from the CelebA dataset are shown below.

<img src="/posts/the-structure-of-adversarial-perturbations-part1/images/adversarial_pair.svg" width="750">

With the attack objective shown above, we generate a training set of adversarial input-target pairs $$(\textbf{x}_s + \boldsymbol{\delta}, \textbf{x}_t)$$, much like the experimental procedure described by Ilyas et al. \cite{}.

<figure>
    <img src="/posts/the-structure-of-adversarial-perturbations-part1/images/dataset_construction.svg" width="750">
    <figcaption>A dataset of adversarial source and target pairs is constructed by crafting targeted attacks on the encoder and pairing perturbed images with the corresponding target image.</figcaption>
</figure>

Interestingly, we observe that a newly initialized autoencoder trained on the adversarial training set learns to reconstruct unperturbed images from the standard test set. In our case, the added perturbation $$\boldsymbol{\delta}$$ isn't merely informative of a class label, but of the target image!

<figure>
<img src="/posts/the-structure-of-adversarial-perturbations-part1/images/reconstructions.svg" width="750">
<figcaption>A new autoencoder trained on adversarial source and target pairs learns to reconstruct images from the standard test set.</figcaption>
</figure>

In the next section we examine the worst-case noise of a linear encoder to understand the structure of adversarial perturbations and obtain some insight into this strange observation.

## Adversarial Attacks on a Linear Encoder

In this section we study attacks on a linear encoder, represented as a matrix $$\Phi \in \mathbb{R}^{M \times N}$$, where $$M << N$$. While principal component analysis first comes to mind when constructing a linear encoder, we defer this discussion for later. For now, we focus on the worst-case perturbation $$\boldsymbol{\delta}$$ for a particular source-target pair $$(\textbf{x}_s, \textbf{x}_t)$$.

$$
\begin{align}
\min_{\delta} \twonorm{\Phi (\mathbf{x_s} + \boldsymbol{\delta}) - \Phi \mathbf{x_t}}^2 \\
\twonorm{\boldsymbol{\delta}}^2 \leq \epsilon^2
\label{eq:attack}
\end{align}
$$

Since the above constrained optimization problem is convex, the solution is the critical point of the Lagrangian $$\mathcal{L}(\boldsymbol{\delta}, \lambda)$$.

$$
\begin{align}
\mathcal{L}(\boldsymbol{\delta}, \lambda) &= (\textbf{x}_s - \textbf{x}_t + \boldsymbol{\delta})^T \Phi^T \Phi (\textbf{x}_s - \textbf{x}_t + \boldsymbol{\delta}) + \lambda(\boldsymbol{\delta}^T \boldsymbol{\delta} - \epsilon) \label{eq:lagrangian} \\
\nabla_{\delta}\mathcal{L}(\boldsymbol{\delta}, \lambda) &= 2\Phi^T \Phi (\textbf{x}_s - \textbf{x}_t + \boldsymbol{\delta}) + 2\lambda \boldsymbol{\delta} = 0 \nonumber \\
\boldsymbol{\delta} &= \left (\Phi^T \Phi + \lambda I \right)^{-1} \Phi^T \Phi \left (\textbf{x}_t - \textbf{x}_s \right) \nonumber \\
\end{align}
$$

The solution $$\boldsymbol{\delta}$$ can be decomposed into two components $$\boldsymbol{\delta_s}$$ and $$\boldsymbol{\delta_t}$$, where $$\boldsymbol{\delta_s}$$ is such that $$\twonorm{\Phi \left( \mathbf{x_s} - \boldsymbol{\delta_s}\right )}^2$$ is minimized, while $$\boldsymbol{\delta_t}$$ is such that $$\twonorm{\Phi \left( \boldsymbol{\delta_t} - \mathbf{x_t} \right)}^2$$ is minimized. That is, $$\boldsymbol{\delta_s}$$ is crafted so as to obfuscate $$\textbf{x}_s$$ while $$\boldsymbol{\delta_t}$$ is crafted so as to pass as $$\textbf{x}_t$$.

$$
\begin{align}
\boldsymbol{\delta}_s &= \left (\Phi^T \Phi +  \lambda I \right)^{-1} \Phi^T \Phi \textbf{x}_s \label{eq:delta_s} \\
\boldsymbol{\delta}_t &= \left (\Phi^T \Phi +\lambda I \right)^{-1} \Phi^T \Phi \textbf{x}_t \label{eq:delta_t}
\end{align}
$$

We denote the transformation $$\left (\Phi^T \Phi + \lambda I \right)^{-1} \Phi^T \Phi $$ by the matrix $$\textbf{M}_{\Phi}$$.  The final expression for $$\boldsymbol{\delta}$$ which we use to attack $$\Phi$$ is shown below.


$$
\begin{equation}
\boldsymbol{\delta} = \textbf{M}_\Phi \textbf{x}_t - \textbf{M}_\Phi \textbf{x}_s
\label{eq:final_delta}
\end{equation}
$$


We now consider the case where the linear encoder $$\Phi$$ is constructed from the top $$M$$ principal components of the input distribution. That is, the rows of $$\Phi$$ are orthonormal vectors in $$\mathbb{R}^{N}$$. An input vector $$\textbf{x}$$ consists of a linear combination of the top $$M$$ principal components $$\{\boldsymbol{\phi_1}, ...\boldsymbol{\phi_M}\}$$, which we denote by $$\hat{\textbf{x}}$$, and a residual component that is orthogonal to the span of $$\{\boldsymbol{\phi_1}, ...\boldsymbol{\phi_M}\}$$. We denote the residual component as $$\textbf{r}$$. Note that $$\twonorm{\textbf{r}} = \twonorm{\hat{\textbf{x}} - \textbf{x}}$$. Since the principal components are such that $$\twonorm{\hat{\textbf{x}} - \textbf{x}}$$ is minimized, we may consider the reconstruction error $$\twonorm{\textbf{r}}$$ to be small.


$$
\begin{equation}
\textbf{x} = \sum_{i=1}^{M} \alpha_i \boldsymbol{\phi_i} + \textbf{r}
\end{equation}
$$


The output of $$\Phi \textbf{x}$$ is $$(\alpha_1, \alpha_2, ..., \alpha_M)^T$$ whose norm must be much larger than $$\twonorm{\textbf{n}}$$ if the number of principal components is chosen well. We can immediately see that for a perturbation $$\boldsymbol{\delta}$$ to be successful, $$\boldsymbol{\delta}$$ must be aligned with one or more principal components. In fact, using the form for $$\boldsymbol{\delta}$$ derived above, (after a few algebraic manipulations) we see that the optimal $$\ell2$$-bounded perturbation which passes as $$\textbf{x}$$ is given by $$\boldsymbol{\delta_t} = \frac{1}{\lambda}\Phi^T \Phi \textbf{x}$$ which is just a rescaling of $$\textbf{x}$$. Since the row vectors of $$\Phi$$ are orthonormal and $$\twonorm{\boldsymbol{\delta}}$$ is much smaller than $$\twonorm{\textbf{x}}$$ it follows (by Cauchy-Schwarz) that $${\boldsymbol{\delta}}$$ cannot be a successful attack. A linear encoder obtained from PCA is therefore not vulnerable to adversarial attack. However, natural images aren't well-represented by linear PCA. Sparse coding has been successfully applied to denoising, image compression and super-resolution of natural images. Recent work has also highlighted the connection between sparse coding and CNN's \cite{}. We therefore consider the input to be a sparse combination of dictionary atoms rather than a dense combination of principal components. More concretely, we consider $$\textbf{x}$$ to admit a sparse representation $$\mathbf{v}$$ with respect to a dictionary $$D$$.

$$
\begin{equation}
\mathbf{x} = D\mathbf{v}
\end{equation}
$$


Compressive sensing is an effective method for sparse signal compression \cite{Cands2005StableSR}. Its aim is to answer the following question: given a sparse vector $$\mathbf{v}$$, how must one project $$\mathbf{v}$$ onto a lower dimensional vector $$\mathbf{y}$$ such that $$\mathbf{v}$$ can be recovered from $$\mathbf{y}$$? While the projection done by $$\Phi$$ is in general not invertible, the vector $$\textbf{v}$$ can be recovered from $$\mathbf{y}$$ if it is sufficiently sparse.

The key to the recovery of $$\textbf{v}$$ is to ensure that every unique support of $$K$$ entries of $$\mathbf{v}$$ is mapped by $$\Phi$$ to its own unique subspace of $$M$$-dimensional representations. In other words, the minimal set of column vectors of $$\Phi$$ that are linearly dependent consists of more than $$2K$$ columns.

In the compressive sensing literature, the recovery of $$\mathbf{v}$$ from $$\mathbf{y}$$ is expressed as the $$P_0$$ problem.

$$
\begin{equation}
P_0: \min_v \zeronorm{\mathbf{v}} \text{ s.t. } \Phi \mathbf{v} = \mathbf{y}
\label{eq:p0}
\end{equation}
$$

$P_0$ is guaranteed to have a unique solution $$\mathbf{v}$$ if the sparsity of $$\mathbf{v}$$ is bounded, as shown below.

$$
\begin{equation}
s = \zeronorm{v} \leq \frac{1}{2} \left(1 + \frac{1}{\mu(\Phi)}\right),   
\label{eq:sparsity}
\end{equation}
$$

where $$\mu(\Phi)$$ is the \textit{mutual coherence of $\Phi$} defined as

$$
\begin{equation}
\mu(\Phi) := \max_{i \neq j} \frac{\inner{\Phi_i, \Phi_j}}{\twonorm{\Phi_i}\twonorm{\Phi_j}}.  
\end{equation}
$$

In the worst case, a signal $$\mathbf{v}$$ which does not satisfy the sparsity condition in \ref{eq:sparsity} may have a counterpart signal with similar sparsity and measurement vector. In this case, the $$P_0$$ problem does not admit a unique solution. In our study of adversarial attacks, we consider signals $$\mathbf{v}$$ which satisfy the sparsity condition and may be recovered from their corresponding representation $$\mathbf{y}$$. However, we may still obtain a counterpart signal for $$\mathbf{v}$$ under an $$l2$$-norm constraint given by $$\boldsymbol{\delta} = \left (\Phi^T \Phi +\lambda I \right)^{-1} \Phi^T \Phi \textbf{v}$$.  These norm-constrained counterparts form the building blocks of adversarial attacks. Note that these norm-constrained counterparts admit the same sparse representation given by $$\boldsymbol{v}$$ yet are constructed using the dictionary $$\left (\Phi^T \Phi +\lambda I \right)^{-1} \Phi^T \Phi$$.   

So far, we have shown 
We begin by studying adversarial attacks using a synthetic dataset of structured sparse signals. Our constructed dataset consists of $$28 \times 28$$ images made up of at most 5 discrete Fourier transform (DFT) components. We denote the sparse representation in the DFT domain of a $$28 \times 28$$-dimensional image as $$\mathbf{x}$$. For simplicity, we assume the encoder acts on the 225-dimensional sparse representation of the input. The reason for this assumption is to illustrate the relationship between adversarial perturbations and the sparse representation of the input. To construct $$\Phi$$ we sample i.i.d entries from $$\mathcal{N}(0, \frac{1}{M})$$, this ensures that $$\mu(\Phi)$$ is sufficiently low. We can therefore obtain a dense measurement vector $$\textbf{y} = \Phi \textbf{x}$$ from which the signal $$\mathbf{x}$$ can be recovered. We use a deconvolution network to reconstruct the $28 \times 28$ image from the measurement vector $$\textbf{y}$$. The figure below shows examples of adversarial attacks computed using $$\boldsymbol{\delta} = \textbf{M}_\Phi \textbf{x}_t - \textbf{M}_\Phi \textbf{x}_s$$. We emphasize that the random weights of encoder $\Phi$ were kept fixed while the decoder was updated. Therefore, the adversarial vulnerability of the encoder isn't due to particular *non-robust* features within the input images, but rather the redundancy of the encoding matrix $$\Phi$$.


<div style="text-align: center">
<figure>
<img src="/posts/the-structure-of-adversarial-perturbations-part1/images/Figure_1.png" width="500">
<img src="/posts/the-structure-of-adversarial-perturbations-part1/images/Figure_5.png" width="500">
<figcaption> Examples of adversarial attacks on the synthetic dataset. </figcaption>
</figure>
</div>

A newly initialized autoencoder trained on a dataset of adversarial image-target pairs obtained using the above synthetic dataset also learns to reconstruct unperturbed images. More precisely, the new encoder $$\Psi$$ learns to represent an adversarial input $$(I - \textbf{M}_\Phi) \textbf{x}_s + \textbf{M}_\Phi \textbf{x}_t$$ similarly to the target $$\textbf{x}_t$$, for any source-target pair.

$$
\begin{equation}
\Psi \left [ (I - \textbf{M}_\Phi ) \textbf{x}_s + \textbf{M}_\Phi \textbf{x}_t) \right] \sim \Psi \textbf{x}_t
\end{equation}
$$

Note that the above condition is equivalent to the following shown below, for all sparse representations $$\textbf{x}$$ covered by the synthetic data model:

$$
\begin{equation}
\Psi \textbf{M}_\Phi \textbf{x} \sim \Psi \textbf{x}
\end{equation}
$$

Note that the only component of $$\boldsymbol{\delta}$$ that is informative of the target image is $$\textbf{M}_{\Phi} \textbf{x}_t$$

References
----------

{% bibliography --file adversarial_examples %}
