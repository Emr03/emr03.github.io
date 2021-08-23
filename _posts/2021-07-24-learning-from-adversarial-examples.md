---
title: Learning From Adversarial Examples (Part I)
author: Elsa Riachi
categories: [Research]
tags: [adversarial robustness]
date: 2021-07-24
math: true
---

<div style="display:none">
We examine how training a multi-class perceptron on adversarial attacks can lead to generalization.
This exercise illustrates how adversarial attacks are well-generalizing without
requiring the presence of non-robust features *in the dataset*.
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
\newcommand\argmax{\mathop\mathrm{arg max}} % Defining math symbols
\newcommand\argmin{\mathop\mathrm{arg min}}
$$

</div>

We examine how training a multi-class perceptron on adversarial attacks can lead to generalization.
This exercise illustrates how adversarial attacks can be well-generalizing without
requiring the presence of non-robust features in the dataset.

I am aware that considering a simple hypothesis class makes the following discussion fall short of being a satisfying argument. While a multi-class perceptron is much simpler and less expressive than a neural network, observing similar phenomena in simpler settings allows us to develop a stronger intuition and rethink how we interpret similar behaviour in neural networks.

Consider a multi-class perceptron with weight matrix $$W \in \mathbb{R}^{K \times d}$$, $$\hat{y} = \mathop{\mathrm{arg max}}_j W_j^T x$$ with $$K$$ classes. Let $$W$$ and the data distribution of $\{x_i, y_i\}$ be such that for all $$(x_i, y_i) \sim \mathcal{D}$$,

$$
\begin{equation}
W_{y_i}^T x \geq W_{j}^T x + \gamma \ \ \forall j \neq y_i
\end{equation}
$$

for some positive $$\gamma$$. In other words, we assume the data is linearly separable and $$W$$ achieves a minimum margin $$\gamma$$.


We could formulate a targeted adversarial attack as a norm-bounded $$\delta$$ minimizing the hinge loss or cross-entropy loss with the target label $$t$$, however to simplify the downstream analysis, we consider $$\delta$$ to be the minimal perturbation required to achieve the target prediction $$t$$ with given margins $$\{c_j\}_{j=1,\ j \neq t}^{K}$$, $$c_j > 0$$.

$$
\begin{align}
\delta &= \mathop{\mathrm{arg min}}_{\delta} \frac{1}{2}\twonorm{\delta}^2 \\
\text{s.t } &  W_t^T(x + \delta) \geq W_j^T(x + \delta) + c_j \ \forall j \neq t
\end{align}
$$

Since the constrained optimization problem is convex, we find the optimal $$\delta$$ using the corresponding Lagrangian.
To simplify, we use $$A_t$$ to denote the matrix whose columns consist of $$\{W_t - W_j\}_{j\neq t}$$, and $$\lambda$$ to denote the
vector of Lagrangian multipliers $$\{\lambda_j\}_{j\neq t}$$.


$$
\begin{align}
\mathcal{L}(\delta, \lambda) = \frac{1}{2}\twonorm{\delta}^2 - \sum_{j} [ W_t^T (x + \delta) - W_j^T (x + \delta) - c]
\end{align}
$$


$$
\begin{align}
\delta & = \sum_j \lambda_j (W_t - W_j) \\
\delta & = A_t \lambda
\end{align}
$$

Using the dual formulation we find that

$$
\begin{equation}
\delta = A_t (A_t^TA_t)^{-1}[m - A_t^T x]
\end{equation}
$$

Where $$A_t (A_t^TA_t)^{-1}A_t^T$$ is the orthogonal projection $$P_{A_t}$$ onto the column space of $$A_t$$, and $$A_t (A_t^TA_t)^{-1}$$ is the
pseudo-inverse $$A_t^{\dagger}$$ of $$A_t$$.

Then the resulting adversarial input is,

$$
\begin{equation}
x + \delta = A^{\dagger}_t m  + [I - P_{A_t}^T] x
\end{equation}
$$

We now consider what happens when a newly initialized weight matrix $$\Omega$$ is trained on a dataset of adversarial input-target pairs, using the multi-class perceptron learning algorithm.

If an input-label pair $$(x + \delta, t)$$ is misclassified as class $$q$$, we update the $$t^{th}$$ row of $$\Omega$$ as follows:

$$
\begin{equation}
\Omega_t \leftarrow \Omega_t + x + \delta
\end{equation}
$$

and update the $$q^{th}$$ row of $$\Omega$$ as follows:

$$
\begin{equation}
\Omega_q \leftarrow \Omega_q - x - \delta
\end{equation}
$$

Then the $$n^{th}$$ iterate

$$
\begin{align}
(\Omega_t - \Omega_q)^{(n)T} (W_t - W_q) &= [(\Omega_t - \Omega_q)^{(n-1)} + 2(x + \delta)]^T (W_t - W_q) \\
& = (\Omega_t - \Omega_q)^{(n-1)T} (W_t - W_q) + 2c_q \\
& \geq 2nc_q
\end{align}
$$

Upper bound:

$$
\begin{align}
\twonorm{(\Omega_t - \Omega_q)^{(n)}}^2 &= \twonorm{(\Omega_t - \Omega_q)^{(n-1)} + 2(x + \delta)}^2 \\
& \leq \twonorm{(\Omega_t - \Omega_q)^{(n - 1)}}^2 + \twonorm{2(x + \delta)}^2 \\
& \leq 4n (R + \epsilon)^2
\end{align}
$$

We use the upper and lower bounds to lower bound the cosine similarity between $$\Omega_t - \Omega_q$$ and $$W_t - W_q$$.

$$
\begin{align}
 cos(\Omega_t - \Omega_q, W_t - W_q) \geq \frac{2nc_q}{2\sqrt{n} (R + \epsilon)(\twonorm{W_t - W_q})}
\end{align}
$$


How does $$\Omega$$ do on standard samples of $$\mathcal{D}$$?
(W_t - W_q)^Tx \geq \gamma

angle between (\Omega_t - \Omega_q) and x <= angle between (\Omega_t - \Omega_q) and (W_t - W_q) +
angle between (W_t - W_q) and x.
\theta = \theta1 + theta2
cos(theta) = cos(theta1)cos(theta2) - sin(theta1) 1 - \frac{x^T(W_t - W_q)}{|x|}

$$
\begin{align}
(\Omega_t - \Omega_q)^T x \geq (\Omega_t - \Omega_q)^T (W_t - W_q)
\end{align}
$$
