---
title: Constrained Optimization, The KKT conditions
author: Elsa Riachi
categories: [Notes]
tags: [optimization]
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
\newcommand{\pd}[2]{\frac{\partial{#1}}{\partial{#2}}}
\newcommand{\pdd}[2]{\frac{\partial^2{#1}}{\partial{#2}^2}}
$$

</div>


Duality is a core concept of optimization that can be elusive and difficult to grasp initially. In this post I present an intuitive description
of constrained optimization and duality. It is by no means self-contained. A bit of background knowledge in constrained optimization is assumed. This post is mostly useful for those who would like a more concise and intuitive take of some core ideas. This post is best used as a roadmap to Chapter 12 of Numerical Optimization.

Basically we want to solve problems of the form:

$$
\begin{equation}
\min _{x \in \mathbb{R}^{n}} f(x) \quad \text { subject to }\left\{\begin{array}{ll}
c_{i}(x)=0, & i \in \mathcal{E} \\
c_{i}(x) \geq 0, & i \in \mathcal{I},
\end{array}\right.
\end{equation}
$$

And we will make our lives easier by assuming that $$f(.)$$ and $$\{c_i(.)\}$$ are smooth. This means we can have nice linear approximations of $$f(.)$$ and $$\{c_i(.)\}$$ around points of interest. We will use $$\Omega$$ to denote the feasible set.

Before we begin thinking about global optimizers and optimization procedures, we will ask ourselves a seemingly underwhelming question:

*How do we know that a point $$x$$ is definitely NOT a local optimizer?*

## The Easy Stuff

You've probably seen the two canonical examples, one with an equality constraint and one with an inequality constraint.
So to save everyone some time I won't go over them in detail. Feel free to skip ahead if you're feeling like a smartypants. The main ideas are as follows.

- When we have an equality constraint, legal directions lie on the surface parameterized by $$c(.)$$ and are therefore orthogonal to $$\nabla_x c(x)$$. If all directions orthogonal to $$\nabla_x c(x)$$ are not descent directions then we can't move from a candidate minimizer $$x$$ to a better one. This happens when we can't find a direction $$d$$ that satisfies:

$$
\begin{align*}
\nabla_x f(x)^T d &< 0   \\
\nabla_x c(x)^T d &= 0
\end{align*}
$$

or equivalently $$\nabla_x f(x) = \lambda \nabla_x c(x)^T$$ for some $$\lambda \neq 0$$.

- When we have an *active* inequality constraint, legal directions either lie on the surface parameterized by $$c(.)$$ or along directions of increase of $$c(.)$$. If all directions at an acute angle from $$\nabla_x c(x)$$ are not descent directions then we can't move from a candidate minimizer $$x$$ to a better one. This happens when we can't find a direction $$d$$ that satisfies:

$$
\begin{align*}
\nabla_x f(x)^T d &< 0   \\
\nabla_x c(x)^T d &\geq 0
\end{align*}
$$

or equivalently $$\nabla_x f(x) = \lambda \nabla_x c(x)^T$$ for some $$\lambda > 0$$.

When we only have one constraint it is easy to find conditions under which there are no feasible directions. But what if
we have more than one constraint? We basically have a set of linear inequalities in high dimensions and we want to know if the set of solutions is empty. In these cases our geometric intuitions fail us.

Also, since $$c(x)$$ is smooth we can rely on $$\nabla_x c(x)$$ to point us to feasible directions. But this may not be the
case when we have multiple competing constraints that reduce the feasible set to a single point, then no direction is a feasible direction. In this case the gradients of the constraint functions are *deceiving*.

This is where the following distinction comes in handy:

- The tangent cone (to be trusted): from which directions can different paths within the feasible set reach the optimizer? $$T_{\Omega}\left(x^{*}\right)$$ is the set of all directions $$d$$ such that, for some sequence of feasible points $$z_k$$ and vanishing scalars $$t_k$$:

$$
d = \lim _{k \rightarrow \infty} \frac{z_{k}-x}{t_{k}}
$$


- Set of linearized feasible directions (can be deceiving): directions that appear legal under a first order Taylor approximation of the constraints.

$$\mathcal{F}(x)=\left\{\begin{array}{ll} \left.d \mid \begin{array}{ll}d^{T} \nabla c_{i}(x)=0, & & \text { for all } i \in \mathcal{E}, \\ d^{T} \nabla c_{i}(x) \geq 0, & & \text { for all } i \in \mathcal{A}(x) \cap \mathcal{I}\end{array}\right\}\end{array}\right\}$$

What we are *actually* interested in is the former, but we can much more easily work with the latter. We should therefore find conditions that guarantee that the tangent cone is equivalent to the set of linearized feasible directions.

## In Constraint Qualifications We Trust

*When is $$\mathcal{F}(x)$$ equivalent to $$T_{\Omega}(x)$$?*

To show that $$\mathcal{F}(x) = T_{\Omega}(x)$$ under some condition, we need to show that everything in $$T_{\Omega}(x)$$ is also in $$\mathcal{F}(x)$$ and vice versa.

The first part is easy, if $$d$$ $$\in T_{\Omega}(x)$$ then by definition $$d$$ is the limiting direction of $$x - z_k$$ for some sequence of $\{z_k\}$ in $$\Omega$$. Which means that we can take a sufficiently small step in the direction of $$d$$ and end up somewhere in the feasible set $$\Omega$$. Hence $$d \in \mathcal{F}(x)$$.

The second part is not so straightforward. We need to show that under the right conditions, any direction $$d$$ in $\mathcal{F}(x)$ is also in $$T_{\Omega}(x)$$. More concretely, we need to find a map between each $$d \in \mathcal{F}(x)$$ and some path ${z_k}_{k=0}^{\infty}$ in $$\Omega$$ whose limiting tangent is $$d$$.

For some fixed $$d$$, here are the equations that need to be satisfied for some $$t, z$$:

$$
\begin{align}
z  = x + t d
c_i(z) = c_i(x) + t d^T \nabla c_i(x)
\end{align}
$$

The first equation just expresses the condition for $$d$$ to be in $$T_{\Omega}(x)$$.

The second equation depends on the fact that $$d \in \mathcal{F}(x)$$ and therefore if $$t > 0$$, $$t d^T \nabla c_i(x) \geq 0$$ for active inequality constraints and $$t d^T \nabla c_i(x) = 0$$ for equality constraints, making $$z$$ a feasible point.  

What we have so far is a system of non-linear equations (because of $$c_i(z))$$. Which means it's a good time to use the Implicit Function Theorem (IFT) which may have been gathering dust somewhere in the back of your mind. Here's a breakdown of how it's used in our case:

The IFT states that for some multivariable function $$F(z, t)$$, we can find solutions $$(z, t)$$ to the system $$F(z, t) = 0$$ that are "close enough" to a known solution $$z=x$$, $$t=0$$, if the Jacobian $$\nabla_z F(z, 0)$$ is invertible. Specifically, the solutions take the form of a function $$f(t) \rightarrow z$$.

A quick and intuitive interpretation can be provided via a first order Taylor approximation.

$$
\begin{align}
0 = F(z, t) = \nabla_z F(x, 0)(z - x) + \nabla_t F(x, 0) t \\
\nabla_z F(x, 0)(z - x) =  - \nabla_t F(x, 0)t  \\
z = x - \nabla_z F(x, 0)^{-1} \nabla_t F(x, 0)t  \\
\end{align}
$$

 Since the IFT requires that the Jacobian of $$F(z, t)$$ with respect to $$z$$ be invertible, and this is not the case for our current parameterization, we're going to cheat and construct a different system with an invertible Jacobian. First we'll assume that the $$\{\nabla c_i(x)\}_{i \in \mathcal{A}\}$$ are invertible, and make them rows of the matrix $$A(x)$$. Then we'll use a matrix $$Z$$ instead of $$I$$ in our first equation, whose columns span the null space of $$A(x)$$. Now we have a system:

$$
\begin{equation}
 F(z, t)=\left[\begin{array}{c}c(z)-t A\left(x \right) d \\ Z^{T}\left(z-x -t d\right)\end{array}\right]=\left[\begin{array}{c}0 \\ 0\end{array}\right]
\end{equation}
$$

This makes our Jacobian with respect to $$z$$ invertible, and the IFT says that for some fixed $$d$$ given a "small enough" value $$t$$ we can get a $$z$$ that satisfies the above conditions.

Even though we changed the original system of equations, we know that the solutions of the latter are also solutions of the former. To see this, notice that for small $$t$$ the first equation can be approximated (using Taylor) by: $$A(x)(z - x - td)$$, which means that $$(z - x - td)$$ is required to be in the null space of $$[A(x), Z^T]$$, which by construction is $$\{0\}$$. Therefore $$(z - x - td)$$ must be $$0$$ as specified in the first system.

Note that our cheating revealed a nice condition that we can use as a constraint qualification: $$\{\nabla c_i(x)\}$$ are linearly independent. Note that this condition was used ad hoc to make our system satisfy the conditions of the IFT. It is likely that this sufficient condition is pessimistic, meaning it misses many cases where $$T_\Omega(x) = \mathcal{F}(x)$$.


## More Than One Constraint and Farkas' Lemma:
*Assuming that some constraint qualifications are satisfied, how do we know that there are no feasible directions $$d$$ from a candidate $$x$$?*

To appreciate the difficulty of finding such a necessary condition when there are multiple constraints, consider all the inequalities that need to be satisfied for all active constraints:

$$
\begin{align}
\nabla f(x)^T d &< 0 \\
\nabla c_i(x)^T d & \geq 0 \text{ for i } \in \mathcal{I} \intersect \mathcal{A} \\
\nabla c_i(x)^T d & = 0 \text{ for i } \in \mathcal{E} \\
\end{align}
$$

In the case of say two equality constraints and no inequality constraints, there's no solution if for some $$\lambda_1, \lambda_2 \neq 0$$:

$$
\begin{equation}
\nabla f(x) = \lambda_1 \nabla c_1(x) + \lambda_2 \nabla c_2(x) \\
\end{equation}
$$

Add in an inequality constraint $$c_3(x)$$. It could be that $$\nabla f(x) \not in span\{\nabla c_1(x) , \nabla c_2(x)\}$$, yet $$\nabla f(x) = \lambda_3 c_3(x)$$ for some $$\lambda_3 > 0$$. It could also be that

Note that the set of feasible descent directions at $$x$$ is a cone of the form:

$$
\begin{equation}
K = \{B y + C w: y \geq 0\}
\end{equation}
$$

Where $$B is a matrix whose rows are $$\{\nabla c_i(x)\}$$ for $$i \in \mathcal{I} \intersect \mathcal{A}$$. And $$C$$ is a matrix whose columns span the subspace orthogonal to $$\{\nabla c_i(x)\}$$ for $$i \in \mathcal{E}$$.

###Farkas' Lemma:

For any $$d$$ \in $$\mathbb{R}^{n}$$, exactly one of the following is true:
- either $$d \in K$$
- $$\exists$$ $$g \in \mathbb{R}^{n}$$ such that $$g^T d \leq 0$$, $$B^Tg > 0$$, $$C^Tg = 0$$.











- What is the point of Farkas's Lemma?
