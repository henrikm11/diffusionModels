# Diffusion Models via Score Matching

## Brief Intro to Diffusion

Diffusion models are a machine learning method to generate new samples from an unknown distribution that is only accessible via finitely many samples.

The approach is that given the unknown distribution $\pi$ on $\mathbb{R}^d$ we consider its evolution $X_t$ along a diffusion process given by

$$
dX_t = \mu(X_t,t) dt + \sigma(X_t,t) dW_t,
$$

where $W_t$ is a standard  $k$-dimensional Wiener process, $\mu$ is $\mathbb{R}^d$ valued and $\sigma$ is $\mathbb{R}^{n\times d}$ valued.

For simplicity (and also some more serious reasons), we only consider the case $\sigma(X,t) = \sigma(t) \rm{id}$, where we also abuse notation.
One can show that in this case the time reverse process $Y_t=X_{T-t}$ is a diffusion process itself given by

$$
dY_t =  \left( \mu(Y_t, T-t) + \sigma^2(T-t) \nabla_x \log p (Y_t,T-t)\right) dt + \sigma(T-t) dW_t,
$$

where $p(\cdot,t)$ denotes the density of $X_t$ which solves the associated Fokker--Planck equation.

For choices of $\mu$ and $\sigma$ such that  the distribution of $X_T$ is approximately known for large $T$, e.g. the Ornstein--Uhlenbeck process, 
\this converts the problem of sampling from $\pi$ into the problem of approximating the score function $\nabla_x \log p (Y_t,T-t)$.
This is because if we know both the distribution of $X_T$ and the score approximately, we can use the time reversed diffusion to sample backwards in time to an approximation of the original distribution $\pi$.


Amazingly, it turns out that the problem of approximating the score function can be converted into supervised learning problem.
There is different approaches on how to see this, the one taken by our implementation is to use the denoising score matching which has the objective to minimize

$$
E = \int_t^T \int\{\mathbb{R}^d} \int_{\mathbb{R}^d} | s_\theta(x,t) - \nabla_x p(x,t | x_0)|^2 p(x,t | x_0) d \pi(x_0) dx dt,
$$

over some parametrized class of function $s_\theta$. (In practice this will be some type of neural network.)

For explicit (simple) choices for the functions $\mu$ and $\sigma$ one has an explicit formula for the transition kernel $p(x,t | x_0)$ so that the objective is admissible to be approximated via Monte Carlo.



## Our Implementation

We provide an already fairly general, but als easily extendable, framework for diffusion processes. The interface for this can be found in diffusion.h
This in turn relies on a framework for some basic function objects that allow us convenient and simple arithmetics for the functions involved. The interface and documentation for this can be found in FuncHelper.h which provides polymorphic wrapper classes for functions as they occur in diffusion processes.



