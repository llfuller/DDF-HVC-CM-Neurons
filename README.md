# DDF-HVC-CM-Neurons
 Code for analyzing zebra finch HVC and CM neuron data using radial basis function (RBF) data-driven models (DDMs). Uses Randall Clark's DDF Python code as the core routine.

Notes on DDF algorithm (from readme of Randall Clark's github repo):

# DDF-Applications-to-Neurons

This Repository provides python code for performing DDF on a Neuron. There are two special things about this code that make it different than the usual Radial Basis Code I have in another repository. The first is that it uses Time Delay Dimensions, this is how we overcome the obstacle of only have 1 dimension of observed data; the process is actually quite simple, we just create extra dimensions from the observed data set that include a time delay. The code already does this step for you (the functions will have comments explaining this), and you will only need to input 1 dimension of data into the functions and specify how many time delay dimensions(D) you want and how large you want the delays(tau) to be. The second difference is that we include a polynomial term in the vector field represntation; originally, we only needed a sum of radial basis expansions, now we include a 1st order term polynomial, the current. The Function representation now looks like this:
f(V,t) = sum(RBF(V(t),c_q)) + w*I(t)


Keep in mind that V is a D dimensional vector of time delays (V(t),V(t-tau),...V(t-tau*(D-1)). Then the update rule looks like this:

V_0(n+1) = V_0(n) + sum(RBF(V(n),c_q)) + w*I(n)


V_0 is the leading time delay dimension. In this formulation we only update the leading term and allow the rest of the dimenions to follow it. There is another formulation that we have constructed where each of the time delays are allowed to forecast individually and the only interdependence comes from the V vector in the RBF's; this method was found to be less effective, so we now push for only forecasting the leading term, V_0 and update the time delays to be previous values of V_0 (it's also faster to only predict on 1 dimension).


We also include an example result in the folder titled "Example" where we tested this method on an NaKL Voltage. Note that we did not tell DDF what the other gating variables were, it only knew about the Voltage and its time delays. This is significant because in real world data, we can only measure the voltage of a neuron and the stimulus provided to it (note that the external stimulus is not a dynamical variable of the dynamics, it is external, and we only have 1 dynamical variable, the voltage). I also include the centers for the Radial Basis Function, they were calculated using the K-means function in the python code.


DDF Basics:
Here is a quick refresher on what DDF is doing and what the code is trying to accomplish.
Starting with the dynamical equations, we have some system that has the following differential equations

dx(t)/dt = F(x(t))

We want to model the behavior of the observed variable x(t), but F(x(t)) is unkown to us. We can approximate the problem with the Euler Formula

x(n+1) = x(n) + dt*F(x(t))

Now we want a Function Representation for F(x(t)). Inspired by our knowledge of NaKL, we choose a representation the form:

f(x(t)) = sum(RBF(V(n),c_q)) + w*I(n)

We use these two equations above to write down a cost function to fit our coefficients in the RBF's and the w infront of I:

Minimize sum_length [(x(n+1)-x(n)) - sum(RBF(V(n),c_q)) + w*I(n)]^2

Because the function representation is linear in the coefficients, we can rewrite the formula in terms of W*X where W are the weights, and X is the value of either the RBF or the Current. This minimization problem can be solved with Ridge Regression:

W = YX^T(XX^T)^-1

[Y] = 1 x Time

[X] = Parameter Length x Time

with the minizimation done, f(x(t)) can now be used to forecast forward in time.





