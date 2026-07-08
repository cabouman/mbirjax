Center slice noise
------------------

### Problem

In some conebeam recons, particularly with large sharpness, the central 
few slices display more noise than other slices and are slower to converge.  

### Diagnosis

This appears to be a problem of overshooting on each update.  The large 
sharpness means that the diagonally preconditioned gradient step is the 
dominant term in each update, and the choice of $\alpha$ is designed to minimize
the L2 norm of the error sinogram.  

However, the preconditioned gradient operator $D^{-1} A^T A$ has a top eigenmode 
that is slice dependent, particularly so when the cone angle is large. For full views, 
this eigenmode tends to be fairly uniform within each slice but varies a lot
from slice to slice.  So the update vectors tend to have larger magnitude in the 
slices with large eigenmode magnitude than in other slices.  The central slices
tend to have large eigenmode magnitude.

Since the joint step size balances all the slices, the result tends to be an overshoot
on the large magnitude slices.  This leads to noisy center slices, oscillating 
updates, and slow convergence.  

This is illustrated in center_slice.py, which shows a reconstruction
with large sharpness after only a few iterations, then shows this recon with
narrow intensity window to highlight the noise, next to the top eigenmode
of $D^{-1} A^T A$. The demo also displays 

1. A plot of the L2 norm of this eigenmode as a function of slice
2. A plot of the L2 norm of each update step as a function of slice
3. A plot of $\cos(\theta)$, where $\theta$ is the angle between 2 consecutive update steps. The default settings show negatively correlated updates that grow in magnitude over the range of iterations shown. 

### Possible solutions

#### 1. Add additional preconditioning based on the top eigenmode


Estimate the top eigenmode before recon:
```
        eigenmode = np.random.random(self.get_params('recon_shape'))
        eigenmode *= eigenmode < 0.01
        for j in range(20):
            sinogram = self.forward_project(eigenmode)
            eigenmode = self.back_project(sinogram)
            eigenmode /= fm_hessian.reshape(eigenmode.shape)
            eigenmode /= np.linalg.norm(eigenmode)
        grad_by_slice = np.linalg.norm(eigenmode, axis=(0, 1))
        self.slice_precond = 2 / (2 + np.sqrt(len(grad_by_slice)) * grad_by_slice)
```

Inside the subset updater:  
```
forward_grad = forward_grad * self.slice_precond[None, :]
```
Pros:  Simple update in each step - one multiple per step.

Cons: Need to estimate the eigenmode in advance.  Tends to increase noise on 
extreme slices (which is better than central slices).  

#### 2. Use something like Nesterov or other momentum.  

Pros:  Gradient steps are faster to execute than partition steps.  

Cons: May or may not converge as fast as current VCD. 
Probably requires additional memory on the order of a recon.   

