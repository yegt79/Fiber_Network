
# Fiber Network Based on Human Aortic Elastin

## Project Overview
This project builds on the FEniCS tutorial for fiber networks, available at [FEniCS Arc-Length Fiber Network Tutorial](https://fenics-arclength.readthedocs.io/en/latest/examples/displacement_control/fiber_network.html). 

I’ve extensively modified the original framework to create a custom fiber network model solved for `displacement control` under stretch, designed to replicate the biomechanical behavior of human aortic elastin.

---

## Research Foundation
The project draws from my study of the elastin network in the human aorta:

- **Imaging**: I used a `multi-photon microscope` to image a purified human aortic sample, containing only elastin (no collagen or cells). The fibers were clearly visible.
- ![Elastin Network](Figure_1_Elastin_Network)
- **Analysis**: Using `ImageJ` plugins, I extracted fiber orientations and generated a histogram of their distribution.
- **Purpose**: This data captures the structural organization of elastin fibers in the aorta.

---

## Modeling the Network
To translate the biological data into a computational model, I developed a custom script:

- **Code**: Located in `src/custom_voronoi_generator.py`.
- **Method**: The script uses `SciPy` to generate a `Voronoi-based mesh` from seed points.
- **Alignment**: The mesh incorporates the fiber orientation histogram, ensuring edges align with the actual elastin fibers observed in the aortic sample.

> This approach creates a model that mirrors the real-world structure of human aortic elastin, enabling accurate simulation of its behavior under stretch.

---

## Dependencies <a name="dependencies"></a>
This package relies on FEniCS 2019.1.0. (Note that this is the legacy version NOT FEniCSx). Brief installation instructions are outline below. For more information see the [official FEniCS installation instructions.](https://fenicsproject.org/download/archive/)

### FEniCS on Windows
The simplest way to install FEniCS on Windows 10 is to install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu distribution. Then you can follow the FEniCS installation instructions for a Linux machine.

### FEniCS on Ubuntu
 To install FEniCS on Ubuntu, run these commands:
 
        sudo apt-get install software-properties-common
        sudo add-apt-repository ppa:fenics-packages/fenics
        sudo apt-get update
        sudo apt-get install fenics

**Note that due to recent ufl updates and its issues with legacy dolfin there may be issues running fenics-arclength if you install FEniCS through PPA. See [here](https://fenicsproject.discourse.group/t/announcement-ufl-legacy-and-legacy-dolfin/11583) about known issues and solutions to ufl and legacy dolfin.**
        
### FEniCS on Anaconda (Linux and Mac only) *Recommended*:
 
        conda create -n fenicsproject -c conda-forge fenics
        conda activate fenicsproject

However, the conda FEniCS installation is missing a few important libraries (i.e. scipy, mshr). For convenience, we provide can environment.yml file that contains all the dependencies except for Jupyter notebooks. 
To use the environment.yml file, navigate the to root directory and use the following commands:
 
        conda create -n fenicsproject
        conda activate fenicsproject
        conda env update -f environment.yml

 For M1 macs there might be issues with installing FEniCS. As a workaround, you must first set the conda environment variable to osx-64. As such, the full command to install FEniCS and all the dependencies on an M1 Mac are:
 
        conda create -n fenicsproject
        conda activate fenicsproject
        conda config --env --set subdir osx-64
        conda env update -f environment.yml
 
While the validation scripts and the package can be used without Jupyter notebooks, Jupyter notebooks are required to run the examples in the examples directory. To install jupyter notebooks, the following command should be ruin after using the environment.ymk file:
 
        conda install -c conda-forge jupyter
 
### FEniCS on Docker (Windows, Mac, Linux)
First install [Docker Desktop](https://fenicsproject.org/download/archive/) then run the following command:

        curl -s https://get.fenicsproject.org | bash
You also can start the Docker container with the following command:

        docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current
A more comprehensive and detailed instructions on Docker installation can be found here: [Docker Installation Instructions](https://fenics.readthedocs.io/projects/containers/en/latest/introduction.html).

**Note: For Docker installation, numpy, scipy, and matplotlib should be installed alongside FEniCS by default. To enable jupyter notebooks for FEniCS, please see this link: [Jupyter notebooks for Docker installation of FEniCS](https://fenics.readthedocs.io/projects/containers/en/latest/jupyter.html). However, the scripts in the ``validation`` can be run without Jupyter notebooks.**
 
## Installation
Once FEniCS has been installed, our package can easily be install through pip with the following command:

    pip install git+https://github.com/pprachas/fenics_arclength.git@master

Note that in the case of Docker installation, you might have to add the option --user to use pip installation (i.e. ``pip install --user git+https://github.com/pprachas/fenics_arclength.git@master``).
In cases where the whole github repository is needed, the github repo can first be cloned before installation:

    git clone https://github.com/pprachas/fenics_arclength.git
    cd fenics_arclength
    pip install .

In the case of developer's version, the last line can be replaced with ``pip install -e .``


## Theory <a name="theory"></a>
Here is outline the basic theory of solving nonlinear finite elements and our implementation of the arc-length solver.
### Nonlinear Finite Elements
A nonlinear finite element problem seeks to minimize the residual vector that comes from discretizing the weak form of the energy balance equation (e.g. continuum for beam balance equations). In general the residual cannot be solved exactly and must be approximated through linearization. A common method to solve nonlinear finite element problems uses the Newton-Raphson method:

 ```math
\mathcal{R}(\mathbf{u}_{n+1}) = \mathcal{R}(\mathbf{u}_{n})+\frac{\partial \mathcal{R}(\mathbf{u}_{n})}{\partial \mathbf{u}_{n}}\Delta \mathbf{u}
 ```
 
where $\Delta u = \mathbf{u}_{n+1}-\mathbf{u}_n$.

Newton's method is solved incrementally until the desired convergence criterion. The term $\frac{\partial \mathcal R(\mathbf u_n)}{\partial \mathbf u_n}$
is typically called the tangential stiffness matrix $K_T$. The first term $\mathcal R(\mathbf u_n)$ is the difference between the internal force of the previous step $F^{ext}$, while the second term
$\frac{\partial \mathcal R(\mathbf u_n)}{\partial \mathbf u_n}\Delta \mathbf u$ is the correction of the internal force $F^{int}$, In general, the nonlinear problem is too difficult for the Newton solver to converge. As such, the external load is applied incrementally with the load factor $\lambda^k$ where $k$ is the increment. Putting it all together, the nonlinear problem can be written as:

```math
\mathcal{R}(\mathbf{u}_{n+1},\lambda_{n+1}) = F^{int}(\mathbf{u}_{n+1};\mathbf{u}_{n},\lambda_{n+1})-\lambda_{n+1} F^{ext}(\mathbf{u}_{n})
 ```

#### Conservative Loading
In most cases the external force does not depend on the solution $u$ (i.e. $F^{ext} (u_n) = F^{ext}$ ). These cases are called conservative loading. The problem than can be simplified to:

```math
\mathcal{R}(\mathbf{u}_{n+1},\lambda_{n+1}) = F^{int}(\mathbf{u}_{n+1};\mathbf{u}_{n})-\lambda^k F^{ext}
```

In this case the tangential stiffness matrix $K_T$ can be constructed using just the internal energy (i.e. $K_T = \frac{\partial F^{int}(\mathbf u_{n+1};\mathbf u_n)}{\partial \mathbf{u}_n}$)

#### Non-conservative loading
In the case where the external force depends on the solution $u$, the above assumption cannot be made and the whole residual must be taken into account when constructing the tangential stiffness matrix $K_T$. As a result, $K_T$ will be non-symmetric. Examples of these specific special cases are applied moments around a fixed axis, follower loads (i.e. loads that change direction based on the deformed configuration), pressure loads, etc.


#### Displacement Control
Sometimes instead of prescribing traction, the problem has a boundary condition with prescribed non-zero displacement (i.e. non-homogenous Dirichlet boundary conditions). In this case, similar to Ref.2, the problem is formulated similar to a multifreedom constraint and we construct a constraint matrix $C$ such that: 

```math
\mathbf{u} = C\mathbf{u}_f+\lambda \mathbf{u}_p
```

where $u_f$ and $u_p$ are the free and prescribed displacement nodes respectively, and $\lambda$ is the incremental displacement factor.


The arc length equation needs to be modified and now becomes:

```math
 \mathcal{A}(\mathbf{u}_f,\lambda) = \Delta\mathbf{u}_f^T\Delta\mathbf{u}_f + \psi\Delta\lambda^2Q^TQ-(\Delta s)^2
 ```

where:

```math
 Q = C^TK_T\mathbf{u}_p
```
