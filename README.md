# Julia AGU 2022 workshop
Using Julia and GPU programming for numerical modelling

📚 @AGU 2022 workshop - **SCIWS26: Porting Machine Learning and Modeling from a Laptop to a Supercomputer**

## Automatic notebook generation

The presentation slides and the demo notebook are self-contained in a Jupyter notebook [julia-workshop-agu22.ipynb](julia-workshop-agu22.ipynb) that can be auto-generated using literate programming by deploying the [julia-workshop-agu22.jl](julia-workshop-agu22.jl) script.

To reproduce:
1. Clone this git repo
2. Open Julia and resolve/instantiate the project
```julia-repl
using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()
```
3. Run the deploy script
```julia-repl
julia> using Literate

julia> include("deploy_notebooks.jl")
```
4. Then using IJulia, you can launch the notebook and get it displayed in your web browser:
```julia-repl
julia> using IJulia

julia> notebook(dir=pwd())
```
_To view the notebook as slide, you need to install the [RISE](https://rise.readthedocs.io/en/stable/installation.html) plugin_

## Resources

#### Packages
- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
- [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl)

#### Papers
- Bridging HPC Communities through the Julia Programming Language: https://doi.org/10.48550/ARXIV.2211.02740
- High-performance xPU Stencil Computations in Julia: https://doi.org/10.48550/arXiv.2211.15634
- Distributed Parallelization of xPU Stencil Computations in Julia: https://doi.org/10.48550/arXiv.2211.15716

#### Courses and resources
- ETHZ course on solving PDEs with GPUs: https://pde-on-gpu.vaw.ethz.ch
- Advanced GPU HPC optimisation course at CSCS: https://github.com/omlins/julia-gpu-course and https://github.com/maleadt/cscs_gpu_course/
- More [here](https://pde-on-gpu.vaw.ethz.ch/extras/#extra_material)

#### Misc
- Frontier GPU multi-physics solvers: https://ptsolvers.github.io/GPU4GEO/
