#src # This is needed to make this run as normal Julia file
using Markdown #src

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
_AGU 2022 - SCIWS26: Porting Machine Learning and Modeling from a Laptop to a Supercomputer_

# Using Julia and GPU programming for numerical modelling

#### Parallel high-performance stencil computations on xPUs

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""

#### Ludovic Räss - ETHZ

_with Sam Omlin (CSCS - ETHZ), Ivan Utkin (ETHZ)_

![gpu](./figures/logo2.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## The nice to have features

Wouldn't it be nice to have single code that on can:
- run both on CPUs and GPUs (xPUs)?
- run on laptops and supercomputers?
- use for prototyping and production?
- run at optimal performance?
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Hold on 🙂 ...
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Why to bother with GPU computing

A brief intro about GPU computing:
- Why we do GPU computing
- Why the Julia choice

![gpu](./figures/gpu.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Why we do GPU computing
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Predict the evolution of natural and engineered systems
- e.g. ice cap evolution, stress distribution, etc...

![ice2](./figures/ice2.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Physical processes that describe those systems are **complex** and often **nonlinear**
- no or very limited analytical solution is available

👉 a numerical approach is required to solve the mathematical model
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Computational costs increase
- with complexity (e.g. multi-physics, coupling)
- with dimensions (3D tensors...)
- upon refining spatial and temporal resolution

![Stokes2D_vep](./figures/Stokes2D_vep.gif)

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Use **parallel computing** _(to address this)_

GPUs are massively parallel devices
- SIMD machine (programmed using threads - SPMD) ([more](https://safari.ethz.ch/architecture/fall2020/lib/exe/fetch.php?media=onur-comparch-fall2020-lecture24-simdandgpu-afterlecture.pdf))
- Further increases the Flop vs Bytes gap

![cpu_gpu_evo](./figures/cpu_gpu_evo.png)

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Taking a look at a recent GPU and CPU:
- Nvidia Tesla A100 GPU
- AMD EPYC "Rome" 7282 (16 cores) CPU

| Device         | TFLOP/s (FP64) | Memory BW TB/s |
| :------------: | :------------: | :------------: |
| Tesla A100     | 9.7            | 1.55           |
| AMD EPYC 7282  | 0.7            | 0.085          |

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Current GPUs (and CPUs) can do many more computations in a given amount of time than they can access numbers from main memory.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Quantify the imbalance:

$$ \frac{\mathrm{computation\;peak\;perf.\;[TFLOP/s]}}{\mathrm{memory\;access\;peak\;perf.\;[TB/s]}} × \mathrm{size\;of\;a\;number\;[Bytes]} $$

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
_(Theoretical peak performance values as specified by the vendors can be used)._
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Back to our hardware:

| Device         | TFLOP/s (FP64) | Memory BW TB/s | Imbalance (FP64)     |
| :------------: | :------------: | :------------: | :------------------: |
| Tesla A100     | 9.7            | 1.55           | 9.7 / 1.55  × 8 = 50 |
| AMD EPYC 7282  | 0.7            | 0.085          | 0.7 / 0.085 × 8 = 66 |


_(here computed with double precision values)_
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
**Meaning:** we can do about 50 floating point operations per number accessed from main memory. Floating point operations are "for free" when we work in memory-bounded regimes.

👉 Requires to re-think the numerical implementation and solution strategies
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Unfortunately, the cost of evaluating a first derivative $∂A / ∂x$ using finite-differences consists of:

1 reads + 1 write => $2 × 8$ = **16 Bytes transferred**

1 (fused) addition and division => **1 floating point operations**
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## How to evaluate performance

_The FLOP/s metric is no longer the most adequate for reporting the application performance of many modern applications on modern hardware._
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
### Effective memory throughput metric $T_\mathrm{eff}$

Need for a memory throughput-based performance metric: $T_\mathrm{eff}$ [GiB/s]

➡  Evaluate the performance of iterative stencil-based solvers.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
The effective memory access $A_\mathrm{eff}$ [GiB], is the sum of:
- twice the memory footprint of the unknown fields, $D_\mathrm{u}$
- known fields, $D_\mathrm{k}$. 
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
The effective memory access divided by the execution time per iteration, $t_\mathrm{it}$ [sec], defines the effective memory throughput, $T_\mathrm{eff}$ [GiB/s]:

$$ A_\mathrm{eff} = 2~D_\mathrm{u} + D_\mathrm{k}, \;\;\; T_\mathrm{eff} = \frac{A_\mathrm{eff}}{t_\mathrm{it}} $$

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
> 💡 note: The upper bound of $T_\mathrm{eff}$ is $T_\mathrm{peak}$ as measured, e.g., by [McCalpin, 1995](https://www.researchgate.net/publication/51992086_Memory_bandwidth_and_machine_balance_in_high_performance_computers) for CPUs or a GPU analogue. 
> 
> Defining the $T_\mathrm{eff}$ metric, we assume that:
> 1. we evaluate an iterative stencil-based solver,
> 2. the problem size is much larger than the cache sizes and
>
> All "convenience" fields should not be stored and can be re-computed on the fly or stored on-chip.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### The Julia choice
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Julia + GPUs  ➡  close to **1 TB/s** memory throughput [_arXiv_](https://doi.org/10.48550/arXiv.2211.15634)

Julia + GPUs + MPI  ➡  close to **94%** parallel efficiency on 2000+ GPUs  [_arXiv_](https://doi.org/10.48550/arXiv.2211.15716)

![perf_gpu](./figures/ps_igg_perf.png)

**And one can get there** 🚀
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
#### Solution to the "two-language problem"

![two_lang](./figures/two_lang.png)

Single code for prototyping and production

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Backend agnostic:
- Single code to run on single CPU or thousands of GPUs
- Single code to run on various CPUs (x86, ARM, Power9, ...) \
  and GPUs (Nvidia, AMD, Intel?)
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Interactive:
- No need for third-party visualisation software
- Debugging and interactive REPL mode
- Efficient for development
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
too good to be true?
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
![ParallelStencil](./figures/parallelstencil.png)


[https://github.com/omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Enough teasing
_check out [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)_

We'll solve the heat diffusion equation

$$ c \frac{∂T}{∂t} = ∇⋅λ ∇T $$

using explicit 2D finite-differences on a Cartesian staggered grid

👉 This notebook is available on GitHub: [https://github.com/luraess/julia-agu-2022](https://github.com/luraess/julia-agu-2022)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Heat solver implementations and performance evaluation

1. Array programming and broadcasting (vectorised Julia CPU)
2. Array programming and broadcasting (vectorised Julia GPU)
3. Kernel programming using ParallelStencil with math-close notation (`FiniteDifferences` module)

**Goal:** get as close as possible to GPU's peak performance, 1355 GiB/s for the Nvidia Tesla A100 GPU.
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Setting up the environment

Before we start, let's activate the environment:
"""
using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()
Pkg.status()

md"""
And add the package(s) we will use
"""
using Plots, CUDA, BenchmarkTools

md"""
### 1. Array programming on CPU

$$ c \frac{∂T}{∂t} = ∇⋅λ ∇T $$

A 24 lines code including visualisation:
"""
function diffusion2D()
    ## Physics
    λ      = 1.0                                           # Thermal conductivity
    c0     = 1.0                                           # Heat capacity
    lx, ly = 10.0, 10.0                                    # Length of computational domain in dimension x and y
    ## Numerics
    nx, ny = 32*2, 32*2                                    # Number of grid points in dimensions x and y
    nt     = 100                                           # Number of time steps
    dx, dy = lx/(nx-1), ly/(ny-1)                          # Space step in x and y-dimension
    ## Array initializations
    T      = zeros(Float64,nx  ,ny  )                      # Temperature
    Ci     = zeros(Float64,nx  ,ny  )                      # 1/Heat capacity
    qTx    = zeros(Float64,nx-1,ny-2)                      # Heat flux in x-dim
    qTy    = zeros(Float64,nx-2,ny-1)                      # Heat flux in y-dim
    ## Initial conditions
    Ci    .= 1.0/c0                                        # 1/Heat capacity (could vary in space)
    T     .= [exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2) for ix=1:size(T,1), iy=1:size(T,2)] # Initial Gaussian Temp
    ## Time loop
    dt     = min(dx^2,dy^2)/λ/maximum(Ci)/4.1              # Time step for 2D Heat diffusion
    opts   = (aspect_ratio=1,xlims=(1,nx),ylims=(1,ny),clims=(0.0,1.0),c=:turbo,xlabel="Lx",ylabel="Ly") # plotting options
    @gif for it = 1:nt
        qTx .= .-λ .* diff(T[:,2:end-1],dims=1)./dx
        qTy .= .-λ .* diff(T[2:end-1,:],dims=2)./dy
        T[2:end-1,2:end-1] .+= dt.*Ci[2:end-1,2:end-1].*(.-diff(qTx,dims=1)./dx .-diff(qTy,dims=2)./dy)
        heatmap(Array(T)',title="it=$it"; opts...)        # Visualization
    end
end
#-
diffusion2D()

md"""
The above example runs on the CPU. What if we want to execute it on the GPU?

### 2. Array programming on GPU

In Julia, this is pretty simple as we can use the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package 
"""
using CUDA

md"""
and add initialise our arrays as `CuArray`s:
"""
function diffusion2D()
    ## Physics
    λ      = 1.0                                           # Thermal conductivity
    c0     = 1.0                                           # Heat capacity
    lx, ly = 10.0, 10.0                                    # Length of computational domain in dimension x and y
    ## Numerics
    nx, ny = 32*2, 32*2                                    # Number of grid points in dimensions x and y
    nt     = 100                                           # Number of time steps
    dx, dy = lx/(nx-1), ly/(ny-1)                          # Space step in x and y-dimension
    ## Array initializations
    T      = CUDA.zeros(Float64,nx  ,ny  )                 # Temperature
    Ci     = CUDA.zeros(Float64,nx  ,ny  )                 # 1/Heat capacity
    qTx    = CUDA.zeros(Float64,nx-1,ny-2)                 # Heat flux in x-dim
    qTy    = CUDA.zeros(Float64,nx-2,ny-1)                 # Heat flux in y-dim
    ## Initial conditions
    Ci    .= 1.0/c0                                        # 1/Heat capacity (could vary in space)
    T     .= CuArray([exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2) for ix=1:size(T,1), iy=1:size(T,2)]) # Initial Gaussian Temp
    ## Time loop
    dt     = min(dx^2,dy^2)/λ/maximum(Ci)/4.1              # Time step for 2D Heat diffusion
    opts   = (aspect_ratio=1,xlims=(1,nx),ylims=(1,ny),clims=(0.0,1.0),c=:turbo,xlabel="Lx",ylabel="Ly") # plotting options
    @gif for it = 1:nt
        qTx .= .-λ .* diff(T[:,2:end-1],dims=1)./dx
        qTy .= .-λ .* diff(T[2:end-1,:],dims=2)./dy
        T[2:end-1,2:end-1] .+= dt.*Ci[2:end-1,2:end-1].*(.-diff(qTx,dims=1)./dx .-diff(qTy,dims=2)./dy)
        heatmap(Array(T)',title="it=$it"; opts...)        # Visualization
    end
end
#-
diffusion2D()

md"""
Nice, so it runs on the GPU now. But how much faster - what did we gain? Let's determine the effective memory throughput $T_\mathrm{eff}$ for both implementations.

### CPU vs GPU array programming performance

For this, we can isolate the physics computation into a function that we will evaluate for benchmarking
"""
function update_temperature!(T, qTx, qTy, Ci, λ, dt, dx, dy)
    @inbounds qTx .= .-λ .* diff(T[:,2:end-1],dims=1)./dx
    @inbounds qTy .= .-λ .* diff(T[2:end-1,:],dims=2)./dy
    @inbounds T[2:end-1,2:end-1] .+= dt.*Ci[2:end-1,2:end-1].*(.-diff(qTx,dims=1)./dx .-diff(qTy,dims=2)./dy)
    return
end

md"""
Moreover, for benchmarking activities, we will require the following arrays and scalars and make sure to use sufficiently large arrays in order to saturate the memory bandwidth:
"""
nx = ny = 512*32
T   = rand(Float64,nx  ,ny  )
Ci  = rand(Float64,nx  ,ny  )
qTx = rand(Float64,nx-1,ny-2)
qTy = rand(Float64,nx-2,ny-1)
λ = dx = dy = dt = rand();

md"""
And use `@belapsed` macro from [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) to sample our perf:
"""
t_it = @belapsed begin update_temperature!($T, $qTx, $qTy, $Ci, $λ, $dt, $dx, $dy); end
T_eff_cpu_bcast = (2*1+1)/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_cpu_bcast) GiB/s using CPU array programming")

md"""
Let's repeat the experiment using the GPU
"""
nx = ny = 512*32
T   = CUDA.rand(Float64,nx  ,ny  )
Ci  = CUDA.rand(Float64,nx  ,ny  )
qTx = CUDA.rand(Float64,nx-1,ny-2)
qTy = CUDA.rand(Float64,nx-2,ny-1)
λ = dx = dy = dt = rand();

md"""
And sample again our performance from the GPU execution this time:
"""
t_it = @belapsed begin update_temperature!($T, $qTx, $qTy, $Ci, $λ, $dt, $dx, $dy); synchronize(); end
T_eff_gpu_bcast = (2*1+1)/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_gpu_bcast) GiB/s using GPU array programming")

md"""
We see some improvement from performing the computations on the GPU, however, $T_\mathrm{eff}$ is not yet close to GPU's peak memory bandwidth

How to improve? Now it's time for ParallelStencil

### 3. Kernel programming using ParallelStencil

In this first example, we'll use the `FiniteDifferences` module to enable math-close notation and the `CUDA` "backend". We could simply switch the backend to `Threads` if we want the same code to run on multiple CPU threads using Julia's native multi-threading capabilities. But for time issues, we won't investigate this today.
"""
USE_GPU=true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA,Float64,2)
    CUDA.device!(2) # select specific GPU
else
    @init_parallel_stencil(Threads,Float64,2)
end
nx = ny = 512*64
T   = @rand(nx  ,ny  )
Ci  = @rand(nx  ,ny  )
qTx = @rand(nx-1,ny-2)
qTy = @rand(nx-2,ny-1)
λ = dx = dy = dt = rand();

md"""
Using math-close notations from the `FiniteDifferences2D` module, our update kernel can be re-written as following:
"""
@parallel function update_temperature_ps!(T, qTx, qTy, Ci, λ, dt, dx, dy)
    @all(qTx) = -λ * @d_xi(T)/dx
    @all(qTy) = -λ * @d_yi(T)/dy
    @inn(T)   = @inn(T) + dt*@inn(Ci)*(-@d_xa(qTx)/dx -@d_ya(qTy)/dy)
    return
end

md"""
And sample again our performance on the GPU using ParallelStencil this time:
"""
t_it = @belapsed begin @parallel update_temperature_ps!($T, $qTx, $qTy, $Ci, $λ, $dt, $dx, $dy); end
T_eff_ps = (2*1+1)/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_ps) GiB/s using ParallelStencil on GPU and the FiniteDifferences2D module")

md"""
It's already much better, but we can do more in order to approach the peak memory bandwidth of the GPU - 1355 GiB/s for the Nvidia Tesla A100.

We should now remove the convenience arrays `qTx`, `qTy` (no time dependence), as these intermediate storage add pressure on the memory bandwidth which slows down the calculations since we are memory-bound.
We can rewrite it as following assuming that `λ` is constant (a scalar here).
"""
@parallel function update_temperature_ps2!(T2, T, Ci, λ, dt, dx, dy)
    @inn(T2) = @inn(T) + dt*λ*@inn(Ci)*(@d2_xi(T)/dx/dx + @d2_yi(T)/dy/dy)
    return
end

md"""
We can sample again our performance on the GPU:
"""
CUDA.unsafe_free!(qTx)
CUDA.unsafe_free!(qTy)
T2 = copy(T)
t_it = @belapsed begin @parallel update_temperature_ps2!($T2, $T, $Ci, $λ, $dt, $dx, $dy); end
T_eff_ps2 = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff_ps2) GiB/s using ParallelStencil on GPU without convenience arrays")
println("So, we made it. Our 2D diffusion kernel runs on the GPU at $(T_eff_ps2/1355) % of memory copy!")

md"""
So that's cool. We are getting close to hardware limit 🚀

> 💡 note: We need a buffer array now in order to avoid race conditions and erroneous results when accessing the `T` array in parallel.

Time to recap what we've seen.
"""

md"""
## Conclusions

- Starting with performance, we can now clearly see our 4 data points of $T_\mathrm{eff}$ and how close the GPU performance is from the peak memory bandwidth of the GPU
"""
xPU  = ("CPU-AP", "GPU-AP", "GPU-PS", "GPU-PS2")
Teff = [T_eff_cpu_bcast, T_eff_gpu_bcast, T_eff_ps, T_eff_ps2]
plot(Teff,ylabel="T_eff [GiB/s]",xlabel="implementation",xticks=(1:length(xPU),xPU),xaxis=([0.7, 4.3]),linewidth=0,markershape=:square,markersize=8,legend=false,fontfamily="Courier",framestyle=:box)
plot!([0.7,4.3],[1355,1355],linewidth=3)
annotate!(1.4,1300,"memory copy")

md"""
- Julia and ParallelStencil permit to solve the two-language problem
- ParallelStencil and Julia GPU permit to exploit close to GPUs' peak memory throughput

![parallelstencil](./figures/parallelstencil.png)

Wanna more? Check out [https://github.com/omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) and the [miniapps](https://github.com/omlins/ParallelStencil.jl#concise-singlemulti-xpu-miniapps)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Outlook

Advance features not covered today:
- Using shared-memory and 2.5D blocking (see [here](https://github.com/omlins/ParallelStencil.jl#support-for-architecture-agnostic-low-level-kernel-programming) with [example](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion2D_shmem_novis.jl))
- Multi-GPU with communication-computation overlap combining ParallelStencil and [ImplicitGlobalGrid]()
- Stay tuned, AMDGPU support is coming soon 🚀
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Enjoy reading? **Check out**

- Bridging HPC Communities through the Julia Programming Language [arXiv](https://doi.org/10.48550/ARXIV.2211.02740)
- High-performance xPU Stencil Computations in Julia [arXiv](https://doi.org/10.48550/arXiv.2211.15634)
- Distributed Parallelization of xPU Stencil Computations in Julia [arXiv](https://doi.org/10.48550/arXiv.2211.15716)

_contact: luraess@ethz.ch_
"""
