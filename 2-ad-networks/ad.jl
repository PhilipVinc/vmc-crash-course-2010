### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ acb3cc9c-12b4-11eb-1ea6-4f4fd21b7a71
begin
	using Pkg
	Pkg.add(["PlutoUI", "Plots", "ForwardDiff"])
	using Plots
	using PlutoUI
	using ForwardDiff
end

# ╔═╡ 0521b3c8-12bb-11eb-20fd-c5abc540dcd2
html"<center><button onclick=present()>Activate supermagical presentation mode</button></center>"

# ╔═╡ 49ab7f72-151c-11eb-2d99-2540c575e59e
md"""
$(html"<h1><center>Crash Course in VMC</h1></center>")
$(html"<h3><center>30 minutes of Automatic Differentiation</h3></center>")

$(html"<h4><center>(in Julia)</h4></center>")

$(html"<p><center> Filippo Vicentini</center></p><p><center> filippo.vicentini@epfl.ch</center></p>")
"""

# ╔═╡ 88624a8c-151e-11eb-162f-81d8f91c1cb7
md"## Setup "

# ╔═╡ c5619c64-151c-11eb-3953-7da3f7df89cb
md"""
This is a fancy, super-interactive notebook on Automatic Differentiation.

No netket here... :😭

If you are curious... How to run this notebook:

- Install Julia: see [https://julialang.org/downloads/](https://julialang.org/downloads/)
  - MacOs: 😊 `brew cask install julia` 😊 
  - Linux: 😱 Check https://julialang.org/downloads/platform/#linux_and_freebsd
    - don't use `apt intstall julia` unless you have recent distro
  - Don't tell Giuseppe

- Install dependencies (only the first time)
```
julia --project=. -e "using Pkg; Pkg.instantiate()" 
```

- Run the notebook
```
julia --project=. -e "using Pluto; Pluto.run()"
```

"""

# ╔═╡ 0e96be80-1786-11eb-2cbc-b1560bcc7295
md"## Optimising a Cost function

A cost function is a _Scalar_ _real valued_ function

$\mathcal{C} : \mathbb{R}^N \rightarrow \mathbb{R}$

And to optimise it we usually need to compute it's gradient

$\vec\nabla\mathcal{C} : \mathbb{R}^N \rightarrow \mathbb{R}^N$
$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \mathcal{W} \rightarrow \vec\nabla\mathcal{C}(\mathcal{W})$ 

"

# ╔═╡ d7bd8802-1786-11eb-31aa-711b6ec57c69
md"""
## Gradients: the general case

Let's now consider a slightly more general case

$f : \mathbb{R}^{D_0} \rightarrow \mathbb{R}^{D_1}$


The first-order differential structure in the neighborhood of the point $x\in\mathbb{R}^{D_0}$ is encoded into the _Jacobian Matrix_

$\left(\mathcal{J}_{f}\right)^i_j(x) = \frac{(\partial f)^i}{\partial x^j}(x)$

Given another function 

$g : \mathbb{R}^{D_1} \rightarrow \mathbb{R}^{D_2}$

the Jacobian matrix for the composed function $h = g\circ f : \mathbb{R}^{D_0}\rightarrow \mathbb{R}^{D_2}$ can be written as:

$\left(\mathcal{J}_{h}\right)^i_j(x) = \frac{(\partial g)^i}{\partial y^k}(f(x))\frac{(\partial f)^k}{\partial x^j}(x)$
$\left(\mathcal{J}_{h}\right)^i_j(x) = \left(\mathcal{J}_{g}\right)^i_k(f(x)) \left(\mathcal{J}_{f}\right)^k_j(x)$

This is the chain rule
"""

# ╔═╡ 1764b466-12b0-11eb-25e2-df662ae5de48
md"""# Our objective: derivatives
In the following: How to compute the derivative of a function $f(x) : \mathbb{R} \rightarrow \mathbb{R}$ on a computer.

The derivative is defined as:

$f'(x) = \frac{df}{dx}(x) = \lim_{\delta\rightarrow 0}\frac{f(x+\delta) - f(x)}{\delta}$
"""

# ╔═╡ 646dade6-12b0-11eb-08c2-fb699277970f
md"""
## Roundoff errors

The easiest way to compute the derivative is by taking a small $\delta$ and using the formula above. 

Choosing a good $\delta$ is hard:

  - if $\delta$ too big $\rightarrow$ wrong result because the formula is asymptotic

  - if $\delta$ too small $\rightarrow$ wrong result because of roundoff errors:

"""


# ╔═╡ 4614c9c0-12b2-11eb-324e-d10bf3b788a1
md"If I take a random (flating-point) number around $10^{-10}$...

$(@bind __δexp Slider(-20:20, default=-10))"

# ╔═╡ fb649acc-12b1-11eb-169d-7ffdc8438666
δ = rand()*10.0 ^(__δexp)

# ╔═╡ 5be24bb0-12b2-11eb-0c9b-e9d432aed109
md"The _machine epsilon_ of $\delta$ is the smallest number we can add to δ and obtain a number different from δ"

# ╔═╡ 0947fa56-12b2-11eb-3559-679b852517a6
eps(δ)

# ╔═╡ fc1cfad0-12b2-11eb-1312-f524f61fd631
md"Usually with double precision (8 byte) floating point number we have 16 digits of precision ($25 - 10 \approx 16$) in the *low* digits."

# ╔═╡ 22693a08-12b2-11eb-0cc7-ef5ca4fca5ff
δ₂ = 1 + δ

# ╔═╡ fa39648c-12b3-11eb-0273-2d3c723d49ad
md"You see that now $δ₂$ has lost information about the lowest digits of $δ$, because it's accurate only up to the 16-th digit."

# ╔═╡ 2d4c133a-12b2-11eb-2e35-abb596e8bd84
eps(δ₂)

# ╔═╡ e3404d00-151e-11eb-0401-7307e3780834
δ̃ = δ₂ - 1

# ╔═╡ 62e89888-12bc-11eb-133e-911baa5701bc
md"## Finite Differencing 

So if i consider the simple function:"

# ╔═╡ 73162d24-12b4-11eb-15a6-079db4539ba9
f(x) = exp(x) + sin(x);

# ╔═╡ 892c68ee-12bc-11eb-36bb-5f72dec05239
md"and it's analytical derivative:"

# ╔═╡ 5c7c1c24-12b5-11eb-0b5b-47016c08ceb9
dfdx(x) = exp(x) + cos(x);

# ╔═╡ 84cde1a0-12b8-11eb-24a6-c72179eb1821
diff_error(diff_fun, x, δ) = abs.(diff_fun.(f, x, δ) .- dfdx(x));

# ╔═╡ 83bd9c62-12b4-11eb-2fd9-577321bfd8eb
forward_difference(f, x, δ) = (f(x+δ) - f(x))/δ; 

# ╔═╡ a8a48c20-12b4-11eb-22fc-39bafed06616
# I first compute the exact values using Arbitrary precision arithmetics
δ_arr = 10 .^ (range(-16, 0, length=100))

# ╔═╡ d6e6d412-1531-11eb-19c9-e9ec766dcddc
md"I select a point x₀ in it's domain and compute the derivative there: $...

$(@bind x₀ Slider(-1:0.01:1, default=0.3))
"


# ╔═╡ 2cb2439a-1532-11eb-2cc4-ef25533e9747
md"x₀ =$x₀"

# ╔═╡ 6124c9f0-12b6-11eb-1cfd-2fa0a8073c57
ϵ_fwd = diff_error(forward_difference, x₀, δ_arr)

# ╔═╡ 422fcb0c-12b5-11eb-35d1-2939fcef5e31
pl = plot(δ_arr, ϵ_fwd, xscale=:log10, yscale=:log10, label="forward difference",  xlabel="step δ", ylabel="error ϵ")

# ╔═╡ a2eb5c90-12bc-11eb-253c-cdb93e5d9634
md"## Finite Differencing: take 2

Instead of using the forward differencing formula, we can use more accurate central difference:"

# ╔═╡ 8b6d33c2-12b5-11eb-160d-650dbb5da796
central_difference(f, x, δ) = (f(x+δ) - f(x-δ))/2δ;

# ╔═╡ 97ecb6e6-12b6-11eb-1e25-25bad787aa96
ϵ_cnt = diff_error(central_difference, x₀, δ_arr)

# ╔═╡ 832609d8-12b6-11eb-06a4-17a58dd35a9b
pl2 = plot!(deepcopy(pl), δ_arr, ϵ_cnt, xscale=:log10, yscale=:log10, label="central difference")

# ╔═╡ 32b8deea-12b7-11eb-2f14-0565a9db5882
md"And this slope also depends on the function that we are evaluating..."

# ╔═╡ 8094bf72-12b7-11eb-02d6-a365f76bcb92
md"""
## Algebraic Approach: Complex step 


The problem with finite differencing is that we are mixing our really small number with the really large number, and so when we do the subtract we lose accuracy.

* We want to keep the perturbation (f'(x)) and the value (f(x)) completely separate 
"""

# ╔═╡ d9100e2e-12b7-11eb-1ee2-6949a38abcd1
md"$f(x + i\delta) = f(x) + f'(x) i \delta + \mathcal{O}(\delta^2)$"

# ╔═╡ 058ff8f6-12b8-11eb-2f97-d59c4fc7517a
md"$if'(x) = \frac{f(x + iδ) - f(x)}{\delta} + \mathcal{O}(\delta)$"

# ╔═╡ 3c22b5c0-12b8-11eb-3cf0-1b5a0755450b
md"If $x$ is real and $f$ is real-valued then $if'$ is purely imaginary, therefore by taking the imaginary part of the lhs and rhs..."

# ╔═╡ 2e2cd25c-12b8-11eb-0985-6de4c6fedc94
md"$f'(x) = \frac{\Im[f(x + iδ) + 0]}{\delta} + \mathcal{O}(\delta)$"

# ╔═╡ 75e37722-12b8-11eb-2d73-691b4aa1634a
md"Let's try this approach:"

# ╔═╡ 7a928a1a-12b8-11eb-2722-a96569ef6407
complex_difference(f, x, δ) = imag(f(x+im*δ)/δ)

# ╔═╡ c1629480-12b8-11eb-1fb9-c3a4ff1e63d2
ϵ_cmplx = diff_error(complex_difference, x₀, δ_arr) .+ eps(Float64)

# ╔═╡ d11c2ac6-12b8-11eb-30cb-8109d0a36300
pl3 = plot!(pl2, δ_arr, ϵ_cmplx, xscale=:log10, yscale=:log10, label="complex step")

# ╔═╡ 6544e846-12b9-11eb-1ea1-8f37a7d315f3
md"WOW! practically 0 error!

This is because f is a real function of real inputs, and the step $iδ$ is purely imaginary, so the real and imaginary part never mix! 

No mixing $\rightarrow$ no numerical cancellation and roundoff errors!
"

# ╔═╡ d8e75d86-12b9-11eb-01d0-451c26fdf2ed
md"""
### Generalizing: Sensitivities and Dual numbers

The derivative can be thought as the sensitivity of a function to it's input:

	how much does f(x₀) changes when the input x₀ changes by a small amount δ?

Thanks to the Taylor's theorem:

$f(x_0+\delta) = f(x_0) + \delta f'(x_0) + ...$

And now, think about the $\delta$ as a component (a bit like the $i$ unit, quaternion directions $i,j,k$, grassman $\epsilon$...)

$\mathbb{D}(\mathbb{R}) \sim \mathbb{R}\times\mathbb{R}$

$a\in\mathbb{D}(\mathbb{R}) = a_{value} + \delta a_{sensitivity}$

$f : \mathbb{D}(\mathbb{R}) \rightarrow \mathbb{D}(\mathbb{R})$
$f(a) = f(a) + \delta f'(a)$ 

$a + b = (a_v + b_v) + \delta ( a_s + b_s)$
$a b = (a_v  b_v) + \delta ( a_s  b_v + a_v  b_s)$

"""

# ╔═╡ 4830a0bc-12d0-11eb-250b-8fd51130ef12
struct Dual{T}
	val::T
	sen::T
end

# ╔═╡ 49e434e6-12d0-11eb-0d5c-c1befca3ed92
Base.:+(f::Dual, g::Dual) = Dual(f.val + g.val, f.sen + g.sen)

# ╔═╡ a7294eb4-12d2-11eb-0751-892ee34934c6
Base.:+(f::Dual, α::Number) = Dual(f.val + α, f.sen)

# ╔═╡ a78da71a-12d2-11eb-12d1-83a0d0fc6d15
Base.:+(α::Number, f::Dual) = f + α

# ╔═╡ b5bd2d70-12e5-11eb-2a70-176997854b11
Base.:*(f::Dual, g::Dual) = Dual(f.val*g.val, f.sen*g.val + f.val*g.sen)

# ╔═╡ a05bf786-12e5-11eb-0aff-1358f8a0d23f
Base.exp(f::Dual) = Dual(exp(f.val), exp(f.val) * f.sen)

# ╔═╡ 7967d372-1789-11eb-092c-07226f7c3a2e
Base.log(f::Dual) = Dual(exp(f.val), inv(f.val) * f.sen)

# ╔═╡ 3a8b83f2-1387-11eb-1edf-75541c38c98d
Base.sin(f::Dual) = Dual(sin(f.val), cos(f.val) * f.sen)

# ╔═╡ d5d8a97a-12e5-11eb-283e-e7ac4613e43c
f(Dual(x₀,1.0)).sen

# ╔═╡ 301e3eb6-1387-11eb-0190-2931daf82f1a
dfdx(x₀)

# ╔═╡ 888a9050-1532-11eb-1823-c327dd8d6834
md"""
However, think about the function
"""

# ╔═╡ f7a98b50-1788-11eb-30d0-a163d703e1fd
md"""
## An example

A nested function 

"""

# ╔═╡ 0ccf3c6e-1789-11eb-03ff-dd75fff32410
h = exp ∘ sin ∘ log

# ╔═╡ 206bbb3a-1789-11eb-3c83-15d41dcd4d23
plot(range(0.01, 3.0, length=100), h)

# ╔═╡ 51bbba96-1789-11eb-218e-ebc94527410a
# Forward mode:
xᵢ = Dual(0.5, 1.0)

# ╔═╡ 62dcd53a-1789-11eb-13f3-39d907a8ba76
y₁ = log(xᵢ)

# ╔═╡ 9e845f9a-1789-11eb-1760-59dfdf42d6e5
y₂ = sin(y₁)

# ╔═╡ b9982c44-1789-11eb-3c11-31655a8c77cf
y₃ = sin(y₂)

# ╔═╡ Cell order:
# ╟─0521b3c8-12bb-11eb-20fd-c5abc540dcd2
# ╟─49ab7f72-151c-11eb-2d99-2540c575e59e
# ╟─88624a8c-151e-11eb-162f-81d8f91c1cb7
# ╟─c5619c64-151c-11eb-3953-7da3f7df89cb
# ╠═acb3cc9c-12b4-11eb-1ea6-4f4fd21b7a71
# ╠═0e96be80-1786-11eb-2cbc-b1560bcc7295
# ╟─d7bd8802-1786-11eb-31aa-711b6ec57c69
# ╠═1764b466-12b0-11eb-25e2-df662ae5de48
# ╟─646dade6-12b0-11eb-08c2-fb699277970f
# ╟─4614c9c0-12b2-11eb-324e-d10bf3b788a1
# ╟─fb649acc-12b1-11eb-169d-7ffdc8438666
# ╟─5be24bb0-12b2-11eb-0c9b-e9d432aed109
# ╠═0947fa56-12b2-11eb-3559-679b852517a6
# ╟─fc1cfad0-12b2-11eb-1312-f524f61fd631
# ╠═22693a08-12b2-11eb-0cc7-ef5ca4fca5ff
# ╟─fa39648c-12b3-11eb-0273-2d3c723d49ad
# ╠═2d4c133a-12b2-11eb-2e35-abb596e8bd84
# ╠═e3404d00-151e-11eb-0401-7307e3780834
# ╟─84cde1a0-12b8-11eb-24a6-c72179eb1821
# ╟─62e89888-12bc-11eb-133e-911baa5701bc
# ╠═73162d24-12b4-11eb-15a6-079db4539ba9
# ╟─892c68ee-12bc-11eb-36bb-5f72dec05239
# ╠═5c7c1c24-12b5-11eb-0b5b-47016c08ceb9
# ╠═83bd9c62-12b4-11eb-2fd9-577321bfd8eb
# ╠═a8a48c20-12b4-11eb-22fc-39bafed06616
# ╟─d6e6d412-1531-11eb-19c9-e9ec766dcddc
# ╟─2cb2439a-1532-11eb-2cc4-ef25533e9747
# ╠═6124c9f0-12b6-11eb-1cfd-2fa0a8073c57
# ╠═422fcb0c-12b5-11eb-35d1-2939fcef5e31
# ╟─a2eb5c90-12bc-11eb-253c-cdb93e5d9634
# ╠═8b6d33c2-12b5-11eb-160d-650dbb5da796
# ╠═97ecb6e6-12b6-11eb-1e25-25bad787aa96
# ╠═832609d8-12b6-11eb-06a4-17a58dd35a9b
# ╟─32b8deea-12b7-11eb-2f14-0565a9db5882
# ╟─8094bf72-12b7-11eb-02d6-a365f76bcb92
# ╟─d9100e2e-12b7-11eb-1ee2-6949a38abcd1
# ╟─058ff8f6-12b8-11eb-2f97-d59c4fc7517a
# ╟─3c22b5c0-12b8-11eb-3cf0-1b5a0755450b
# ╟─2e2cd25c-12b8-11eb-0985-6de4c6fedc94
# ╟─75e37722-12b8-11eb-2d73-691b4aa1634a
# ╠═7a928a1a-12b8-11eb-2722-a96569ef6407
# ╠═c1629480-12b8-11eb-1fb9-c3a4ff1e63d2
# ╠═d11c2ac6-12b8-11eb-30cb-8109d0a36300
# ╟─6544e846-12b9-11eb-1ea1-8f37a7d315f3
# ╟─d8e75d86-12b9-11eb-01d0-451c26fdf2ed
# ╠═4830a0bc-12d0-11eb-250b-8fd51130ef12
# ╠═49e434e6-12d0-11eb-0d5c-c1befca3ed92
# ╠═a7294eb4-12d2-11eb-0751-892ee34934c6
# ╠═a78da71a-12d2-11eb-12d1-83a0d0fc6d15
# ╠═b5bd2d70-12e5-11eb-2a70-176997854b11
# ╠═a05bf786-12e5-11eb-0aff-1358f8a0d23f
# ╠═7967d372-1789-11eb-092c-07226f7c3a2e
# ╠═3a8b83f2-1387-11eb-1edf-75541c38c98d
# ╠═d5d8a97a-12e5-11eb-283e-e7ac4613e43c
# ╠═301e3eb6-1387-11eb-0190-2931daf82f1a
# ╠═888a9050-1532-11eb-1823-c327dd8d6834
# ╟─f7a98b50-1788-11eb-30d0-a163d703e1fd
# ╠═0ccf3c6e-1789-11eb-03ff-dd75fff32410
# ╠═206bbb3a-1789-11eb-3c83-15d41dcd4d23
# ╠═51bbba96-1789-11eb-218e-ebc94527410a
# ╠═62dcd53a-1789-11eb-13f3-39d907a8ba76
# ╠═9e845f9a-1789-11eb-1760-59dfdf42d6e5
# ╠═b9982c44-1789-11eb-3c11-31655a8c77cf
