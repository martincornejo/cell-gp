using Dates
using CSV
using DataFrames
using DataInterpolations

using ModelingToolkit
using DifferentialEquations

using Optimization
using OptimizationOptimJL
import ComponentArrays: ComponentArray

using GLMakie

include("checkup.jl")
include("ocv.jl")
include("profiles.jl")
include("ecm.jl")

## load data
file = "data/data_lab_raw/1_Capacity/BaSyTec_1_Capacity_Rate_SDI94Ah_Samsung SDI 94Ah_CH01_CH01_2987_11032022.csv"
cap = reference_capacity(file)

file = "data/data_lab_raw/3_OCV/BaSyTec_Samsung SDI 94Ah_CH01_CH01_3_OCV_SDI94Ah_3029.csv"
ocv_ch, ocv_dch, ocv = extract_ocv(file)

file = "data/data_lab_raw/2_Intraday/BaSyTec2_Intraday_SDI94Ah_2_Samsung SDI 94Ah_CH01_CH01_3015_16032022.csv"
ti = (1.5, 97.5)
data = load_profile(file, ti)

tt = 0:60:(24*3600.0)
df = sample_dataset(data, tt)

## build ECM
fi(t) = data.i(t)
focv(soc) = ocv(soc * cap)

@mtkbuild ecm = ECM(; focv, fi)
ode = ODEProblem(ecm, [ecm.v1 => 0.0, ecm.soc => 0.0], (0.0, 24 * 3600.0), [])


## fit parameters
# initial parameters (and boundaries)
p0 = ComponentArray((;
    u0=ode.u0,
    p=ode.p
))
# lb = ComponentArray((;
#     u0=[0.0, 0.0],
#     p=[75.0, 0.0, 0.0, 0.0]
# ))
# ub = ComponentArray((;
#     u0=[1.0, 0.1],
#     p=[95.0, 10e-3, 10e-3, 10e3]
# ))

adtype = Optimization.AutoForwardDiff() # auto-diff framework
loss = build_loss_function(ode, ecm, df) # loss function
optf = OptimizationFunction((u, p) -> loss(u), adtype)
# opt = OptimizationProblem(optf, p0; lb, ub)
opt = OptimizationProblem(optf, p0)

opt_sol = solve(opt, LBFGS())

## plot fitted model
@unpack u0, p = opt_sol.u
tspan = (0, 3 * 24 * 3600)
ode_opt = remake(ode; u0, p, tspan)
sol = solve(ode_opt)

begin
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    lines!(ax[1], sol.t, data.v(sol.t))
    lines!(ax[1], sol.t, sol[ecm.v])
    lines!(ax[2], sol.t, sol[ecm.v] - data.v(sol.t))
    fig
end

