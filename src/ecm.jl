## mtk model
@variables t
D = Differential(t)

@mtkmodel ECM begin
    @parameters begin
        Q = 4.9
        R0 = 1.0e-3
        R1 = 1.0e-3
        τ1 = 50
    end
    @variables begin
        i(t)
        v(t)
        vr(t)
        v1(t)
        ocv(t)
        soc(t)
    end
    @structural_parameters begin
        focv
        fi
    end
    @equations begin
        D(soc) ~ i / (Q * 3600.0)
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1) # check sign
        vr ~ i * R0
        ocv ~ focv(soc)
        i ~ fi(t)
        v ~ ocv + vr + v1
    end
end


## fit model
function build_loss_function(prob, model, df)
    loss(x) = begin
        @unpack u0, p = x
        newprob = remake(prob; u0, p)
        sol = solve(newprob, Tsit5(); saveat=df.t)
        v = sol[model.v]
        sum(abs2, v - df.v)
    end
end

function fit_ecm(df, tt, focv) # input only data instead?
    # load dataset
    profile = load_profile(df)
    df_train = sample_dataset(profile, tt)

    # build model
    fi = t -> profile.i(t)
    @mtkbuild ecm = ECM(; focv, fi)

    tspan = (tt[begin], tt[end])
    ode = ODEProblem(ecm, [ecm.v1 => 0.0, ecm.soc => 0.5], tspan, [])

    # build optimization problem
    p0 = ComponentArray((;
        u0=ode.u0,
        p=ode.p
    ))
    adtype = Optimization.AutoForwardDiff() # auto-diff framework
    loss = build_loss_function(ode, ecm, df_train) # loss function
    optf = OptimizationFunction((u, p) -> loss(u), adtype)
    opt = OptimizationProblem(optf, p0)
    # opt_sol = solve(opt, LBFGS()) <-- simple LBFGS optimization

    # optimize with multiple alpha-values, select best fit
    solutions = []
    for alpha in (1, 10, 100, 1000)
        alg = LBFGS(;
            alphaguess=Optim.LineSearches.InitialStatic(; alpha),
            linesearch=Optim.LineSearches.BackTracking(),
        )
        sol = solve(opt, alg)
        if sol.retcode == ReturnCode.Success
            push!(solutions, sol)
        end
    end
    opt_sol = argmin(sol -> sol.objective, solutions)

    @unpack u0, p = opt_sol.u
    ode = remake(ode; u0, p, tspan)

    return (; ecm, ode)
end
