## mtk model
@mtkmodel ECM begin
    begin
        @variables t
        D = Differential(t)
    end
    @parameters begin
        Q = 4.8
        R0 = 1e-3
        R1 = 1e-3
        τ1 = 60
        R2 = 1e-3
        τ2 = 3600
    end
    @variables begin
        i(t)
        v(t)
        vr(t)
        v1(t) = 0.0
        v2(t) = 0.0
        ocv(t)
        soc(t)
    end
    @structural_parameters begin
        focv
        fi
    end
    @equations begin
        D(soc) ~ i / (Q * 3600.0)
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1)
        D(v2) ~ -v2 / τ2 + i * (R2 / τ2)
        vr ~ i * R0
        ocv ~ focv(soc)
        i ~ fi(t)
        v ~ ocv + vr + v1 + v2
    end
end


## fit model
function build_loss_function(prob, model, df)
    loss(x) = begin
        (; u, p) = x
        u0 = vcat(prob.u0[1], u) # add inital soc

        newprob = remake(prob; u0, p)
        sol = solve(newprob, Tsit5(); saveat=df.t)
        v = sol[model.v]
        sum(abs2, v - df.v)
    end
end

function fit_ecm(df, tt, focv)
    # load dataset
    profile = load_profile(df)
    df_train = sample_dataset(profile, tt)
    soc = initial_soc(df)

    # build model
    fi = t -> profile.i(t)
    @mtkbuild ecm = ECM(; focv, fi)
    tspan = (tt[begin], tt[end])
    ode = ODEProblem(ecm, [ecm.soc => soc], tspan, [])

    # build optimization problem
    p0 = ComponentArray((; # param initial guess
        u=zeros(2),
        p=ode.p
    ))
    adtype = Optimization.AutoForwardDiff() # auto-diff framework
    loss = build_loss_function(ode, ecm, df_train) # loss function
    optf = OptimizationFunction((u, p) -> loss(u), adtype)
    opt = OptimizationProblem(optf, p0)

    # optimize with multiple initial values, select best fit
    solutions = []
    for Q in 3.6:0.4:4.8
        # update initial guess of Q
        p0.p[1] = Q
        opt = remake(opt; u0=p0)

        # solve
        alg = LBFGS(linesearch=BackTracking())
        sol = solve(opt, alg; reltol=1e-4)
        if sol.retcode == ReturnCode.Success
            push!(solutions, sol)
        end
    end
    opt_sol = argmin(sol -> sol.objective, solutions)

    # build parametrized model
    (; u, p) = opt_sol.u
    u0 = vcat(ode.u0[1], u) # add inital soc
    ode = remake(ode; p, u0)

    return (; ecm, ode)
end


function fit_ecm_series(data)
    tt = 0:(3*24*3600.0) # training time window
    focv = fresh_focv()
    res = Dict()
    Threads.@threads for id in collect(eachindex(data))
        df = data[id]
        res[id] = fit_ecm(df, tt, focv) # model
        @info "$id ECM fit complete."
    end
    return res
end

