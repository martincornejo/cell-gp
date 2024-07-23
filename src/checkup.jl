## load dataset
function read_basytec(file; kwargs...)
    columns = [
        "Time[h]",
        "DataSet",
        "t-Step[h]",
        "Line",
        "Command",
        "U[V]",
        "I[A]",
        "Ah[Ah]",
        "Ah-Step",
        "Wh[Wh]",
        "Wh-Step",
        "T1[°C]",
        "Cyc-Count",
        "Count",
        "State",
    ]
    return CSV.File(file; header=columns, delim='\t', comment="~", kwargs...) |> DataFrame
end

function get_cell_id(file)
    pattern = r"(MF\d+|LGL\d+)"
    match_result = match(pattern, file)
    return match_result |> first
end

function load_data(files)
    data = Dict{Symbol,DataFrame}()
    for file in files
        id = get_cell_id(file) |> Symbol
        df = read_basytec(file)
        data[id] = df
    end
    return data
end

## capacity
function calc_capa_cccv(df; line=(21, 22))
    # cc: constant current / cv: constant voltage
    cc, cv = line

    df_cc = filter(:Line => ∈(cc), df)
    df_cv = filter(:Line => ∈(cv), df)

    cap_cc = df_cc[end, "Ah-Step"] |> abs
    cap_cv = df_cv[end, "Ah-Step"] |> abs
    return cap_cc + cap_cv
end

function calc_capa_cc(df; line=21)
    df_cc = filter(:Line => ∈(line), df)
    cap_cc = df_cc[end, "Ah-Step"] |> abs
    return cap_cc
end

function sort_cell_ids(data)
    # sort cell-id by soh
    soh = [calc_capa_cccv(df) for df in values(data)]
    ids = collect(keys(data))[sortperm(soh; rev=true)]
    return ids
end

## OCV
function calc_cocv(df)
    df_cc = filter(:Line => ∈(29), df)
    df_cv = filter(:Line => ∈(30), df)

    cap_cc = df_cc[end, "Ah-Step"] |> abs
    cap_cv = df_cv[end, "Ah-Step"] |> abs
    cap = cap_cc + cap_cv

    v_cc = df_cc[:, "U[V]"]
    s_cc = df_cc[:, "Ah-Step"]
    v_cv = df_cv[:, "U[V]"]
    s_cv = df_cv[:, "Ah-Step"] .+ cap_cc

    v = [v_cc; v_cv]
    s = [s_cc; s_cv]
    f = LinearInterpolation(v, s ./ cap; extrapolate=true)
    return f
end

function calc_docv(df)
    df_cc = filter(:Line => ∈(27), df)
    df_cv = filter(:Line => ∈(28), df)

    cap_cc = df_cc[end, "Ah-Step"] |> abs
    cap_cv = df_cv[end, "Ah-Step"] |> abs
    cap = cap_cc + cap_cv

    v_cc = df_cc[:, "U[V]"]
    s_cc = df_cc[:, "Ah-Step"] .+ cap
    v_cv = df_cv[:, "U[V]"]
    s_cv = df_cv[:, "Ah-Step"] .+ cap_cv

    v = reverse([v_cc; v_cv])
    s = reverse([s_cc; s_cv])
    f = LinearInterpolation(v, s ./ cap; extrapolate=true)
    return f
end

function calc_pocv(df)
    ocv_c = calc_cocv(df)
    ocv_d = calc_docv(df)
    ocv = c -> (ocv_c(c) + ocv_d(c)) / 2
    return ocv
end

function fresh_focv(data)
    return calc_pocv(data[:LGL13818])
end

## internal resistance
function calc_rint(df; timestep=9.99, line=49, i=1.6166)
    cycles = 9
    timestep = timestep / 3600 # hours -> seconds

    # line = 49 # 51
    # i = 4.85 / 3

    # line = 53 # 55
    # i = 4.85 * 2 // 3

    resistances = Float64[]
    for cycle in 1:cycles
        # initial_voltage
        df2 = copy(df)
        filter!(:Line => ∈(line), df2)
        filter!(:Count => ==(cycle), df2)
        init_time = df2[begin, "Time[h]"]
        init_voltage = df2[begin, "U[V]"]

        # voltage after timestep
        idx2 = findfirst(>(init_time + timestep), df[:, "Time[h]"])
        voltage = df[idx2, "U[V]"]

        r = abs(voltage - init_voltage) / i * 1e3 # mΩ
        append!(resistances, r)
    end
    return DataFrame(; soc=0.1:0.1:0.9, r=reverse(resistances))
end


# check-up analysis (table)
function summarize_checkups(data)
    ids = sort_cell_ids(data)

    # SOH
    cap = [calc_capa_cccv(data[id]) for id in ids]
    soh = cap ./ 4.8

    # RDC (50% SOC)
    df_rint = DataFrame(; soc=0.1:0.1:0.9)
    for id in ids
        df_id = calc_rint(data[id])
        df_rint[!, id] .= df_id.r
    end
    filter!(:soc => ==(0.5), df_rint)
    rdc = df_rint[:, ids] |> Array |> vec

    # ΔOCV MAE
    focv = fresh_focv(data)
    soc = 0:0.001:1.0
    ocv_mae = map(ids) do id
        pocv = calc_pocv(data[id])
        δv = (pocv(soc) - focv(soc)) * 1e3 # mV
        mean(abs, δv)
    end

    cap = round.(cap, digits=2)
    soh = round.(soh * 100, digits=1)
    rdc = round.(rdc, digits=1)
    ocv_mae = round.(ocv_mae, digits=1)
    DataFrame(; id=ids, cap, soh, rdc, ocv_mae)
end


## profile
function calc_ibias(df)
    # integrate current profile
    dt = diff(df.t)
    q = cumsum(df[1:end-1, "I[A]"] .* dt) / 3600 # integrated capacity
    pushfirst!(q, 0) # start from 0.0

    # compare to CTS coulomb counter
    Δq = df[end, "Ah-Step"] - q[end] # capacity diff
    Δt = df.t[end] / 3600
    ib = Δq / Δt
    return ib
end

function load_profile(df)
    df = filter(:Line => ∈(35), df)
    df[:, :t] = (df[:, "Time[h]"] .- df[begin, "Time[h]"]) * 3600 # hours -> seconds

    # bias current / correct current integration
    ibias = calc_ibias(df)

    # interpolation functions
    i = ConstantInterpolation(df[:, "I[A]"] .+ ibias, df.t)
    v = ConstantInterpolation(df[:, "U[V]"], df.t)
    q = ConstantInterpolation(df[:, "Ah-Step"], df.t)
    T = ConstantInterpolation(df[:, "T1[°C]"], df.t)

    return (; i, q, v, T)
end

function sample_dataset(data, tt)
    return DataFrame(
        :t => tt,
        :i => data.i(tt),
        :v => data.v(tt),
        :q => data.q(tt),
        :T => data.T(tt)
    )
end

function initial_soc(df)
    # inital SOC was defined in test as 38% of CC capacity
    # -> convert CC capacity to CCCV capacity based SOC
    capa_cccv = calc_capa_cccv(df)
    capa_cc = calc_capa_cc(df)
    return capa_cc / capa_cccv * 0.38
end
