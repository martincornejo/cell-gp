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

function fresh_focv()
    file = "data/check-ups/2098LG_INR21700-M50L_SammyLGL13818NewFullCU.txt"
    df = read_basytec(file)
    return calc_pocv(df)
end

## internal resistance
function calc_rint(df; timestep=9.99)
    cycles = 9
    timestep = timestep / 3600 # hours -> seconds

    # line = 49 # 51
    # i = 4.85 / 3

    line = 53 # 55
    i = 4.85 * 2 // 3

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

        r = abs(voltage - init_voltage) / i
        append!(resistances, r)
    end
    return resistances
end


## profile
function load_profile(df)
    df = filter(:Line => ∈(35), df)

    df[:, :t] = (df[:, "Time[h]"] .- df[begin, "Time[h]"]) * 3600 # hours -> seconds

    i = ConstantInterpolation(df[:, "I[A]"], df.t)
    v = ConstantInterpolation(df[:, "U[V]"], df.t)
    s = ConstantInterpolation(df[:, "Ah-Step"], df.t)
    T = ConstantInterpolation(df[:, "T1[°C]"], df.t)

    return (; i, s, v, T)
end

function sample_dataset(data, tt)
    return DataFrame(
        :t => tt,
        :i => data.i(tt),
        :v => data.v(tt),
        :s => data.s(tt),
        :T => data.T(tt)
    )
end
