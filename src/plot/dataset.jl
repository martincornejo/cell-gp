function plot_checkup_profile(df)
    fig = Figure(size=(516, 180), fontsize=10, figure_padding=5)
    gl = GridLayout(fig[1, 1])
    ax1 = Axis(gl[1, 1]; ylabel="Voltage / V")
    ax2 = Axis(gl[2, 1]; ylabel="Current / A", xlabel="Time / h")

    lines!(ax1, df[:, "Time[h]"], df[:, "U[V]"], color=Cycled(1))
    lines!(ax2, df[:, "Time[h]"], df[:, "I[A]"], color=Cycled(2))
    xlims!(ax1, (df[begin, "Time[h]"], df[end, "Time[h]"]))
    xlims!(ax2, (df[begin, "Time[h]"], df[end, "Time[h]"]))

    idx = [findfirst(df[:, "Line"] .== line) for line in (25, 35, 36, 47)]
    vlines!(ax1, df[idx, "Time[h]"], color=:gray, linestyle=:dashdot)
    vlines!(ax2, df[idx, "Time[h]"], color=:gray, linestyle=:dashdot)

    hidexdecorations!(ax1, grid=false, ticks=false)
    rowgap!(gl, 5)
    fig
end

function plot_ocvs(data)
    fig = Figure(size=(252, 350), fontsize=10, figure_padding=8)
    gl = GridLayout(fig[1, 1])
    colormap = :dense
    colorrange = (0.6, 1.0)

    Colorbar(gl[1, 1]; colorrange, colormap, label="SOH / p.u.", vertical=false) #, flipaxis=false)
    ax1 = Axis(gl[2, 1]; ylabel="OCV / V") # mean pOCV
    ax2 = Axis(gl[3, 1]; xlabel="SOC / p.u.", ylabel="δOCV / mV")


    focv = fresh_focv()
    s = 0:0.001:1.0

    for (id, df) in data
        soh = calc_capa_cccv(df) / 4.9

        # mean ocv
        pocv = calc_pocv(df)
        lines!(ax1, s, pocv(s); color=soh, colorrange, colormap)

        # degradation
        if id != :LGL13818
            δv = (pocv(s) - focv(s)) * 1e3 # mV
            lines!(ax2, s, δv; color=soh, colorrange, colormap)
        end
    end
    ylims!(ax2, -120, 35)
    rowsize!(gl, 2, Auto(1.5))
    rowsize!(gl, 3, Auto(1))
    rowgap!(gl, 6)
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1, grid=false, ticks=false)
    return fig
end

function plot_rints(data; timestep=9.99)
    fig = Figure(size=(252, 300), fontsize=10, figure_padding=8)
    gl = GridLayout(fig[1, 1])
    soc = 0.9:-0.1:0.1
    colormap = :dense
    colorrange = (0.6, 1.0)

    Colorbar(gl[1, 1]; colorrange, colormap, label="SOH / p.u.", vertical=false)
    ax = Axis(gl[2, 1]; xlabel="SOC / p.u.", ylabel="Pulse resistance / mΩ")
    for (id, df) in data
        soh = calc_capa_cccv(df) / 4.9
        r = calc_rint(df; timestep) * 1e3 # mΩ

        scatter!(ax, soc, r; color=soh, colorrange, colormap)
    end

    rowgap!(gl, 6)
    return fig
end
