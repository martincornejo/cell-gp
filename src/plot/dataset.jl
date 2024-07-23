function plot_checkup_profile(df)
    fig = Figure(size=(516, 180), fontsize=10, figure_padding=5)
    gl = GridLayout(fig[1, 1])
    ax1 = Axis(gl[1, 1]; ylabel="Voltage / V")
    ax2 = Axis(gl[2, 1]; ylabel="Current / A", xlabel="Time / h")
    linewidth = 1.2

    lines!(ax1, df[:, "Time[h]"], df[:, "U[V]"]; color=Cycled(1), linewidth)
    lines!(ax2, df[:, "Time[h]"], df[:, "I[A]"]; color=Cycled(2), linewidth)
    xlims!(ax1, (df[begin, "Time[h]"], df[end, "Time[h]"]))
    xlims!(ax2, (df[begin, "Time[h]"], df[end, "Time[h]"]))

    idx = [findfirst(df[:, "Line"] .== line) for line in (25, 35, 36, 47)]
    vlines!(ax1, df[idx, "Time[h]"], color=:gray, linestyle=:dashdot)
    vlines!(ax2, df[idx, "Time[h]"], color=:gray, linestyle=:dashdot)

    hidexdecorations!(ax1, grid=false, ticks=false)
    rowgap!(gl, 5)
    fig
end


function plot_checkups(data)
    fig = Figure(size=(512, 225), fontsize=10, figure_padding=5)
    colormap = ColorSchemes.dense
    colorrange = (0.6, 1.0)

    gl1 = GridLayout(fig[1, 1])
    ax11 = Axis(gl1[1, 1]; ylabel="OCV / V") # mean pOCV
    ax12 = Axis(gl1[2, 1]; xlabel="SOC / p.u.", ylabel="ΔOCV / mV")

    ax13 = Axis(fig[1, 1],
        width=Relative(0.45),
        height=Relative(0.32),
        halign=0.9,
        valign=0.6,
    )
    hidedecorations!(ax13)
    ylims!(ax13, 3.9, 4.1)
    xlims!(ax13, 0.7, 0.9)
    # ax13.yticks = [3.9, 4.1]
    # ax13.xticks = [0.7, 0.9]
    lines!(ax11, [0.5, 0.7], [3.5, 3.9], color=(:black, 0.5), linestyle=:dot, strokewidth=1)
    lines!(ax11, [0.9, 1.0], [3.9, 3.5], color=(:black, 0.5), linestyle=:dot, strokewidth=1)
    poly!(ax11, Point2f[(0.7, 3.9), (0.9, 3.9), (0.9, 4.1), (0.7, 4.1)], color=:white, strokewidth=1, linestyle=:dash, strokecolor=(:black, 0.5))

    ylims!(ax12, -120, 35)
    rowsize!(gl1, 1, Auto(2))
    rowsize!(gl1, 2, Auto(1))
    hidexdecorations!(ax11, grid=false, ticks=false)
    linkxaxes!(ax11, ax12)
    rowgap!(gl1, 6)

    gl2 = GridLayout(fig[1, 2])
    ax21 = Axis(gl2[1, 1]; xlabel="SOC / p.u.", ylabel="Pulse resistance / mΩ")

    Colorbar(fig[1, 3]; colorrange, colormap, label="SOH / p.u.") #, flipaxis=false)

    focv = fresh_focv(data)
    soc = 0:0.001:1.0

    for (id, df) in data
        # soh
        soh = calc_capa_cccv(df) / 4.8
        color = get(colormap, soh, colorrange)

        # mean ocv
        pocv = calc_pocv(df)
        lines!(ax11, soc, pocv(soc); color)
        lines!(ax13, soc, pocv(soc); color)

        # ocv degradation
        if id != :LGL13818
            Δv = (pocv(soc) - focv(soc)) * 1e3 # mV
            lines!(ax12, soc, Δv; color)
        end

        # r dc
        r = calc_rint(df)
        scatter!(ax21, r.soc, r.r; color)
    end

    return fig
end
