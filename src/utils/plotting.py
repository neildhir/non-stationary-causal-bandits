from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsfonts}",
}
plt.rcParams.update(tex_fonts)


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_binary_var_across_time(data: dict, var: str):
    df = DataFrame(data[var]).melt(var_name="t", value_name=var)
    sns.displot(
        data=df,
        x=var,
        col="t",
        kde=False,
        color="crimson",
        height=3,
        bins=4,
        facet_kws={"sharey": False, "sharex": False},
    )
    plt.show()


def plot_all_barplots(rewards: List[list], save: bool = False):

    assert all(len(rewards[0]) == len(rewards[t]) for t in range(1, len(rewards)))
    arms = range(1, len(rewards[0]) + 1)
    sns.set_theme(style="ticks")
    sns.set_context("talk", font_scale=1.2)
    width = 345

    palettes = [{"gray"}, {"gray", "blue"}, {"gray", "red"}]

    # Prepare data and plot for each set of rewards
    for t, reward in enumerate(rewards):

        if t == 0:
            df = DataFrame({"Pomis": arms, "SCM-MAB": reward})
            original_reward = reward
        else:
            df = DataFrame({"Pomis": arms, "SCM-MAB": original_reward, "CCB": reward})

        tidy = df.melt(id_vars="Pomis").rename(columns=str.title)
        cols = tidy.columns
        tidy.rename(columns=dict(zip(cols, ["Pomis", "Model", "Value"])), inplace=True)

        fig, ax = plt.subplots(1, 1, figsize=set_size(width))
        sns.barplot(x="Pomis", y="Value", hue="Model", data=tidy, ax=ax, palette=palettes[t])
        sns.despine()
        plt.rcParams.update(tex_fonts)

        # Axis labels
        plt.ylim(0, 1.0)
        plt.ylabel(r"$\mathbb{{E}}\left[Y_{} \mid \mathrm{{do}}(a)\right]$".format(t))
        plt.xlabel(r"POMIS Arm ID $[a]$")
        # Legend stuff
        plt.gca().legend().set_title("")
        plt.legend(ncol=2, prop={"size": 15}, loc="upper center", framealpha=0.0, bbox_to_anchor=(0.5, 1.1))

        if save:
            fig.savefig(
                "../figures/barplot_t_{}.pdf".format(t), bbox_inches="tight",
            )

        plt.show()


def plot_CR(out, filename=None):
    plt.rcParams.update(tex_fonts)
    sns.set_theme(style="ticks")
    sns.set_context("talk", font_scale=1.2)
    width = 500
    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    for (pi, c) in zip(["TS", "UCB"], ["red", "b"]):
        time_points, mean_x, lower, upper = out[pi]
        ax.plot(time_points, mean_x[time_points], c, lw=3, label=pi, ls="--" if pi == "TS" else "-")
        ax.fill_between(time_points, lower[time_points], upper[time_points], color=c, alpha=0.1, lw=2)

    sns.despine(trim=True)
    plt.legend(ncol=2, prop={"size": 15}, loc="upper center", framealpha=0.0, bbox_to_anchor=(0.5, 1.0))
    plt.ylabel(r"$R_{N_1}$")
    plt.xlabel(r"Round $[n / N_1]$")

    if filename:
        fig.savefig(
            "../figures/CR_{}.pdf".format(filename), bbox_inches="tight",
        )

    plt.show()
