from pandas import DataFrame
from seaborn import displot
from matplotlib.pyplot import show


def plot_binary_var_across_time(data: dict, var: str):
    df = DataFrame(data[var]).melt(var_name="t", value_name=var)
    displot(
        data=df,
        x=var,
        col="t",
        kde=False,
        color="crimson",
        height=3,
        bins=4,
        facet_kws={"sharey": False, "sharex": False},
    )
    show()
