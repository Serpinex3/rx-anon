"""Main application to evaluate and plot experiment results"""
import json
import os
import statistics

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from pathlib import Path

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 16
})


y_min = -0.05
y_max = 1.05
y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fig_width = 8
fig_height = 8
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', 'lime', 'darkblue', 'magenta']

information_loss_filter = ["0_0.", "0_1.", "0_2.", "0_3.", "0_4.", "0_5.", "1_0.", "gdf"]


def merge_information_loss_results(path, information_loss_files, destination):
    """Merges partial information loss files to a single file"""
    data_frames = []
    if len(information_loss_files) > 0:
        for information_loss_file in information_loss_files:
            data_frames.append(pd.read_csv(path / information_loss_file, index_col="k"))
        df_il = pd.concat(data_frames, axis=1)
        df_il = df_il.reindex(sorted(df_il.columns), axis=1)
        df_il.to_csv(path / destination)


def merge_partition_json(path, partition_size_files, destination):
    """Merges partial partition size files to a single file"""
    merged_partition_sizes = {}
    if len(partition_size_files) > 0:
        for partition_size_file in partition_size_files:
            with open(path / partition_size_file) as partition_size_json:
                loaded_partition_sizes = json.load(partition_size_json)
                for partitioning_strategy in loaded_partition_sizes:
                    merged_partition_sizes[partitioning_strategy] = loaded_partition_sizes[partitioning_strategy]
    merged_partition_sizes = {k: v for k, v in sorted(merged_partition_sizes.items())}

    with open(path / destination, 'w') as partition_size_file:
        json.dump(merged_partition_sizes, partition_size_file, ensure_ascii=False)


def merge_experiment_results(path):
    raw_files = [os.path.basename(child) for child in path.iterdir()]

    relational_information_loss_files = [child for child in raw_files if child.startswith("relational_information_loss_") and any(f in child for f in information_loss_filter)]
    textual_information_loss_files = [child for child in raw_files if child.startswith("textual_information_loss_") and any(f in child for f in information_loss_filter)]
    total_information_loss_files = [child for child in raw_files if child.startswith("total_information_loss_") and any(f in child for f in information_loss_filter)]
    partition_sizes_files = [child for child in raw_files if child.startswith("partition_distribution_")]
    partition_split_files = [child for child in raw_files if child.startswith("partition_splits_")]

    merge_information_loss_results(path, relational_information_loss_files, "relational_information_loss.csv")
    merge_information_loss_results(path, textual_information_loss_files, "textual_information_loss.csv")
    merge_information_loss_results(path, total_information_loss_files, "total_information_loss.csv")
    merge_partition_json(path, partition_sizes_files, "partition_distribution.json")
    merge_partition_json(path, partition_split_files, "partition_splits.json")


# Taken from https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color='#000000')


def main():
    raw_path = Path("experiment_results/raw")
    raw_directories = [child for child in raw_path.iterdir() if child.is_dir()]

    ps_c_blogs_fig, ps_c_blogs_axes = plt.subplots(nrows=7, ncols=2, sharex='col', sharey='row')
    ps_c_hotels_fig, ps_c_hotels_axes = plt.subplots(nrows=7, ncols=2, sharex='col', sharey='row')

    for exp_run, raw_directory in enumerate(raw_directories):

        print("Creating plots for {}".format(raw_directory))

        detailed_il_files = [child for child in raw_directory.iterdir() if os.path.basename(child).startswith("detailed_textual_information_loss")]

        # Merge results to single files per run
        merge_experiment_results(raw_directory)

        # Load result files
        with open(raw_directory / 'partition_distribution.json') as json_file:
            partition_sizes = json.load(json_file)

        with open(raw_directory / 'partition_splits.json') as json_file:
            partition_splits = json.load(json_file)

        total_information_loss = pd.read_csv(raw_directory / "total_information_loss.csv", index_col="k")
        relational_information_loss = pd.read_csv(raw_directory / "relational_information_loss.csv", index_col="k")
        textual_information_loss = pd.read_csv(raw_directory / "textual_information_loss.csv", index_col="k")

        # Read all available strategies from partition sizes
        il_strategies = total_information_loss.columns
        partionioning_strategies = [x for x in list(partition_sizes.keys())]

        # Define legend names
        il_legend_names = [x.replace("mondrian", "mon").replace("-", r", $\lambda=") for x in il_strategies]
        il_legend_names = [x + "$" if "mon" in x else x for x in il_legend_names]
        part_legend_names = [x.replace("mondrian", "mon").replace("-", r", $\lambda=") for x in partionioning_strategies]
        part_legend_names = [x + "$" if "mon" in x else x for x in part_legend_names]

        # Read values for k
        k_values = [int(k) for k in partition_sizes[partionioning_strategies[0]].keys()]

        result_directory = Path("experiment_results/results") / os.path.basename(raw_directory)
        result_directory.mkdir(parents=True, exist_ok=True)

        # Transform information loss straight to latex tables
        total_information_loss.to_latex(buf=result_directory / "total_information_loss.tex")
        relational_information_loss.to_latex(buf=result_directory / "relational_information_loss.tex")
        textual_information_loss.to_latex(buf=result_directory / "textual_information_loss.tex")

        # Calculate mean and std for partition sizes
        partioning_distribution_table = pd.DataFrame(columns=pd.MultiIndex.from_product([part_legend_names, ["count", "mean", "std"]]), index=k_values)

        for ii, strategy in enumerate(partition_sizes):
            values = partition_sizes[strategy]
            for k in values:
                partition_results = values[k]
                partioning_distribution_table.loc[int(k), (part_legend_names[ii], "count")] = len(partition_results)
                partioning_distribution_table.loc[int(k), (part_legend_names[ii], "mean")] = statistics.mean(partition_results)
                if len(partition_results) > 1:
                    partioning_distribution_table.loc[int(k), (part_legend_names[ii], "std")] = statistics.stdev(partition_results)
                else:
                    partioning_distribution_table.loc[int(k), (part_legend_names[ii], "std")] = 0

        partioning_distribution_table.to_csv(result_directory / 'partition_distribution.csv')
        partioning_distribution_table.transpose().to_latex(buf=result_directory / "partition_distribution.tex", float_format="{:0.2f}".format)

        # Calculate number of relational and textual splits
        splits_legend_names = [name for name in part_legend_names if "mon" in name]
        partition_splits_table = pd.DataFrame(columns=pd.MultiIndex.from_product([k_values, splits_legend_names]), index=["relational", "textual"])

        for ii, strategy in enumerate(partition_splits):
            values = partition_splits[strategy]
            for k in values:
                relational_splits = values[k]["relational"]
                textual_splits = values[k]["textual"]
                partition_splits_table.loc["relational", (int(k), splits_legend_names[ii])] = relational_splits
                partition_splits_table.loc["textual", (int(k), splits_legend_names[ii])] = textual_splits

        partition_splits_table.to_csv(result_directory / 'partition_splits.csv')

        # Plot partition distributions
        inverted_partitions = {}
        for strategy in partition_sizes:
            values = partition_sizes[strategy]
            for k in values:
                inverted_partitions.setdefault(k, {})[strategy] = values[k]

        for k in inverted_partitions:
            fig = plt.figure()
            fig.set_figheight(fig_height)
            fig.set_figwidth(fig_width)

            plt.boxplot([arr for arr in inverted_partitions[k].values()], positions=list(range(1, len(part_legend_names) + 1)), labels=part_legend_names, sym='', widths=0.6)

            plt.xlabel('partitioning strategy')
            plt.ylabel('partition size', rotation=90)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(result_directory / 'partition_distribution-k_{}.pgf'.format(k))
            fig.savefig(result_directory / 'partition_distribution-k_{}.pdf'.format(k), bbox_inches='tight')

        # Plots for number of splits per strategy for each value of k
        for jj, k in enumerate(k_values):
            ps_plot = partition_splits_table[k].transpose().plot(kind='bar', stacked=True)
            ps_plot.set_xlabel(r"$\lambda$")
            ps_plot.set_xticklabels([s.replace('mon, $\\lambda=', '')[:-1] for s in splits_legend_names], rotation=0)
            ps_plot.set_ylabel('number of splits')
            ps_plot.legend(["relational attribute", "textual attribute"], loc='lower center', ncol=2, fancybox=True, bbox_to_anchor=(0.5, -0.2))
            ps_fig = ps_plot.get_figure()
            ps_fig.set_figheight(fig_height)
            ps_fig.set_figwidth(fig_width)
            ps_fig.tight_layout()
            ps_fig.savefig(result_directory / 'partition_splits-k_{}.pgf'.format(k))
            ps_fig.savefig(result_directory / 'partition_splits-k_{}.pdf'.format(k), bbox_inches='tight')

            str_result_dir = str(result_directory)
            if "blog_authorship_corpus" in str_result_dir:
                if "all_entities" in str_result_dir:
                    ps_combined_plot = partition_splits_table[k].transpose().plot(ax=ps_c_blogs_axes[jj, 0], kind='bar', stacked=True, legend=False)
                else:
                    ps_combined_plot = partition_splits_table[k].transpose().plot(ax=ps_c_blogs_axes[jj, 1], kind='bar', stacked=True, legend=False)
            else:
                if "all_entities" in str_result_dir:
                    ps_combined_plot = partition_splits_table[k].transpose().plot(ax=ps_c_hotels_axes[jj, 0], kind='bar', stacked=True, legend=False)
                else:
                    ps_combined_plot = partition_splits_table[k].transpose().plot(ax=ps_c_hotels_axes[jj, 1], kind='bar', stacked=True, legend=False)

            ps_combined_plot.set_xlabel(r"$\lambda$")
            ps_combined_plot.set_xticklabels([s.replace('mon, $\\lambda=', '')[:-1] for s in splits_legend_names], rotation=0)

        # Plot detailed textual information loss
        attribute_details = {}
        for f in detailed_il_files:
            strategy = f.name.split("loss_")[1].split(".")[0].replace("_", ".")
            if 'gdf' not in strategy:
                strategy = r"mon, $\lambda={}$".format(strategy)
            detailed_xil = pd.read_csv(f, header=[0, 1], index_col=[0])
            for lvl in [0, 1]:
                detailed_xil.columns.set_levels(detailed_xil.columns.levels[lvl].str.replace("_", "\\_"), level=lvl, inplace=True)
            detailed_xil.sort_index(axis=1, inplace=True)

            for attr in detailed_xil.columns.get_level_values(0):
                attr_xil = detailed_xil[attr]
                attribute_details.setdefault(attr, {})[strategy] = attr_xil

        for attr in attribute_details:
            # Single heatmap plots
            for ii, key in enumerate(sorted(attribute_details[attr])):
                heatmap_fig = plt.figure()
                df = attribute_details[attr][key].drop("total", axis=1).dropna(axis=1)
                sns_plot = sns.heatmap(df, xticklabels=True, yticklabels=True, cbar=True, vmin=0.2, vmax=1)
                sns_plot.tick_params(left=False, labelbottom=False, bottom=False, top=False, labeltop=True)
                sns_plot.set_xticklabels(sns_plot.get_xticklabels(), va="bottom", rotation=90)
                heatmap_fig.set_figheight(0.75 * fig_height)
                heatmap_fig.set_figwidth(fig_width)
                heatmap_fig.tight_layout()
                if "mon" in key:
                    file_ext = key.replace('mon, $\\lambda=', '')[:-1].replace('.', "_")
                else:
                    file_ext = key
                heatmap_fig.savefig(result_directory / "heatmap_{}_{}.pdf".format(attr.replace("\\", ""), file_ext), bbox_inches='tight')
                heatmap_fig.savefig(result_directory / "heatmap_{}_{}.pgf".format(attr.replace("\\", ""), file_ext), bbox_inches='tight')

            # Combined heatmap plots
            combined_heatmap, ax = plt.subplots(ncols=3, nrows=4, sharey=True, sharex=True)
            cbar_ax = combined_heatmap.add_axes([.91, .25, .03, .4])
            for ii, key in enumerate(sorted(attribute_details[attr])):
                df = attribute_details[attr][key].drop("total", axis=1).dropna(axis=1)
                sub_ax = ax[ii // 3, ii % 3]
                sns_plot = sns.heatmap(df, xticklabels=True, yticklabels=True, ax=sub_ax, cbar=ii == 0, vmin=0.2, vmax=1, cbar_ax=None if ii else cbar_ax)
                sub_ax.set_title(key, y=-0.1)
                ltop = True if ii < 3 else False
                sub_ax.tick_params(left=False, labelbottom=False, bottom=False, top=False, labeltop=ltop)
                if ltop:
                    sub_ax.set_xticklabels(sns_plot.get_xticklabels(), va="bottom", rotation=90)
                if ii % 3 != 0:
                    sub_ax.set_ylabel('')
            combined_heatmap.set_figheight(2 * fig_height)
            combined_heatmap.set_figwidth(2 * fig_width)
            combined_heatmap.tight_layout(rect=[0, 0, .9, 1])
            combined_heatmap.savefig(result_directory / "heatmap_{}.pdf".format(attr.replace("\\", "")), bbox_inches='tight')
            combined_heatmap.savefig(result_directory / "heatmap_{}.pgf".format(attr.replace("\\", "")), bbox_inches='tight')

        # Plot for total information loss
        til_plot = total_information_loss.plot(xticks=k_values, marker='o', color=colors)
        til_plot.legend(il_legend_names, loc='lower center', ncol=3, fancybox=True, bbox_to_anchor=(0.5, -0.3))
        til_fig = til_plot.get_figure()
        til_fig.set_figheight(fig_height)
        til_fig.set_figwidth(fig_width)
        til_plot.set_xlabel('k')
        til_plot.set_ylabel('NCP', rotation=90)
        til_plot.set_ylim([y_min, y_max])
        til_plot.set_yticks(y_ticks)
        til_fig.tight_layout()
        til_fig.savefig(result_directory / 'total_information_loss.pgf')
        til_fig.savefig(result_directory / 'total_information_loss.pdf', bbox_inches='tight')

        # Plot for relational information loss
        ril_plot = relational_information_loss.plot(xticks=k_values, marker='o', color=colors)
        ril_plot.legend(il_legend_names, loc='lower center', ncol=3, fancybox=True, bbox_to_anchor=(0.5, -0.3))
        ril_fig = ril_plot.get_figure()
        ril_fig.set_figheight(fig_height)
        ril_fig.set_figwidth(fig_width)
        ril_plot.set_xlabel('k')
        ril_plot.set_ylabel('$NCP_A$', rotation=90)
        ril_plot.set_ylim([y_min, y_max])
        ril_plot.set_yticks(y_ticks)
        ril_fig.tight_layout()
        ril_fig.savefig(result_directory / 'relational_information_loss.pgf')
        ril_fig.savefig(result_directory / 'relational_information_loss.pdf', bbox_inches='tight')

        # Plot for textual information loss
        xil_plot = textual_information_loss.plot(xticks=k_values, marker='o', color=colors)
        xil_plot.legend(il_legend_names, loc='lower center', ncol=3, fancybox=True, bbox_to_anchor=(0.5, -0.3))
        xil_fig = xil_plot.get_figure()
        xil_fig.set_figheight(fig_height)
        xil_fig.set_figwidth(fig_width)
        xil_plot.set_xlabel('k')
        xil_plot.set_ylabel('$NCP_X$', rotation=90)
        xil_plot.set_ylim([y_min, y_max])
        xil_plot.set_yticks(y_ticks)
        xil_fig.tight_layout()
        xil_fig.savefig(result_directory / 'textual_information_loss.pgf')
        xil_fig.savefig(result_directory / 'textual_information_loss.pdf', bbox_inches='tight')

        xil_zoomed = textual_information_loss.plot(xticks=k_values, marker='o', color=colors)
        xil_zoomed.legend(il_legend_names, loc='lower center', ncol=3, fancybox=True, bbox_to_anchor=(0.5, -0.3))
        xil_zoomed_fig = xil_zoomed.get_figure()
        xil_zoomed_fig.set_figheight(fig_height)
        xil_zoomed_fig.set_figwidth(fig_width)
        xil_zoomed.set_xlabel('k')
        xil_zoomed.set_ylabel('$NCP_X$', rotation=90)
        xil_zoomed_fig.tight_layout()
        xil_zoomed_fig.savefig(result_directory / 'textual_information_loss_zoomed.pgf')
        xil_zoomed_fig.savefig(result_directory / 'textual_information_loss_zoomed.pdf', bbox_inches='tight')

    # Combined plot
    combined_results = ["blogs", "hotels"]
    pad = 5  # in points
    for ii, (ps_c_fig, ps_c_axes) in enumerate([(ps_c_blogs_fig, ps_c_blogs_axes), (ps_c_hotels_fig, ps_c_hotels_axes)]):
        handles_labels = [ax.get_legend_handles_labels() for ax in ps_c_fig.axes]
        handles, labels = [sum(lol, []) for lol in zip(*handles_labels)]
        ps_c_fig.legend(handles, ["relational attribute", "textual attribute"], loc='lower center', ncol=2, fancybox=True, bbox_to_anchor=(0.53, -0.03))
        ps_c_fig.set_figheight(2 * fig_height)
        ps_c_fig.set_figwidth(2 * fig_width)

        cols = ["all entities", "only GPE"]
        rows = ["$k={}$".format(k) for k in k_values]

        for ax, col in zip(ps_c_axes[0], cols):
            ax.set_title(col)

        for ax, row in zip(ps_c_axes[:, 0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)

        for ax, row in zip(ps_c_axes[:, 0], rows):
            ax.set_ylabel("number of splits", rotation=90)

        ps_c_fig.subplots_adjust(left=0.15, top=0.95)
        ps_c_fig.tight_layout()
        results_path = Path("experiment_results/results")
        ps_c_fig.savefig(results_path / 'partition_splits_combined_{}.pgf'.format(combined_results[ii]), bbox_inches='tight')
        ps_c_fig.savefig(results_path / 'partition_splits_combined_{}.pdf'.format(combined_results[ii]), bbox_inches='tight')


if __name__ == "__main__":
    main()
