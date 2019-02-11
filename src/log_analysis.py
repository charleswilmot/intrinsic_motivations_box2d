import tensorflow as tf
import numpy as np
import os
import shutil
import glob
import re
import pylatex as tex
import argparse
import matplotlib.pyplot as plt
import pickle
import json


def read_event_file(path):
    return [x for x in tf.train.summary_iterator(path)]


def to_x_y_arrays(list_of_tuples):
    data = np.array(list_of_tuples).T
    steps = data[0]
    args = np.argsort(steps)
    data = data[:, args]
    return data[0], data[1]


def group_by_tag(summaries):
    # return a dict of dict of numpy arrays
    ret = {}
    for summary in summaries:
        step = summary.step
        for datapoint in summary.summary.value:
            tag_01 = datapoint.tag
            tag_0, tag_1 = tag_01.split("/")
            if tag_0 not in ret:
                ret[tag_0] = {}
            if tag_1 not in ret[tag_0]:
                ret[tag_0][tag_1] = []
            ret[tag_0][tag_1].append((step, datapoint.simple_value))
    for tag_0 in ret:
        for tag_1 in ret[tag_0]:
            ret[tag_0][tag_1] = to_x_y_arrays(ret[tag_0][tag_1])
    return ret


def read_data(path_to_log):
    # return all data from one experiment groupde by tag
    summaries = []
    workers_paths = [path_to_log + "/" + x for x in os.listdir(path_to_log) if re.match("worker[0-9]+", x) is not None]
    for path_to_worker in workers_paths:
        events_paths = [path_to_worker + "/" + x for x in os.listdir(path_to_worker)]
        for path_to_events in events_paths:
            summaries += read_event_file(path_to_events)
    return group_by_tag(summaries)


def weighted_stats(data, weights):
    average = np.average(data, weights=weights)
    variance = np.average((data - average)**2, weights=weights)
    return average, np.sqrt(variance)


def smoothen(data, smooth_param):
    mini = int(np.min(data[0]))
    maxi = int(np.max(data[0]))
    x = np.arange(mini, maxi)
    length = maxi - mini

    def get_weights(data, index):
        return np.exp(- (data[0] - index - mini) ** 2 / smooth_param ** 2)

    mean = np.zeros(length)
    std = np.zeros(length)
    for i in range(length):
        m, s = weighted_stats(data[1], get_weights(data, i))
        mean[i] = m
        std[i] = s
    return x, mean, std


def plot_stats(ax, x, mean, std):
    p0, p1 = np.percentile(mean - std, 5), np.percentile(mean + std, 95)
    off = 0.5 * (p1 - p0)
    ax.fill_between(x, mean - std, mean + std, color="b", alpha=0.3)
    ax.plot(x, mean, color="b")
    ax.set_ylim([p0 - off, p1 + off])


# def plot_stage(outpath, stage_data, smooth_param=50):
#     if not os.path.exists(outpath):
#         os.mkdir(outpath)
#     filepath = outpath + "/plot.png"
#     if not os.path.exists(filepath):
#         f = plt.figure(figsize=(16, 4), dpi=200)
#         f.subplots_adjust(hspace=0.8, wspace=0.8)
#         for i, key in enumerate(stage_data):
#             if not os.path.exists(filepath):
#                 smooth_data = smoothen(stage_data[key], smooth_param)
#                 ax = f.add_subplot(2, 4, i + 1)
#                 plot_stats(ax, *smooth_data)
#                 ax.set_title(key.replace("_", " "))
#         f.savefig(filepath, bbox_inches='tight')
#         plt.close(f)


# def plot_all_discount_factors(outpath, tuple_to_data, exp_type, keys, lookback=0.01):
#     sorted_df = sorted(list(set([df for df, t in tuple_to_data if t == exp_type])))
#     stages = tuple_to_data[(sorted_df[0], exp_type)].keys()
#     f = plt.figure(figsize=(16, 4 * len(keys)), dpi=200)
#     f.subplots_adjust(hspace=0.4, wspace=0.4)
#     for jj, key in enumerate(keys):
#         sub_data = [[tuple_to_data[(df, exp_type)][stage][key][1] for df in sorted_df] for stage in stages]
#         a = sub_data[0][0]
#         s = a[-int(len(a) * lookback):]
#         means = []
#         for i in range(len(sub_data)):
#             means.append([])
#             for j in range(len(sub_data[i])):
#                 a = sub_data[i][j]
#                 means[i].append(np.mean(a[-int(len(a) * lookback):]))
#         ax = f.add_subplot(len(keys), 1, jj + 1)
#         index = np.arange(len(sorted_df))
#         bar_width = 0.9 / len(sorted_df)
#         for i, (d, opacity) in enumerate(zip(means, np.linspace(0.5, 1, len(sorted_df)))):
#             ax.bar(index + i * bar_width, d, bar_width, alpha=opacity, color='b')
#         all_means = [item for sublist in means for item in sublist]
#         ylim = ax.get_ylim()
#         if np.isnan(all_means).any():
#             ylim = [0, np.nanmax(all_means)]
#         if ylim[1] > 100:
#             ylim = [0, np.nanmedian(all_means) * 3]
#         if ylim[0] < -100:
#             ylim = [np.nanmedian(all_means) * 3, 0]
#         print("!!!!", ylim)
#         ax.set_ylim(ylim)
#         ax.set_title(key.replace("_", " "))
#         ax.set_xlabel('Discount factors')
#         ax.set_ylabel(key.replace("_", " "))
#         ax.set_xticks(index + bar_width / 2)
#         ax.set_xticklabels(sorted_df)
#     t = exp_type
#     exp_name = "minimize" if t == -2 else "maximize" if t == -1 else "target_{:04d}".format(int(t * 1000))
#     filepath = outpath + "/" + exp_name + ".png"
#     f.savefig(filepath, bbox_inches='tight')
#     plt.close(f)


# def stage_has_diverged(stage_data):
#     for key in stage_data:
#         if np.isnan(stage_data[key]).any():
#             return True
#     return False


# def get_divergent_stage(data):
#     has_diverged = {}
#     for key in data:
#         stage_number = int(re.match("stage_([0-9]+)", key).group(1))
#         has_diverged[stage_number] = stage_has_diverged(data[key])
#     for stage_number in sorted(has_diverged):
#         if has_diverged[stage_number]:
#             return stage_number
#     return -1


# def path_to_tuple(path):
#     m = re.match(".*\/df_([0-9]+)\/(minimize|maximize|target_([0-9]+))", path)
#     df = float(m.group(1)) / 100
#     if m.group(2) == "minimize":
#         t = -2
#     elif m.group(2) == "maximize":
#         t = -1
#     else:
#         t = float(m.group(3)) / 1000
#     return (df, t)


# def make_convergence_tabular(doc, tuple_to_convergence, targets):
#     target_to_df = {t: sorted([df for df, tt in tuple_to_convergence if tt == t]) for t in targets}
#     sorted_df = sorted(list(set(sum([target_to_df[k] for k in targets], []))))
#     with doc.create(tex.Tabular('l|' + 'c' * len(sorted_df))) as table:
#         table.add_row(["Discount factor"] + sorted_df)
#         table.add_hline()
#         for t in sorted(targets):
#             line_name = "Minimize" if t == -2 else "Maximize" if t == -1 else "Target {}".format(t)
#             table.add_row([line_name] + [tuple_to_convergence[(df, t)] if df in target_to_df[t] else "" for df in sorted_df])


# def include_all_plots(doc, tuple_to_path, section_name, exp_type):
#     with doc.create(tex.Section(section_name)):
#         for df in sorted(list(set([df for df, t in tuple_to_path if t == exp_type]))):
#             with doc.create(tex.Subsection('Discount factor {}'.format(df))):
#                 plots_path = tuple_to_path[(df, exp_type)] + "plots/"
#                 stages = sorted(os.listdir(plots_path), key=lambda x: int(re.match("stage_([0-9]+)", x).group(1)))
#                 for stage in stages:
#                     for plot in sorted(os.listdir(plots_path + stage)):
#                         with doc.create(tex.Figure(position='!ht')) as plot_figure:
#                             image_path = plots_path + stage + "/" + plot
#                             print(image_path)
#                             plot_figure.add_image(image_path, width=tex.utils.NoEscape('\linewidth'))
#                             plot_figure.add_caption(stage)
#                 doc.append(tex.Command("clearpage"))


# def generate_latex(path_to_array, regenerate_plots=False, reload_data=False):
#     path_to_array = os.path.abspath(path_to_array)
#     experiments_paths = [x for x in glob.glob(path_to_array + "/*/*/")
#                          if re.match(".*\/df_([0-9]+)\/(minimize|maximize|target_([0-9]+))", x)]
#     experiments_paths.sort(key=path_to_tuple)
#     tuples = [path_to_tuple(path) for path in experiments_paths]
#     tuple_to_path = {tup: path for tup, path in zip(tuples, experiments_paths)}
#     df_values = [df for df, t in tuples]
#     targets = [t for df, t in tuples]
#     print("Loading data ... ")
#     if not os.path.exists(path_to_array + "/data/"):
#         os.mkdir(path_to_array + "/data/")
#     if reload_data or (not os.path.exists(path_to_array + "/data/tuple_to_array.pkl")):
#         tuple_to_data = {tup: read_data(tuple_to_path[tup] + "/log/") for tup in tuples}
#         with open(path_to_array + "/data/tuple_to_array.pkl", "wb") as f:
#             pickle.dump(tuple_to_data, f)
#     else:
#         if not os.path.exists(path_to_array + "/data/tuple_to_array.pkl"):
#             os.mkdir(path_to_array + "/data/tuple_to_array.pkl")
#         with open(path_to_array + "/data/tuple_to_array.pkl", "rb") as f:
#             tuple_to_data = pickle.load(f)
#     print("Done")
#     tuple_to_convergence = {tup: get_divergent_stage(tuple_to_data[tup]) for tup in tuples}
#     ### PLOTS ###
#     plots_path = path_to_array + "/plots_overview/"
#     if regenerate_plots and os.path.exists(plots_path) and os.path.isdir(plots_path):
#         shutil.rmtree(plots_path)
#     if not os.path.exists(plots_path):
#         os.mkdir(plots_path)
#     keys = ["actor_stddev", "model_loss_at_rl_time"]
#     for exp_type in sorted(list(set(targets))):
#         plot_all_discount_factors(plots_path, tuple_to_data, exp_type, keys)
#     for tup in tuples:
#         data = tuple_to_data[tup]
#         path = tuple_to_path[tup]
#         print(path)
#         plots_path = path + "/plots/"
#         if regenerate_plots and os.path.exists(plots_path) and os.path.isdir(plots_path):
#             shutil.rmtree(plots_path)
#         if not os.path.exists(plots_path):
#             os.mkdir(plots_path)
#         for stage_name in data:
#             stage_path = plots_path + stage_name
#             plot_stage(stage_path, data[stage_name])
#     ### LATEX ###
#     geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
#     doc = tex.Document(geometry_options=geometry_options)
#     doc.append(tex.Command("tableofcontents"))
#     doc.append(tex.Command("clearpage"))
#     with doc.create(tex.Section('Description')):
#         description_path = path_to_array + "/description.tex"
#         if os.path.exists(description_path):
#             doc.append(tex.Command("input", arguments=description_path))
#         else:
#             doc.append('please create a description tex file at {}'.format(description_path))
#     with doc.create(tex.Section('Convergence overview')):
#         with doc.create(tex.Subsection('Minimize prediction error')):
#             make_convergence_tabular(doc, tuple_to_convergence, [-2])
#         with doc.create(tex.Subsection('Maximize prediction error')):
#             make_convergence_tabular(doc, tuple_to_convergence, [-1])
#         with doc.create(tex.Subsection('Target prediction error')):
#             make_convergence_tabular(doc, tuple_to_convergence, sorted(list(set([t for t in targets if t >= 0]))))
#     with doc.create(tex.Section('Analysis wrt discount factor / stage')):
#         with doc.create(tex.Subsection('Minimize prediction error')):
#             with doc.create(tex.Figure(position='!ht')) as plot_figure:
#                 image_path = path_to_array + "/plots_overview/minimize.png"
#                 print(image_path)
#                 plot_figure.add_image(image_path, width=tex.utils.NoEscape('\linewidth'))
#         doc.append(tex.Command("clearpage"))
#         with doc.create(tex.Subsection('Maximize prediction error')):
#             with doc.create(tex.Figure(position='!ht')) as plot_figure:
#                 image_path = path_to_array + "/plots_overview/maximize.png"
#                 print(image_path)
#                 plot_figure.add_image(image_path, width=tex.utils.NoEscape('\linewidth'))
#         doc.append(tex.Command("clearpage"))
#         with doc.create(tex.Subsection('Target prediction error')):
#             for target in sorted(list(set([t for t in targets if t >= 0]))):
#                 with doc.create(tex.Figure(position='!ht')) as plot_figure:
#                     image_path = path_to_array + "/plots_overview/target_{:04d}.png".format(int(target * 1000))
#                     print(image_path)
#                     plot_figure.add_image(image_path, width=tex.utils.NoEscape('\linewidth'))
#                     plot_figure.add_caption("Target = {}".format(target))
#         doc.append(tex.Command("clearpage"))
#     include_all_plots(doc, tuple_to_path, 'Minimize prediction error', -2)
#     include_all_plots(doc, tuple_to_path, 'Maximize prediction error', -1)
#     for target in sorted(list(set([t for t in targets if t >= 0]))):
#         include_all_plots(doc, tuple_to_path, 'Target prediction error {}'.format(target), target)
#     doc.generate_pdf(filepath=path_to_array + "/test", clean_tex=False)


def get_all_paths(path_to_exps):
    paths = [x for x in glob.glob(path_to_exps) if os.path.exists(x + "/log")]
    paths.sort()
    return paths


def load_data(path_to_exps=None, path_to_data_in=None, path_to_data_out=None):
    if path_to_exps is None and path_to_data_in is None:
        raise AttributeError("Must specify a path to data")
    if path_to_data_in:
        with open(path_to_data_in, "rb") as f:
            data = pickle.load(f)
    elif path_to_exps:
        print("Reading data from log files:")
        paths = get_all_paths(path_to_exps)
        data = {}
        for i, p in enumerate(paths):
            print("{} / {}\t\t{}".format(i, len(paths), p))
            data[p] = read_data(p + "/log/")
    if path_to_data_out:
        with open(path_to_data_out, "wb") as f:
            pickle.dump(data, f)
    return data


def load_parameters(path_to_exps):
    paths = get_all_paths(path_to_exps)
    parameters = {}
    for p in paths:
        with open(p + "/parameters.json", "r") as f:
            parameters[p] = json.load(f)
    return parameters


def group_parameters_by(parameters, key):
    groups = {}
    for path in sorted(parameters):
        current_key = key(parameters[path])
        if current_key in groups:
            groups[current_key][path] = parameters[path]
        else:
            groups[current_key] = {path: parameters[path]}
    return groups


def generate_all_plots(data, regenerate=False):
    print("Generating experiment plots:")
    fig = plt.figure(figsize=(16, 4), dpi=200)
    fig.subplots_adjust(hspace=0.8, wspace=0.8)
    for i, path in enumerate(sorted(data)):
        print("{} / {}\t\t{}".format(i, len(data), path))
        generate_experiment_plots(fig, path + "/plots/", data[path], regenerate=regenerate)
    plt.close(fig)


def generate_experiment_plots(fig, path, data, regenerate=False):
    if regenerate and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)
    for key in sorted(data):
        print("\t\t{}".format(key))
        filepath = path + "/{}.png".format(key)
        generate_sub_experiment_plot(fig, filepath, data[key], regenerate=regenerate)


def generate_sub_experiment_plot(fig, filepath, data, regenerate=False, smooth_param=50):
    if not os.path.exists(filepath):
        for i, key in enumerate(sorted(data)):
            smooth_data = smoothen(data[key], smooth_param)
            ax = fig.add_subplot(2, 4, i + 1)
            plot_stats(ax, *smooth_data)
            ax.set_title(key.replace("_", " "))
        fig.savefig(filepath, bbox_inches='tight')
        fig.clear()


class TexFile(tex.base_classes.Container):
    def dumps(self):
        return self.dumps_content()

    def generate_latex(self, path):
        with open(path, "w") as f:
            f.write(self.dumps())


def generate_all_experiment_tex_files(data):
    print("Generating experiments LaTeX files:")
    for i, path in enumerate(sorted(data)):
        print("{} / {}\t\t{}".format(i, len(data), path))
        texfile = TexFile()
        for key in sorted(data[path]):
            with texfile.create(tex.Figure(position='!ht')) as plot_figure:
                image_path = os.path.abspath(path + "plots/{}.png".format(key))
                plot_figure.add_image(image_path, width=tex.utils.NoEscape('\linewidth'))
                plot_figure.add_caption(key.replace("_", " "))
        texfile.generate_latex(path + "/experiment.tex")


class ArraySummary(tex.Document):
    def __init__(self, title):
        super().__init__()
        self.preamble.append(tex.Package('graphicx'))
        self.preamble.append(tex.Command('title', title))
        self.preamble.append(tex.Command('author', 'Charles Wilmot'))
        self.preamble.append(tex.Command('date', tex.utils.NoEscape(r'\today')))
        self.append(tex.utils.NoEscape(r'\maketitle'))
        self.clearpage()
        self.append(tex.Command("tableofcontents"))
        self.clearpage()

    def add_experiment(self, path):
        self.append(tex.Command("input", tex.NoEscape(os.path.abspath(path + "experiment.tex"))))

    def add_tabular(self, array):
        with doc.create(tex.Tabular('l|' + 'c' * (len(array[0]) - 1))) as table:
            for i, line in enumerate(array):
                table.add_row(line)
                # table.add_row([str(x) for x in line])
                if i == 0:
                    table.add_hline()

    def clearpage(self):
        self.append(tex.Command("clearpage"))


# generate_latex("../experiments/array_no_entropy_reg", regenerate_plots=False)
path_to_array = "../experiments/array_df_085_continous_vs_stages/"
data = load_data(path_to_data_in=path_to_array + "data.pkl")
# data = load_data(path_to_exps=path_to_array + "*/",
#                  path_to_data_out=path_to_array + "data.pkl")
generate_all_plots(data)
generate_all_experiment_tex_files(data)
parameters = load_parameters(path_to_array + "*/")


def by_reward_type(d):
    if not d['minimize_pred_err'] is None:
        return 'minimize'
    if not d['maximize_pred_err'] is None:
        return 'maximize'
    if not d['target_pred_err'] is None:
        return 'target'
    if not d['range_pred_err'] is None:
        return 'range'


parameters_by_reward_type = group_parameters_by(parameters, by_reward_type)


arrsum = ArraySummary("Continuous vs Stage-wise")
for reward_type in ["minimize", "maximize", "target", "range"]:
    with arrsum.create(tex.Section(reward_type)):
        groups = group_parameters_by(parameters_by_reward_type[reward_type], lambda x: "stages" if x["continuous"] is None else "continuous")
        with arrsum.create(tex.Subsection("Stages")):
            for p in groups["stages"]:
                arrsum.add_experiment(p)
        with arrsum.create(tex.Subsection("Continuous")):
            for p in groups["continuous"]:
                arrsum.add_experiment(p)
    arrsum.clearpage()
arrsum.generate_pdf(filepath=path_to_array + "new_impl", clean_tex=False)
