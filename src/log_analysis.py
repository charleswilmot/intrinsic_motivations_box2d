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

    def get_weights(X, index):
        return np.exp(- (X - index - mini) ** 2 / smooth_param ** 2)

    mean = np.zeros(length)
    std = np.zeros(length)
    for i in range(length):
        where = np.where(np.logical_and(data[0] < i + mini + 2 * smooth_param, i + mini - 2 * smooth_param < data[0]))
        X = data[0][where]
        Y = data[1][where]
        m, s = weighted_stats(Y, get_weights(X, i))
        mean[i] = m
        std[i] = s
    return x, mean, std


def plot_stats(ax, x, mean, std):
    p0, p1 = np.percentile(mean - std, 5), np.percentile(mean + std, 95)
    off = 0.5 * (p1 - p0)
    ax.fill_between(x, mean - std, mean + std, color="b", alpha=0.3)
    ax.plot(x, mean, color="b")
    ax.set_ylim([p0 - off, p1 + off])


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


def generate_all_plots(data, regenerate=False, keys=None):
    print("Generating experiment plots:")
    fig = plt.figure(figsize=(20, 2), dpi=200)
    fig.subplots_adjust(hspace=0.8, wspace=0.8)
    for i, path in enumerate(sorted(data)):
        print("{} / {}\t\t{}".format(i, len(data), path))
        generate_experiment_plots(fig, path + "/plots/", data[path], regenerate=regenerate, keys=keys)
    plt.close(fig)


def generate_experiment_plots(fig, path, data, regenerate=False, keys=None):
    if regenerate and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)
    for key in sorted(data):
        if keys is None or key in keys:
            print("\t\t{}".format(key))
            filepath = path + "/{}.png".format(key)
            generate_sub_experiment_plot(fig, filepath, data[key], regenerate=regenerate)


def generate_sub_experiment_plot(fig, filepath, data, regenerate=False, smooth_param=50):
    if not os.path.exists(filepath):
        for i, key in enumerate(sorted(data)):
            print("\t\t\t{}".format(key))
            smooth_data = smoothen(data[key], smooth_param)
            ax = fig.add_subplot(1, len(data), i + 1)
            plot_stats(ax, *smooth_data)
            ax.set_title(key.replace("_", " "), fontsize=10)
        fig.savefig(filepath) #, bbox_inches='tight')
        fig.clear()


class TexFile(tex.base_classes.Container):
    def dumps(self):
        return self.dumps_content()

    def generate_latex(self, path):
        with open(path, "w") as f:
            f.write(self.dumps())


def generate_all_experiment_tex_files(data, keys=None):
    print("Generating experiments LaTeX files:")
    for i, path in enumerate(sorted(data)):
        print("{} / {}\t\t{}".format(i, len(data), path))
        texfile = TexFile()
        for key in sorted(data[path]):
            if keys is None or key in keys:
                with texfile.create(tex.Figure(position='!ht')) as plot_figure:
                    image_path = os.path.abspath(path + "plots/{}.png".format(key))
                    plot_figure.add_image(image_path, width=tex.utils.NoEscape('\linewidth'))
                    plot_figure.add_caption(key.replace("_", " ") + " {}".format(os.path.basename(path)))
        texfile.generate_latex(path + "/experiment.tex")


def get_tab_data(data, params, by1, by2, func, name=""):
    val_dict = {}
    for k in params:
        by1_val = by1(params[k])
        by2_val = by2(params[k])
        if by1_val not in val_dict:
            val_dict[by1_val] = {}
        if by2_val not in val_dict[by1_val]:
            val_dict[by1_val][by2_val] = func(data[k])
        else:
            raise ValueError("{}\t{}\t>>\t{}    already found".format(k, by1_val, by2_val))
    ret = []
    ret.append([name] + [str(x) for x in list(sorted(val_dict))])
    dummy_key = list(val_dict)[0]
    for k in sorted(val_dict[dummy_key]):
        ret.append([str(k)] + [str(x) for x in [val_dict[kk][k] for kk in sorted(val_dict)]])
    return ret


def by_reward(x):
    if x["minimize_pred_err"]:
        return "minimize {}".format(x["minimize_pred_err"])
    if x["maximize_pred_err"]:
        return "maximize {}".format(x["maximize_pred_err"])
    if x["target_pred_err"]:
        return "target {}".format(x["target_pred_err"])
    if x["range_pred_err"]:
        return "range {}".format(x["range_pred_err"])


def by_exp_type(x):
    continuous = "continuous" if x["continuous"] else "stagewise"
    separate = "separate" if x["separate"] else "joint"
    return "{} {}".format(continuous, separate)


def get_std(data):
    s = 0
    for j in ["arm1_arm2_left", "arm1_arm2_right", "ground_arm1_left", "ground_arm1_right"]:
        s += np.mean(data["joints"][j + "_std"][1][-100:])
    return "{:.4f}".format(s / 4)


class ArraySummary(tex.Document):
    def __init__(self, title):
        super().__init__(geometry_options={
            "tmargin": "1cm",
            "lmargin": "1cm",
            "rmargin": "1cm",
            "bmargin": "2cm"})
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
        # \textcolor{\color{red!20!green!30}}{<text>}
        with self.create(tex.Tabular('l|' + 'c' * (len(array[0]) - 1))) as table:
            for i, line in enumerate(array):
                table.add_row(line)
                # table.add_row([str(x) for x in line])
                if i == 0:
                    table.add_hline()
        self.append(tex.utils.NoEscape("\\\\~\\\\"))

    def clearpage(self):
        self.append(tex.Command("clearpage"))


path_to_array = "../experiments/discrete_a3c/test_experiments_factor_1_alr_clr_5e-4_softmax_plus_mini_20/"
# data = load_data(path_to_data_in=path_to_array + "data.pkl")
data = load_data(path_to_exps=path_to_array + "*/",
                 path_to_data_out=path_to_array + "data.pkl")
generate_all_plots(data, regenerate=False, keys=["rl"])
generate_all_experiment_tex_files(data, keys=["rl"])
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


def by_reward_type_and_param(d):
    if not d['minimize_pred_err'] is None:
        return 'minimize {}'.format(d['minimize_pred_err'])
    if not d['maximize_pred_err'] is None:
        return 'maximize {}'.format(d['maximize_pred_err'])
    if not d['target_pred_err'] is None:
        return 'target {}'.format(d['target_pred_err'])
    if not d['range_pred_err'] is None:
        return 'range {} {}'.format(d['range_pred_err'])


parameters_by_reward_type = group_parameters_by(parameters, by_reward_type)


def by_clr(x):
    return "{}".format(np.log10(x["critic_lr"]))


def by_alr(x):
    return "{}".format(np.log10(x["actor_lr"]))


def by_mlr(x):
    return "{}".format(np.log10(x["model_lr"]))


def by_discount_factor(x):
    return "{}".format(x["discount_factor"])


def by_buffer_size(x):
    return int(x["model_buffer_size"])


def by_nothing(x):
    return " "


LOOKBACK_START = -200
LOOKBACK_END = -1


def get_mean_end_reward(data):
    return "{:.3f}".format(np.mean(data["rl"]["reward"][1][LOOKBACK_START:LOOKBACK_END]))


def get_std_end_reward(data):
    return "{:.3f}".format(np.std(data["rl"]["reward"][1][LOOKBACK_START:LOOKBACK_END]))


def get_mean_end_loss_at_rl(data):
    return "{:.3f}".format(np.mean(data["rl"]["loss"][1][LOOKBACK_START:LOOKBACK_END]))


def get_mean_end_loss_at_model(data):
    return "{:.3f}".format(np.mean(data["model"]["loss"][1][LOOKBACK_START:LOOKBACK_END]))


def get_std_end_loss(data):
    return "{:.3f}".format(np.std(data["rl"]["loss"][1][LOOKBACK_START:LOOKBACK_END]))


def get_mean_critic_q(data):
    return "{:.3f}".format(np.mean(data["rl"]["critic_quality"][1][LOOKBACK_START:LOOKBACK_END]))


def continuous_stagewise_separate():
    arrsum = ArraySummary("Continuous vs Stage-wise vs Separate")
    with arrsum.create(tex.Section("Overview")):
        arrsum.add_tabular(get_tab_data(data, parameters, by_exp_type, by_reward, get_std, name="STD"))
    for reward_type in ["minimize", "maximize", "target", "range"]:
        grouped_by_reward_params = group_parameters_by(parameters_by_reward_type[reward_type], lambda x: tuple(x[reward_type + "_pred_err"]) if type(x[reward_type + "_pred_err"]) == list else x[reward_type + "_pred_err"])
        for k in grouped_by_reward_params:
            with arrsum.create(tex.Section(reward_type + str(k))):
                groups = group_parameters_by(grouped_by_reward_params[k], lambda x: "stages" if x["continuous"] is None else "continuous")
                with arrsum.create(tex.Subsection("Stages")):
                    sub_groups = group_parameters_by(groups["stages"], lambda x: "separate" if x["separate"] else "joint")
                    with arrsum.create(tex.Subsubsection("Separate")):
                        for p in sub_groups["separate"]:
                            arrsum.add_experiment(p)
                    with arrsum.create(tex.Subsubsection("Joint")):
                        for p in sub_groups["joint"]:
                            arrsum.add_experiment(p)
                arrsum.clearpage()
                with arrsum.create(tex.Subsection("Continuous")):
                    sub_groups = group_parameters_by(groups["continuous"], lambda x: "separate" if x["separate"] else "joint")
                    with arrsum.create(tex.Subsubsection("Separate")):
                        for p in sub_groups["separate"]:
                            arrsum.add_experiment(p)
                    with arrsum.create(tex.Subsubsection("Joint")):
                        for p in sub_groups["joint"]:
                            arrsum.add_experiment(p)
                arrsum.clearpage()
    arrsum.generate_pdf(filepath=path_to_array + "new_impl", clean_tex=False)


def learning_rates():
    arrsum = ArraySummary("Find best learning rate")
    with arrsum.create(tex.Section("Overview")):
        arrsum.add_tabular(get_tab_data(data, parameters, by_clr, by_alr, get_mean_critic_q, name="critic quality"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_clr, by_alr, get_mean_end_reward, name="mean reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_clr, by_alr, get_std_end_reward, name="std reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_clr, by_alr, get_mean_end_loss_at_rl, name="mean loss"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_clr, by_alr, get_std_end_loss, name="std loss"))
    grouped_by_clr = group_parameters_by(parameters, by_clr)
    for clr in sorted(grouped_by_clr):
        grouped_by_alr = group_parameters_by(grouped_by_clr[clr], by_alr)
        for alr in sorted(grouped_by_alr):
            with arrsum.create(tex.Section("Critic {}    Actor {}".format(clr, alr))):
                for p in grouped_by_alr[alr]:
                    arrsum.add_experiment(p)
                arrsum.clearpage()
    arrsum.generate_pdf(filepath=path_to_array + "new_impl", clean_tex=False)


def model_learning_rates():
    arrsum = ArraySummary("Find best learning rate")
    with arrsum.create(tex.Section("Overview")):
        arrsum.add_tabular(get_tab_data(data, parameters, by_mlr, by_nothing, get_mean_critic_q, name="critic quality"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_mlr, by_nothing, get_mean_end_reward, name="mean reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_mlr, by_nothing, get_std_end_reward, name="std reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_mlr, by_nothing, get_mean_end_loss_at_rl, name="mean loss"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_mlr, by_nothing, get_std_end_loss, name="std loss"))
    grouped_by_mlr = group_parameters_by(parameters, by_mlr)
    for mlr in sorted(grouped_by_mlr):
        with arrsum.create(tex.Section("Model learning rate  {}".format(mlr))):
            for p in grouped_by_mlr[mlr]:
                arrsum.add_experiment(p)
            arrsum.clearpage()
    arrsum.generate_pdf(filepath=path_to_array + "new_impl", clean_tex=False)


def model_target():
    arrsum = ArraySummary("Find best learning rate")
    with arrsum.create(tex.Section("Overview")):
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_reward_type_and_param, get_mean_critic_q, name="critic quality"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_reward_type_and_param, get_mean_end_reward, name="mean reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_reward_type_and_param, get_std_end_reward, name="std reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_reward_type_and_param, get_mean_end_loss_at_rl, name="mean loss"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_reward_type_and_param, get_std_end_loss, name="std loss"))
    grouped_by_target = group_parameters_by(parameters, by_reward_type_and_param)
    for target in sorted(grouped_by_target):
        with arrsum.create(tex.Section("{}".format(target))):
            for p in grouped_by_target[target]:
                arrsum.add_experiment(p)
            arrsum.clearpage()
    arrsum.generate_pdf(filepath=path_to_array + "new_impl", clean_tex=False)


def standards_experiments():
    arrsum = ArraySummary("Standard experiments")
    with arrsum.create(tex.Section("Overview")):
        arrsum.add_tabular(get_tab_data(data, parameters, by_reward_type_and_param, by_discount_factor, get_mean_critic_q, name="critic quality"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_reward_type_and_param, by_discount_factor, get_mean_end_reward, name="mean reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_reward_type_and_param, by_discount_factor, get_std_end_reward, name="std reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_reward_type_and_param, by_discount_factor, get_mean_end_loss_at_rl, name="mean loss"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_reward_type_and_param, by_discount_factor, get_std_end_loss, name="std loss"))
    grouped_by_type = group_parameters_by(parameters, by_reward_type_and_param)
    for exp_type in sorted(grouped_by_type):
        with arrsum.create(tex.Section("{}".format(exp_type))):
            for p in sorted(grouped_by_type[exp_type]):
                arrsum.add_experiment(p)
            arrsum.clearpage()
    arrsum.generate_pdf(filepath=path_to_array + "new_impl", clean_tex=False)


def array_wrt_buffer_size():
    arrsum = ArraySummary("WRT buffer size")
    with arrsum.create(tex.Section("Overview")):
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_buffer_size, get_mean_critic_q, name="critic quality"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_buffer_size, get_mean_end_reward, name="mean reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_buffer_size, get_std_end_reward, name="std reward"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_buffer_size, get_mean_end_loss_at_rl, name="mean loss at rl"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_buffer_size, get_mean_end_loss_at_model, name="mean loss at model"))
        arrsum.add_tabular(get_tab_data(data, parameters, by_nothing, by_buffer_size, get_std_end_loss, name="std loss"))
    grouped_by_buffer_size = group_parameters_by(parameters, by_buffer_size)
    for buf_size in sorted(grouped_by_buffer_size):
        with arrsum.create(tex.Section("{}".format(buf_size))):
            for p in grouped_by_buffer_size[buf_size]:
                arrsum.add_experiment(p)
            arrsum.clearpage()
    arrsum.generate_pdf(filepath=path_to_array + "new_impl", clean_tex=False)


standards_experiments()
