if __name__ == "__main__":
    import argparse
    import dtgoal2
    import pickle
    import os
    import re

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_path',
        type=str,
        action='store',
        help="Path to a worker pickle file."
    )

    args = parser.parse_args()
    dumps_path = os.path.split(args.input_path)[0]
    max_plot = [x for x in os.listdir(dumps_path + "/../")]
    max_plot = [re.match("plots_([0-9]+)", x) for x in max_plot if re.match("plots_([0-9]+)", x)]
    max_plot = [int(x.group(1)) for x in max_plot if x]
    max_plot.sort()
    max_plot = 0 if len(max_plot) == 0 else max_plot[-1] + 1
    output_path = dumps_path + "/../plots_{}".format(max_plot)
    output_path = os.path.realpath(output_path)
    os.mkdir(output_path)


    print("\n", output_path, "\n")


    with open(args.input_path, "rb") as f:
        g = pickle.load(f)

    g.plot_per_goal(output_path, "default_name")

    os.system("gnome-open " + output_path + "/default_name_goal_0000.png")
