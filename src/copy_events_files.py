if __name__ == "__main__":
    import os
    import argparse
    from tempfile import TemporaryDirectory


    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        nargs="+",
        action='store',
        help="Path to the checkpoint."
    )

    args = parser.parse_args()

    def commonprefix(m):
        "Given a list of pathnames, returns the longest common leading component"
        if not m: return ''
        s1 = min(m)
        s2 = max(m)
        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i]
        return s1

    basenames = [os.path.basename(p) for p in args.path]
    prefix = commonprefix(basenames)

    grouppath = "../grouped_events_files/{}/".format(prefix)
    os.mkdir(grouppath)
    for path, basename in zip(args.path, basenames):
        runname = basename.replace(prefix, "")
        os.mkdir(grouppath + "/" + runname)
        filename = [f for f in os.listdir(path + "/log") if f.find("tfevents") != -1][0]
        os.link(path + '/log/' + filename, grouppath + "/" + runname + "/" + filename)
    os.system("tensorboard --logdir {}".format(grouppath))
