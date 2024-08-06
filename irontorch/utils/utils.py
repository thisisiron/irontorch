import pkg_resources


def check_library_version(cur_version, min_version, must_be_same=False):
    current, minimum = (pkg_resources.parse_version(x) for x in (cur_version, min_version))
    return (current == minimum) if must_be_same else (current >= minimum)  # bool

