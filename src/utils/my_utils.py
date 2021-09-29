def remove_exo(my_vars, assigned):
    return dict((k, assigned[k]) for k in my_vars if k in assigned)


def remove_duplicate_dicts(my_list):
    return [dict(t) for t in {tuple(d.items()) for d in my_list}]
