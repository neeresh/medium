from sklearn.model_selection import StratifiedKFold


def k_fold(data, target):
    skf = StratifiedKFold(5, shuffle=True)

    train_sets, train_targets = [], []
    val_sets, val_targets = [], []
    test_sets, test_targets = [], []

    for raw_index, test_index in skf.split(data, target):
        raw_set = data[raw_index]
        raw_target = target[raw_index]

        test_sets.append(data[test_index])
        test_targets.append(target[test_index])

        train_index, val_index = next(StratifiedKFold(4, shuffle=True).split(raw_set, raw_target))
        train_sets.append(raw_set[train_index])
        train_targets.append(raw_target[train_index])

        val_sets.append(raw_set[val_index])
        val_targets.append(raw_target[val_index])

    return train_sets, train_targets, val_sets, val_targets, test_sets, test_targets
