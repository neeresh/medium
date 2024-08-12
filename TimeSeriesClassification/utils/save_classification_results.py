import os

import pandas as pd


def save_cls_result(save_dir, save_csv_name, dataset_name, test_accu, test_std, train_time, end_val_epoch, seeds=42):
    save_path = os.path.join(save_dir, '', save_csv_name + '_cls_result.csv')
    accu = test_accu.cpu().numpy()
    std = test_std.cpu().numpy()
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path, index_col=0)
    else:
        result_form = pd.DataFrame(
            columns=['dataset_name', 'test_accuracy', 'test_std', 'train_time', 'end_val_epoch', 'seeds'])

    new_row = pd.DataFrame([{
        'dataset_name': dataset_name,
        'test_accuracy': '%.4f' % accu,
        'test_std': '%.4f' % std,
        'train_time': '%.4f' % train_time,
        'end_val_epoch': '%.2f' % end_val_epoch,
        'seeds': '%d' % seeds
    }])

    result_form = pd.concat([result_form, new_row], ignore_index=True)

    result_form.to_csv(save_path, index=True, index_label="id")
