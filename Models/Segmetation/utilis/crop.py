def center_crop(x, down_x):
    diff_x = down_x.shape[-1] - x.shape[-1]
    diff_y = down_x.shape[-2] - x.shape[-2]

    down_x = down_x[..., diff_y // 2: down_x.shape[-2] - diff_y // 2, diff_x // 2: down_x.shape[-1] - diff_x // 2]

    return down_x
