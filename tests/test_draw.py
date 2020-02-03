# can't actually assertion test anything here; just check that it runs
from arm_pytorch_utilities import draw
import matplotlib.pyplot as plt
import torch


def verify_cumulative_dist():
    a = torch.tensor([[60.68678289, 66.3535433, 143.95848761, 87.77245593, 86.9702349, 74.83818095, 79.16551046,
                       80.57348212, 68.10790145, 81.05427654],
                      [789.96081832, 220.45785346, 205.60528553, 226.78709325, 747.52439502,
                       713.47555814, 835.64565573, 953.39352829, 758.24503011, 245.5031224]
                      ])
    draw.cumulative_dist(a, ['something', 'another'], 'costs')
    plt.show()


if __name__ == "__main__":
    verify_cumulative_dist()
