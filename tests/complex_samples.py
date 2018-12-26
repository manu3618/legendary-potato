"""
Data used to test complex kernels
"""
import math

import legendary_potato.kernel

KERNEL_SAMPLES = {
    legendary_potato.kernel.l2: [
        math.sin,
        math.cos,
        lambda x: 1,
        lambda x: 0,
        lambda x: -1,
        lambda x: x,
        lambda x: abs(x),
    ]
}
