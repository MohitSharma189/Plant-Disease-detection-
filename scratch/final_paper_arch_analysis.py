"""
Comparative analysis between ResNet18 and EfficientNet-B0 on edge devices.
"""

ARCH_EDGE_ANALYSIS = """\
<b>Architectural Latency Dynamics & Computational Bottleneck Profile:</b><br/>
Although EfficientNet-B0 has fewer total parameters (4.06M) than ResNet18 (11.20M), \
its single-image inference latency on CPU (24.58 ms) is almost identical to ResNet18 (25.54 ms), \
and its end-to-end multi-scale pipeline latency is actually 21.3% higher (2,457.4 ms vs. 2,025.9 ms). \
This empirical counter-intuition stems from the memory access cost (MAC) of depthwise separable \
convolutions and inverted bottleneck expansions. In standard convolutional layers (such as in ResNet18), \
large matrix-multiplication operations are highly optimized via Intel MKL/OpenBLAS GEMM kernels with \
high arithmetic intensity. Conversely, EfficientNet-B0 and MobileNetV3 rely extensively on depthwise \
convolutions with low arithmetic intensity, which become memory-bandwidth bound on x86 CPU architectures \
without dedicated neural accelerator hardware. \
MobileNetV3-Small mitigates this memory-access bottleneck by reducing channel depths in intermediate \
layers and selectively applying Squeeze-and-Excitation modules to only half of the inverted bottleneck blocks, \
enabling it to achieve 9.42 ms latency—2.6&times; faster than EfficientNet-B0 and 2.7&times; faster than ResNet18.
"""

