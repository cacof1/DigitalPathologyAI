import numpy as np

## Conv2D Transpose
def FindTransposeSize(W,stride,padding,dilation,kernel_size,output_padding):
    return (W-1)*stride -2*padding + dilation*(kernel_size - 1) + output_padding + 1

def FindConvSize(W,stride,padding,dilation,kernel_size):
    return (W + 2*padding - dilation*(kernel_size - 1) -1)/stride + 1

W              = np.array([256,256])
stride         = np.array([2,2])
padding        = np.array([0,0])
dilation       = np.array([1,1])
kernel_size    = np.array([2,2])
output_padding = np.array([0,0])
print("Original Size", W)
print("Tranpose size", FindTransposeSize(W,stride,padding,dilation,kernel_size,output_padding))
print("Conv size", FindConvSize(W,stride,padding,dilation,kernel_size))
