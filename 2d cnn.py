import numpy as np

# ============================================================
# Utility functions
# ============================================================

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_hat, y):
    N = y.shape[0]
    return -np.sum(np.log(y_hat[np.arange(N), y])) / N

def softmax_cross_entropy_backward(logits, y):
    """
    dL/dz for softmax + cross entropy
    """
    y_hat = softmax(logits)
    N = y.shape[0]
    y_hat[np.arange(N), y] -= 1
    return y_hat / N


# ============================================================
# Layers
# ============================================================

class Conv2D:
    def __init__(self, num_filters, kernel_size):
        self.F = num_filters
        self.K = kernel_size
        self.W = np.random.randn(self.F, self.K, self.K) * 0.1
        self.b = np.zeros(self.F)

    def forward(self, x):
        # x: (N, H, W)
        self.x = x
        N, H, W = x.shape
        out_h = H - self.K + 1
        out_w = W - self.K + 1

        out = np.zeros((N, self.F, out_h, out_w))

        for n in range(N):
            for f in range(self.F):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = x[n, i:i+self.K, j:j+self.K]
                        out[n, f, i, j] = np.sum(patch * self.W[f]) + self.b[f]

        return out

    def backward(self, d_out, lr):
        N, H, W = self.x.shape
        _, _, out_h, out_w = d_out.shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(self.x)

        for n in range(N):
            for f in range(self.F):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = self.x[n, i:i+self.K, j:j+self.K]
                        dW[f] += d_out[n, f, i, j] * patch
                        db[f] += d_out[n, f, i, j]
                        dx[n, i:i+self.K, j:j+self.K] += d_out[n, f, i, j] * self.W[f]

        self.W -= lr * dW
        self.b -= lr * db

        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask


class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * 0.1
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d_out, lr):
        dW = self.x.T @ d_out
        db = np.sum(d_out, axis=0)
        dx = d_out @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db

        return dx


# ============================================================
# Training step (single iteration)
# ============================================================

np.random.seed(0)

# Fake MNIST-like data
X = np.random.rand(2, 8, 8)   # (batch=2, height=8, width=8)
y = np.array([3, 1])          # labels

# Model
conv = Conv2D(num_filters=2, kernel_size=3)
relu = ReLU()
fc = Dense(in_dim=2 * 6 * 6, out_dim=10)

lr = 0.01

# ---------------- FORWARD ----------------
conv_out = conv.forward(X)                # (2, 2, 6, 6)
relu_out = relu.forward(conv_out)

flat = relu_out.reshape(2, -1)             # (2, 72)
logits = fc.forward(flat)                  # (2, 10)

loss = cross_entropy(softmax(logits), y)
print("Loss:", loss)

# ---------------- BACKWARD ----------------
d_logits = softmax_cross_entropy_backward(logits, y)
d_flat = fc.backward(d_logits, lr)

d_relu_in = d_flat.reshape(relu_out.shape)
d_relu = relu.backward(d_relu_in)
_ = conv.backward(d_relu, lr)

print("Backward pass complete")
