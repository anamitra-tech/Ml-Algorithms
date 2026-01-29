import numpy as np

class Adam:
    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None   # first moment
        self.v = None   # second moment
        self.t = 0      # time step

    def update(self, params, grads):
        """
        params: numpy array of parameters
        grads: numpy array of gradients (same shape)
        """

        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Parameter update
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return params
