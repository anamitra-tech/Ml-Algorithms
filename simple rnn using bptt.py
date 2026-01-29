def forward(xs, h0):
    """
    xs: list of x_t, each (D, 1)
    h0: initial hidden state (H, 1)
    """
    hs = {}
    hs[-1] = h0
    ys = {}

    for t in range(T):
        hs[t] = np.tanh(
            W_xh @ xs[t] +
            W_hh @ hs[t-1] +
            b_h
        )
        ys[t] = W_hy @ hs[t] + b_y

    return ys, hs
def backward(xs, ys, hs, y_true):
    # Initialize gradients
    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    dW_hy = np.zeros_like(W_hy)
    db_h = np.zeros_like(b_h)
    db_y = np.zeros_like(b_y)

    # Gradient of loss wrt output
    dy = ys[T-1] - y_true   # dL/dy_T

    # Output layer gradients
    dW_hy += dy @ hs[T-1].T
    db_y += dy

    # Initial gradient flowing into last hidden state
    dh_next = W_hy.T @ dy   # dL/dh_T

    # ---------------------------
    # BPTT loop
    # ---------------------------
    for t in reversed(range(T)):
        # tanh derivative
        dh = dh_next
        dtanh = dh * (1 - hs[t] ** 2)

        db_h += dtanh
        dW_xh += dtanh @ xs[t].T
        dW_hh += dtanh @ hs[t-1].T

        # propagate to previous time step
        dh_next = W_hh.T @ dtanh

    return dW_xh, dW_hh, dW_hy, db_h, db_y
