def forward(xs):
    h = np.zeros((H, 1))
    c = np.zeros((H, 1))

    cache = []

    for x in xs:
        x = x.reshape(D, 1)
        z = np.vstack((h, x))

        f = sigmoid(Wf @ z + bf)
        i = sigmoid(Wi @ z + bi)
        c_hat = np.tanh(Wc @ z + bc)
        c = f * c + i * c_hat
        o = sigmoid(Wo @ z + bo)
        h = o * np.tanh(c)

        cache.append((z, f, i, c_hat, c, o, h))

    y = Wy @ h + by
    return y, h, c, cache
def backward(xs, y_true, y_pred, cache):
    global Wf, Wi, Wc, Wo, Wy
    global bf, bi, bc, bo, by

    # Gradients
    dWf = np.zeros_like(Wf)
    dWi = np.zeros_like(Wi)
    dWc = np.zeros_like(Wc)
    dWo = np.zeros_like(Wo)
    dWy = np.zeros_like(Wy)

    dbf = np.zeros_like(bf)
    dbi = np.zeros_like(bi)
    dbc = np.zeros_like(bc)
    dbo = np.zeros_like(bo)

    # ---- output layer (many-to-one) ----
    dy = y_pred - y_true
    dWy = dy @ cache[-1][6].T
    dby = dy                      # ← NO += (single use)

    dh_next = Wy.T @ dy
    dc_next = np.zeros((H, 1))

    # ---- BPTT ----
    for t in reversed(range(T)):
        z, f, i, c_hat, c, o, h = cache[t]
        c_prev = cache[t-1][4] if t > 0 else np.zeros_like(c)

        tanh_c = np.tanh(c)

        dh = dh_next
        dc = dc_next + dh * o * (1 - tanh_c**2)

        do = dh * tanh_c
        do *= o * (1 - o)

        df = dc * c_prev
        df *= f * (1 - f)

        di = dc * c_hat
        di *= i * (1 - i)

        dc_hat = dc * i
        dc_hat *= (1 - c_hat**2)

        dWo += do @ z.T
        dWf += df @ z.T
        dWi += di @ z.T
        dWc += dc_hat @ z.T
        

        dbo += do
        dbf += df
        dbi += di
        dbc += dc_hat

        dz = (
            Wo.T @ do +
            Wf.T @ df +
            Wi.T @ di +
            Wc.T @ dc_hat
        )

        dh_next = dz[:H]
        dc_next = dc * f

    # Update
    for param, grad in zip(
        [Wf, Wi, Wc, Wo, Wy, bf, bi, bc, bo, by],
        [dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby]
    ):
        param -= lr * grad
