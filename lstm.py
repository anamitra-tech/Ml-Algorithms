import numpy as np

# ============================================================
# BASIC FUNCTIONS
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    # derivative of sigmoid, assuming y = sigmoid(x)
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    # derivative of tanh, assuming y = tanh(x)
    return 1 - y**2


# ============================================================
# MODEL DIMENSIONS
# ============================================================

input_size = 3     # features per timestep
hidden_size = 2    # memory size
seq_len = 4        # length of sequence

np.random.seed(0)


# ============================================================
# WEIGHT INITIALIZATION
# Each gate has its own parameters
# ============================================================

def init_gate():
    W = np.random.randn(hidden_size, hidden_size + input_size)
    b = np.zeros(hidden_size)
    return W, b

Wf, bf = init_gate()  # forget gate
Wi, bi = init_gate()  # input gate
Wc, bc = init_gate()  # candidate memory
Wo, bo = init_gate()  # output gate


# ============================================================
# INPUT SEQUENCE
# ============================================================

X = np.random.randn(seq_len, input_size)


# ============================================================
# FORWARD PASS STORAGE
# ============================================================

h = np.zeros((seq_len + 1, hidden_size))  # hidden states
c = np.zeros((seq_len + 1, hidden_size))  # cell states

f_list, i_list, c_hat_list, o_list, z_list = [], [], [], [], []


# ============================================================
# FORWARD PASS (LSTM)
# ============================================================

for t in range(seq_len):

    x_t = X[t]

    # Combine previous hidden state and current input
    z = np.concatenate([h[t], x_t])
    z_list.append(z)

    # Gates
    f = sigmoid(Wf @ z + bf)        # forget gate
    i = sigmoid(Wi @ z + bi)        # input gate
    c_hat = tanh(Wc @ z + bc)       # candidate memory
    o = sigmoid(Wo @ z + bo)        # output gate

    # Store gate values for backprop
    f_list.append(f)
    i_list.append(i)
    c_hat_list.append(c_hat)
    o_list.append(o)

    # Cell state update (memory highway)
    c[t+1] = f * c[t] + i * c_hat

    # Hidden state (exposed memory)
    h[t+1] = o * tanh(c[t+1])


# ============================================================
# LOSS (simple scalar)
# ============================================================

# Target: final hidden state should be zero
loss = np.sum(h[-1] ** 2)

# Initial gradients from loss
dh_next = 2 * h[-1]
dc_next = np.zeros(hidden_size)


# ============================================================
# BACKPROPAGATION THROUGH TIME (BPTT)
# ============================================================

# Initialize gradients
dWf = np.zeros_like(Wf); dbf = np.zeros_like(bf)
dWi = np.zeros_like(Wi); dbi = np.zeros_like(bi)
dWc = np.zeros_like(Wc); dbc = np.zeros_like(bc)
dWo = np.zeros_like(Wo); dbo = np.zeros_like(bo)


for t in reversed(range(seq_len)):

    # -------------------------------
    # Output gate gradient
    # -------------------------------
    do = dh_next * tanh(c[t+1])
    do_raw = do * dsigmoid(o_list[t])

    # -------------------------------
    # Cell state gradient
    # -------------------------------
    dc = dh_next * o_list[t] * dtanh(tanh(c[t+1]))
    dc += dc_next   # gradient from future timestep

    # -------------------------------
    # Input gate gradient
    # -------------------------------
    di = dc * c_hat_list[t]
    di_raw = di * dsigmoid(i_list[t])

    # -------------------------------
    # Candidate memory gradient
    # -------------------------------
    dc_hat = dc * i_list[t]
    dc_hat_raw = dc_hat * dtanh(c_hat_list[t])

    # -------------------------------
    # Forget gate gradient
    # -------------------------------
    df = dc * c[t]
    df_raw = df * dsigmoid(f_list[t])

    # -------------------------------
    # Parameter gradients
    # -------------------------------
    z = z_list[t]

    dWf += np.outer(df_raw, z); dbf += df_raw
    dWi += np.outer(di_raw, z); dbi += di_raw
    dWc += np.outer(dc_hat_raw, z); dbc += dc_hat_raw
    dWo += np.outer(do_raw, z); dbo += do_raw

    # -------------------------------
    # Propagate to previous timestep
    # -------------------------------
    dz = (
        Wf.T @ df_raw +
        Wi.T @ di_raw +
        Wc.T @ dc_hat_raw +
        Wo.T @ do_raw
    )

    dh_next = dz[:hidden_size]
    dc_next = dc * f_list[t]   # THIS is gradient survival


# ============================================================
# DONE
# ============================================================

print("Loss:", loss)
