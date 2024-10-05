import numpy as np

def softmax(x, axis=None):
    """
    Compute softmax along the specified axis for stability.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def tensor_dot(q, k):
    """
    Compute the scaled dot-product between query and key.
    This is the core of self-attention, and results in an attention score matrix.
    """
    attention_scores = (k @ q) / np.sqrt(q.shape[0])  # Scaled by sqrt of query size
    attention_weights = softmax(attention_scores)
    return attention_weights

def attention_layer(q, k, v):
    """
    Computes the attention output using query, key, and value matrices.
    """
    attention_weights = tensor_dot(q, k)
    return attention_weights @ v

def batched_tensor_dot(q, k):
    """
    Computes batched dot product using Einstein summation notation.
    Efficiently calculates scaled dot-product attention for batches.
    """
    attention_scores = np.einsum("ij,kj->ik", q, k) / np.sqrt(q.shape[0])
    attention_weights = softmax(attention_scores, axis=1)  # Softmax over sequence length
    return attention_weights

def self_attention(x):
    """
    Performs self-attention where query, key, and value are the same input.
    """
    attention_weights = batched_tensor_dot(x, x)  # Self-attention: q = k = x
    return attention_weights @ x

# Random weight initialization for the trainable self-attention mechanism
w_q = np.random.normal(size=(4, 4))  # Query weights
w_k = np.random.normal(size=(4, 4))  # Key weights
w_v = np.random.normal(size=(4, 2))  # Value weights

def trainable_self_attention(x, w_q, w_k, w_v):
    """
    Trainable self-attention with learned query, key, and value weights.
    """
    q = x @ w_q  # Compute the query
    k = x @ w_k  # Compute the key
    v = x @ w_v  # Compute the value
    attention_weights = batched_tensor_dot(q, k)
    return attention_weights @ v  # Return the weighted sum of values

# Multi-head attention weights for two heads
w_q_h1 = np.random.normal(size=(4, 4))
w_k_h1 = np.random.normal(size=(4, 4))
w_v_h1 = np.random.normal(size=(4, 2))
w_q_h2 = np.random.normal(size=(4, 4))
w_k_h2 = np.random.normal(size=(4, 4))
w_v_h2 = np.random.normal(size=(4, 2))
w_h = np.random.normal(size=(2, 1))  # Output projection after concatenating heads

def multihead_attention(x, w_q_h1, w_k_h1, w_v_h1, w_q_h2, w_k_h2, w_v_h2):
    """
    Multi-head attention with two attention heads.
    Each head computes self-attention independently, and the results are concatenated.
    """
    h1_out = trainable_self_attention(x, w_q_h1, w_k_h1, w_v_h1)  # Head 1
    h2_out = trainable_self_attention(x, w_q_h2, w_k_h2, w_v_h2)  # Head 2
    # Concatenate the two heads along the last axis
    all_heads = np.stack((h1_out, h2_out), axis=-1)
    return np.squeeze(all_heads @ w_h)  # Apply output projection

# Example usage with random input queries, keys, and values
i_query = np.random.normal(size=(4,))
i_keys = np.random.normal(size=(11, 4))

# Test tensor dot for basic attention
attention_scores = tensor_dot(i_query, i_keys)
print("Attention Scores (single query):", attention_scores)

# Test attention layer with values
i_values = np.random.normal(size=(11, 2))
attention_output = attention_layer(i_query, i_keys, i_values)
print("Attention Layer Output:", attention_output)

# Test batched self-attention
i_batched_query = np.random.normal(size=(11, 4))
self_attention_output = self_attention(i_batched_query)
print("Self-Attention Output (batched):", self_attention_output)

# Test trainable self-attention
trainable_attention_output = trainable_self_attention(i_batched_query, w_q, w_k, w_v)
print("Trainable Self-Attention Output:", trainable_attention_output)

# Test multi-head attention
multihead_output = multihead_attention(i_batched_query, w_q_h1, w_k_h1, w_v_h1, w_q_h2, w_k_h2, w_v_h2)
print("Multi-Head Attention Output:", multihead_output)
