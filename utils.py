import math
import torch

# --- Log Probability of Gaussian ---
# Calculates the log probability density of a sample `t` under a multivariate
# Gaussian distribution with diagonal covariance.
# Assumes `mean` and `log_std` define the distribution parameters.
def create_log_gaussian(mean, log_std, t):
    # Calculate the quadratic term: -0.5 * ((t - mean) / std)²
    # log_std.exp() computes the standard deviation (std) from log_std
    # .pow(2) squares the result
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    # Get the shape of the mean tensor (used to find the dimension)
    l = mean.shape
    # Calculate the log of the normalization constant component related to std: log(σ)
    log_z = log_std
    # Calculate the log of the normalization constant component related to dimensions: 0.5 * D * log(2π)
    # l[-1] gives the last dimension (number of variables, D)
    z = l[-1] * math.log(2 * math.pi)
    # Compute the final log probability density: sum(quadratic term) - sum(log σ) - 0.5 * D * log(2π)
    # Summing assumes independence across dimensions (diagonal covariance)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    # Return the computed log probability
    return log_p

# --- Log-Sum-Exp Trick ---
# Computes log(sum(exp(inputs))) in a numerically stable way.
# Useful for operations involving probabilities in log-space.
def logsumexp(inputs, dim=None, keepdim=False):
    # If dimension is not specified, flatten the input and compute over dimension 0
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    # Find the maximum value along the specified dimension (`s`)
    # Keep the dimension for broadcasting (`keepdim=True`)
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    # Compute the log-sum-exp using the formula: max(inputs) + log(sum(exp(inputs - max(inputs))))
    # Subtracting the max `s` before exponentiating prevents overflow/underflow.
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    # Remove the extra dimension if `keepdim` was False
    if not keepdim:
        outputs = outputs.squeeze(dim)
    # Return the result
    return outputs

# --- Soft Target Network Update (Polyak Averaging) ---
# Updates the parameters of the target network (`target`) slowly towards the
# parameters of the source network (`source`) using a factor `tau`.
# Formula: target_param = (1 - tau) * target_param + tau * source_param
def soft_update(target, source, tau):
    # Iterate over pairs of parameters from the target and source networks
    for target_param, param in zip(target.parameters(), source.parameters()):
        # Apply the soft update rule using inplace tensor operations (`.data.copy_`)
        # target_param.data is updated with the weighted average
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# --- Hard Target Network Update ---
# Directly copies the parameters from the source network (`source`) to the
# target network (`target`).
# Formula: target_param = source_param
def hard_update(target, source):
    # Iterate over pairs of parameters from the target and source networks
    for target_param, param in zip(target.parameters(), source.parameters()):
        # Copy the source parameter data directly into the target parameter data
        target_param.data.copy_(param.data)