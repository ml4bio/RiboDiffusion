import torch
import torch.nn.functional as F
import math


class NoiseScheduleVP:
    def __init__(self,
                 schedule='discrete',
                 betas=None,
                 alphas_cumprod=None,
                 continuous_beta_0=0.1,
                 continuous_beta_1=20.,
                 dtype=torch.float32):
        """
        Create a wrapper class for the forward SDE (VP type). From DPM-Solver.
        Notes: cosine schedule for continuous-time setting may face numerical issues. Please refer to the latest version
            of DPM-Solver (https://github.com/LuChengTHU/dpm-solver) for further modification.
        """

        if schedule not in ['discrete', 'linear', 'cosine', 'discrete_poly']:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule))

        self.schedule = schedule
        if 'discrete' in schedule:
            if schedule == 'discrete_poly':
                alphas_cumprod = get_polynomial_schedule(1000, power=2)
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            elif beta is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.


    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues.
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas


    def marginal_log_mean_coeff(self, t):
        """Compute log(alpha_t) of a given continuous-time label t in [0,T]."""
        if 'discrete' in self.schedule:
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """Compute alpha_t of a given continuous-time label t in [0, T]."""
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """Compute sigma_t of a given continuous-time label t in [0, T]."""
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_prob(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        return torch.exp(log_mean_coeff), torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif 'discrete' in self.schedule:
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t

    def get_noiseLevel(self, t):
        alpha_t = self.marginal_alpha(t)
        sigma_t = self.marginal_std(t)
        return torch.log(alpha_t ** 2 / sigma_t ** 2)

    def add_noise(self, data, t):
        # TODO: adding noise to data
        pass


#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]


def get_polynomial_schedule(time_steps, s=1e-4, power=2):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power. (from E3 Diffusion)
    """
    steps = time_steps + 1
    x = torch.linspace(0, steps, steps)
    alphas2 = (1 - torch.pow(x / steps, power))**2

    # clip alpha_t / alpha_t-1. This may help improve stability during sampling.
    alphas2 = torch.cat([torch.ones(1), alphas2], dim=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = torch.clamp(alphas_step, min=0.001, max=1.)
    alphas2 = torch.cumprod(alphas_step, dim=0)

    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s

    return alphas2[1:]


if __name__ == "__main__":
    poly_sch = NoiseScheduleVP('discrete_poly')
    cos_sch = NoiseScheduleVP('cosine')
    lin_sch = NoiseScheduleVP('linear')
    t1 = torch.tensor(0.1)
    t2 = torch.tensor(0.2)

    t3 = torch.tensor(0.5)
    t4 = torch.tensor(0.9)
    time_steps = [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3), torch.tensor(0.4),
                  torch.tensor(0.6), torch.tensor(0.7), torch.tensor(0.8), torch.tensor(0.9)]

    for sch in [poly_sch, cos_sch, lin_sch]:
        print(sch.schedule)
        # for tt in time_steps:
        #     print(tt)
        #     print(sch.marginal_alpha(tt), sch.marginal_std(tt))
        print(sch.marginal_prob(torch.tensor(0.)))
        print('-'*80)

