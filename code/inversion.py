"""Fixed-point solver independent of model and tensor backend."""
import math


def validate_fixed_point(iterations, relaxation):
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError('inversion_iterations must be a positive integer.')
    if isinstance(relaxation, bool) or not isinstance(relaxation, (int, float)) or not math.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError('inversion_relaxation must be finite and in (0, 1].')


def fixed_point_solve(previous, predict_noise, invert_step, iterations, relaxation=1.):
    """Solve x = inverse_DDIM(previous, epsilon(x, t)).

    The cleaner latent `previous` stays fixed throughout the solve. The first
    guess is that latent. `predict_noise(guess, final_iteration)` lets the caller
    record attention ONLY on the final model evaluation. No warm-start or extra
    post-solve UNet call is hidden in the iteration count. Returned maps therefore
    correspond to the last evaluated guess, before its final fixed-point update.
    """
    validate_fixed_point(iterations, relaxation)
    guess = previous
    for iteration in range(iterations):
        noise = predict_noise(guess, iteration == iterations - 1)
        proposal = invert_step(noise, previous)
        guess = (1-relaxation) * guess + relaxation * proposal
    return guess
