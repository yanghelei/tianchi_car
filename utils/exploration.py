import math
from typing import Callable


def get_epsilon_greedy_fn(start: float, end: float, decay: int, type_: str = 'exp') -> Callable:
    """
    Overview:
        Generate an epsilon_greedy function with decay, which inputs current timestep and outputs current epsilon.
    Arguments:
        - start (:obj:`float`): Epsilon start value. For 'linear', it should be 1.0.
        - end (:obj:`float`): Epsilon end value.
        - decay (:obj:`int`): Controls the speed that epsilon decreases from ``start`` to ``end``. \
            We recommend epsilon decays according to env step rather than iteration.
        - type (:obj:`str`): How epsilon decays, now supports ['linear', 'exp'(exponential)]
    Returns:
        - eps_fn (:obj:`function`): The epsilon greedy function with decay
    """
    assert type_ in ['linear', 'exp'], type_
    if type_ == 'exp':
        return lambda x: (start - end) * math.exp(-1 * x / decay) + end
    elif type_ == 'linear':

        def eps_fn(x):
            if x >= decay:
                return end
            else:
                return (start - end) * (1 - x / decay) + end

        return eps_fn
