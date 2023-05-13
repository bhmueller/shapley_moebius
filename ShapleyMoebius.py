# Shapley Möbius. This class uses other classes as dependencies and implements them.
from shapley_moebius_independent import shapley_moebius_independent
from shapley_moebius_dependent import shapley_moebius_dependent

class ShapleyMoebius:
    
    def independent(k, n, model, trafo):
        shapley_effects, variance = shapley_moebius_independent(k, n, model, trafo)

    def dependent(k, n, model, trafo, rank_corr, random_mode):
        shapley_effects, variance, evals = shapley_moebius_dependent(k, n, model, trafo, rank_corr, random_mode)