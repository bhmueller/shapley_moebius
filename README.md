# Shapley Effects by Möbius Inverse

In this repo, I implement algorithm 5.1 from Plischke, Rabitti and Borgonovo (2020). The
implementation differs on whether model inputs are dependent or not. I translate the
algorithms from MATLAB to Python and implement corresponding test cases, which are taken
from Iooss and Prieur (2019) and Plischke, Rabitti and Borgonovo (2020).

## Shapley Effects

Shapley effects were introduced as a sensitivity measure by Owen (2014). They yield information
about the input-output relationship for models which are considered a black box. For instance,
a model that can be solved by numerical methods only, has no analytical solution. Thus,
it is not clear, how the output depends on the inputs. Sensitivity analysis tries to shed
light on this relationship. Shapley effects give the contribution to the reduction of
output variance for each input. Shapley effects incorporate effects due to interactions
and input dependence.

## Acknowledgement

I thank Dr. Elmar Plischke for providing me with the MATLAB implementation of algorithm 5.1 for both,
dependent and independent inputs.

## References

Iooss, Betrand and Clémentine Prieur. 2019. Shapley effects for sensitivity analysis with correlated inputs: comparisons with Sobol’ indices, numerical estimation and applications. hal-01556303v6

Owen, Art B. 2014. Sobol’ indices and shapley value. SIAM/ASA Journal on Uncertainty Quantification, 2(1):245–251.

Plischke, Elmar, Giovanni Rabitti, Emanuele Borgonovo. 2020. Computing Shapley Effects for Sensitivity Analysis. arXiv.