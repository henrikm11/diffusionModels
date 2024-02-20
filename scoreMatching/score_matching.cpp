//score_matching.cpp


#include <vector>
#include "score_matching.h"


//implementation of score matching

diffusionModel::ScoreModel::ScoreModel(
    diffusion::GeneralDiffusor diffusor,
    TensorShape<2> networkShape,
    double (*const timeWeight_)(double),
    double (*const activationFunction)(double),
    double (*const activationFunctionGrad)(const double),
    double (*const timeWeights)(const double),
    double maxTime
)
    :diffusor_(diffusor), 
    neuralNet_(networkShape, activationFunction, activationFunctionGrad, timeWeights, maxTime),
    rng_(std::random_device{}())
{
    return;
}