//score matching.h

#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include "diffusion.h"
#include "tensor_template/tensor_template.h"
#include "vanilla_network/neuralNetwork.h"

/*
TO DO: add overloads or derived class for more specific diffusors
*/


namespace diffusionModel{


class ScoreModel{
public:
    ScoreModel(
        diffusion::GeneralDiffusor diffusor,
        TensorShape<2> networkShape,
        double (*const timeWeight_)(double),
        double (*const activationFunction)(double),
        double (*const activationFunctionGrad)(const double),
        double (*const timeWeights)(const double),
        double maxTime
    );
    
    //not implemented yet
    //std::vector<double> sample(void); 


private:
    diffusion::GeneralDiffusor diffusor_; //forward process
    vanillaNeuralNet::neuralNetworkWeightedMSE neuralNet_;
    std::mt19937 rng_;

};



} //end of namespace diffusionModel
