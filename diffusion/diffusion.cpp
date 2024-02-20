//diffusion.cpp


#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include "diffusion.h"

//implementation of GeneralDiffusor

diffusion::GeneralDiffusor::GeneralDiffusor(
    diffusion::FuncHelper driftFct,
    diffusion::FuncHelper diffusionFct
)
    :driftFct_(driftFct),
    diffusionFct_(diffusionFct),
    rng_(std::random_device{}())
{
    return;
}

diffusion::GeneralDiffusor::GeneralDiffusor(
    double (*driftFct)(double,double),
    double (*diffusionFct)(double, double)
)
    :driftFct_(driftFct),
    diffusionFct_(diffusionFct),
    rng_(std::random_device{}())
{
    return;
}

diffusion::GeneralDiffusor::~GeneralDiffusor(void){return;}

double diffusion::GeneralDiffusor::getRandomNormal(){
    std::normal_distribution<double> dist(0,1); 
    return dist(rng_);
}

double diffusion::GeneralDiffusor::diffusionStep(
    double inputState,
    double time,
    double timeStep
)
{
    double z = getRandomNormal();
    double outputState = inputState;
    outputState+=driftFct_(inputState,time)*timeStep;
    outputState+=diffusionFct_(inputState,time)*sqrt(timeStep)*z;
    return outputState;
}

std::vector<double> diffusion::GeneralDiffusor::diffusionStep(
    const std::vector<double>& inputState,
    double time,
    double timeStep
){
    std::vector<double> outputState(inputState.size(),0);
    for(size_t i=0; i<outputState.size(); i++){
        outputState[i]=diffusionStep(inputState[i],time,timeStep);
    }
    return outputState;
}

//GeneralDiffusor
//end of implementation

//OUDiffusor
//start of implementation



diffusion::OUDiffusor::OUDiffusor(
    double (*varianceScheduleFct)(double),
    double (*varianceScheduleIntegralFct)(double)  
)
    :diffusion::GeneralDiffusor(
        diffusion::FuncHelper(varianceScheduleFct, -0.5, 1.0),
        diffusion::FuncHelper(varianceScheduleFct, 1.0, 0.5) 
    ),
    varianceScheduleIntegralFct_(varianceScheduleIntegralFct)
{}

diffusion::OUDiffusor::OUDiffusor(
    FuncHelper varianceScheduleFct,
    FuncHelper varianceScheduleIntegralFct
)
    :
    diffusion::GeneralDiffusor::GeneralDiffusor(
        diffusion::FuncHelper(varianceScheduleFct, -0.5, 1.0),
        diffusion::FuncHelper(varianceScheduleFct, 1.0, 0.5)
    ),
    varianceScheduleIntegralFct_(varianceScheduleIntegralFct)
{}

diffusion::OUDiffusor::~OUDiffusor(){}


double diffusion::OUDiffusor::oneDimSample(
    double inputSample,
    double time
)
{
    double z = getRandomNormal();
    double rate = varianceScheduleIntegralFct_(time);
    double outputSample = exp(-0.5*rate)*inputSample + (1-exp(-rate))*z;
    
    return outputSample;
}

std::vector<double> diffusion::OUDiffusor::sample(
    const std::vector<double>& inputSample,
    double time
)
{   
    std::vector<double> outputSample(inputSample.size(),0);
    for(size_t i=0; i<inputSample.size(); i++){
        outputSample[i]=oneDimSample(inputSample[i],time);
    }
    return outputSample;
}



//OUDiffusor
//end of implementation


//LinearDiffusor
//begin of implementation

//varianceScheduleFct FuncHelper(betaMin, betaMax, timeMax)

diffusion::LinearDiffusor::LinearDiffusor(
    double betaMin,
    double betaMax,
    double timeMax
)
    :OUDiffusor(
        FuncHelper(betaMin, betaMax, timeMax),
        FuncHelper(betaMin, betaMax, timeMax, 1.0, 1.0, true)
    ),
    betaMin_(betaMin),
    betaMax_(betaMax),
    timeMax_(timeMax)
{}





