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
    std::vector<double> outputState = driftFct_(inputState,time); //overload in FuncHelper resolves this!
    for(size_t i=0; i<outputState.size(); i++){
        outputState[i]*=timeStep;
    }

    std::vector<double> noise = diffusionFct_(inputState,time);
    for(size_t i=0; i<noise.size(); i++){
        double z = getRandomNormal();
        outputState[i]+=(noise[i]*sqrt(timeStep)*z);
    }

    return outputState;
}

//sample state at time starting from inputState
double diffusion::GeneralDiffusor::sample(
    double inputState,
    double time,
    double timeStepSize
)
{
    if(time<=0) throw(std::invalid_argument("diffusion::GeneralDiffusor::sample(...)need time>0"));
    if(timeStepSize<=0) throw(std::invalid_argument("diffusion::GeneralDiffusor::sample(...)need timeStepSize>0"));

    double currTime=0;
    double res=inputState;

    while(currTime+timeStepSize<=time){
        res=diffusionStep(res,currTime,timeStepSize);
        currTime+=timeStepSize;
    }
    if(currTime<time) res=diffusionStep(res,currTime,time-currTime);
    
    return res;
}


//literally the same code as function above because of overload for diffusionStep
//so could have used a template
std::vector<double> diffusion::GeneralDiffusor::sample(
    const std::vector<double>& inputState,
    double time,
    double timeStepSize
){
    if(time<=0) throw(std::invalid_argument("diffusion::GeneralDiffusor::sample(...)need time>0"));
    if(timeStepSize<=0) throw(std::invalid_argument("diffusion::GeneralDiffusor::sample(...)need timeStepSize>0"));

    double currTime=0;
    std::vector<double> res=inputState;

    while(currTime+timeStepSize<=time){
        res=diffusionStep(res,currTime,timeStepSize);
        currTime+=timeStepSize;
    }
    if(currTime<time) res=diffusionStep(res,currTime,time-currTime);
    
    return res;
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


double diffusion::OUDiffusor::sample(
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
        outputSample[i]=sample(inputSample[i],time);
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





