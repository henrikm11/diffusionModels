//diffusion.cpp


#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include "diffusion.h"

//implementation of GeneralDiffusor



/*
//constructors
*/
diffusion::GeneralDiffusor::GeneralDiffusor(
    double (*driftFct)(double,double),
    double (*diffusionFct)(double, double)
)
    :rng_(std::random_device{}())
{
    driftFct_=std::make_unique<diffusion::ScalarFuncHelper>(driftFct);
    diffusionFct_=std::make_unique<diffusion::ScalarFuncHelper>(diffusionFct);
    return;
}

diffusion::GeneralDiffusor::GeneralDiffusor(
    std::vector<double> (*driftFct)(const std::vector<double>&, double),
    std::vector<double> (*diffusionFct)(const std::vector<double>&,double)
)
    :rng_(std::random_device{}())
{
    driftFct_=std::make_unique<diffusion::VectorFuncHelper>(driftFct);
    diffusionFct_=std::make_unique<diffusion::VectorFuncHelper>(diffusionFct);
    return;
}

diffusion::GeneralDiffusor::GeneralDiffusor(
    const diffusion::FuncHelper& driftFct,
    const diffusion::FuncHelper& diffusionFct
)
    :
    driftFct_(driftFct.modifiedClone(1.0,1.0,driftFct.multiply_)), //this keeps the original function
    diffusionFct_(diffusionFct.modifiedClone(1.0,1.0,diffusionFct.multiply_)),  
    rng_(std::random_device{}())
{
    return;
}

/*
//destructor
*/
diffusion::GeneralDiffusor::~GeneralDiffusor(void){return;}


/*
//public member functions
*/
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
    outputState+=((*driftFct_)(inputState,time)*timeStep);
    outputState+=((*diffusionFct_)(inputState,time)*sqrt(timeStep)*z);
    return outputState;
}

std::vector<double> diffusion::GeneralDiffusor::diffusionStep(
    const std::vector<double>& inputState,
    double time,
    double timeStep
){
    std::vector<double> outputState = (*driftFct_)(inputState,time); //overload in FuncHelper resolves this!
    for(size_t i=0; i<outputState.size(); i++){
        outputState[i]*=timeStep;
        outputState[i]+=inputState.at(i);
    }

    std::vector<double> noise = (*diffusionFct_)(inputState,time);
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
        diffusion::ScalarFuncHelper(varianceScheduleFct, -0.5, 1.0,true),
        diffusion::ScalarFuncHelper(varianceScheduleFct, 1.0, 0.5,false)
    )
{
    varianceScheduleIntegralFct_=std::make_unique<diffusion::ScalarFuncHelper>(varianceScheduleIntegralFct);
}

diffusion::OUDiffusor::OUDiffusor(
    const ScalarFuncHelper& varianceScheduleFct,
    const ScalarFuncHelper& varianceScheduleIntegralFct
)
    :
    diffusion::GeneralDiffusor::GeneralDiffusor(
        diffusion::ScalarFuncHelper(varianceScheduleFct, -0.5, 1.0,true),
        diffusion::ScalarFuncHelper(varianceScheduleFct, 1.0, 0.5,false)
    )
{
    varianceScheduleIntegralFct_=std::make_unique<diffusion::ScalarFuncHelper>(varianceScheduleIntegralFct);
}

diffusion::OUDiffusor::OUDiffusor(
    double betaMin,
    double betaMax,
    double timeMax
)
    :diffusion::GeneralDiffusor::GeneralDiffusor(
        diffusion::ExplicitFuncHelper(betaMin, betaMax, timeMax, -0.5, 1.0, true, false),
        diffusion::ExplicitFuncHelper(betaMin, betaMax, timeMax, 1.0, 0.5, false, false)
    )
{
    varianceScheduleIntegralFct_=
        std::make_unique<diffusion::ExplicitFuncHelper>(
            betaMin, betaMax, timeMax, 1.0, 1.0, false, true
    );
    return;
}

diffusion::OUDiffusor::~OUDiffusor(){}


double diffusion::OUDiffusor::sample(
    double inputSample,
    double time
)
{
    double z = getRandomNormal();
    double rate = (*varianceScheduleIntegralFct_)(time);
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
    :OUDiffusor(betaMin,betaMax,timeMax)
{}








