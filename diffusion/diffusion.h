//diffiusion.h

/*
header file for forward diffusion
*/

#pragma once


#include <vector>
#include <random>
#include <stdexcept>
#include "FuncHelper.h"

//forward diffusion along SDE
// dX = \mu(X,t)*dt + \sigma(X,t)*dW_t 
// where W_t is the standard Wiener process

namespace diffusion{


//base class for diffusion along
// dX = drift(X,t)dt + diffusion(X,t)dW_t,
// W_t standard Wiener process
class GeneralDiffusor{
public:
    GeneralDiffusor(
        diffusion::FuncHelper driftFct,
        diffusion::FuncHelper diffusionFct
    );
    GeneralDiffusor(
        double (*driftFct)(double,double),
        double (*diffusionFct)(double, double)
    );
    virtual ~GeneralDiffusor(void);

    /// @brief returns pseudo random number ~ N(0,1)
    double getRandomNormal(void);


    /// @brief diffusion along dY = \mu(Y,t)*dt +  \sigma(Y,t)*dW_t
    /// @param inputState Y(time)
    /// @param time 
    /// @param timeStep compute Y(time+timestep)
    /// @return Y(time+timestep)
    double diffusionStep(
        double inputState,
        double time,
        double timeStep
    );

    //vectorized version of above function
    /// @brief diffusion along dY = \mu(Y,t)*dt +  \sigma(Y,t)*dW_t
    /// @param inputState Y(time)
    /// @param time 
    /// @param timeStep compute Y(time+timestep)
    /// @return Y(time+timestep)
    std::vector<double> diffusionStep(
        const std::vector<double>& inputState,
        double time, 
        double timeStep
    );

    //functions to sample from process
    //virtual because this can be sped up significantly for more specific processes
    
    virtual double sample(
        double inputState,
        double time,
        double timeStepSize
    );

    virtual std::vector<double> sample(
        const std::vector<double>& inputState,
        double time,
        double timeStepSize
    );


private:
    std::mt19937 rng_;
    diffusion::FuncHelper driftFct_;
    diffusion::FuncHelper diffusionFct_;
};

/*

END GeneralDiffusor

*/


/*

BEGIN OUDiffusor

*/



//derived class for Ornstein Uhlenbeck like processes
// dX = -0.5*beta(t)*X(t)*dt + \sqrt(beta(t))*dW_t,
// beta(t) is fixed variance schedule

class OUDiffusor : public GeneralDiffusor{
public:
    OUDiffusor(
        double (*varianceScheduleFct)(double),
        double (*varianceScheduleIntegralFct)(double)
    );
    OUDiffusor(
        FuncHelper varianceScheduleFct,
        FuncHelper varianceScheduleIntegralFct
    );
    virtual ~OUDiffusor(void);

    /// @brief samples from forward diffusion process
    /// @param inputSample sample from initial distribution
    /// @param time sample process at time
    /// @return sample of diffusion process at time 
    virtual double sample(
        double inputSample,
        double time
    );

    virtual std::vector<double> sample(
        const std::vector<double>& inputSample,
        double time
    );

private:
    FuncHelper varianceScheduleIntegralFct_;
};


//ok up to here




//diffusor with linear variance schedule derived from GeneralDiffusor
//beta(t)=betaMin_+(betaMax_-betaMin_)*t/timeMax_


class LinearDiffusor : public OUDiffusor{
public:
    LinearDiffusor(
        double betaMin,
        double betaMax,
        double timeMax
    );

private:
    //somewhat redundant as these are stored in FuncHelper,
    //but that might be inconvenient to access later
    double betaMin_;
    double betaMax_;
    double timeMax_;
    
};




}
//end of namespace diffusion




