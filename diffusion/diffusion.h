//diffiusion.h

/*
header file for forward diffusion
*/




#pragma once

#include <vector>
#include <random>
#include <memory>
#include "neuralNetwork.h"







//forward diffusion along SDE
// dX = \mu(X,t)*dt + \sigma(X,t)*dW_t 
// where W_t is the standard Wiener process
namespace diffusion{

//forward declarations
class FuncHelper;
class ScalarFuncHelper;
class VectorFuncHelper;
class ExplicitFuncHelper;

//base class for diffusion along
// dX = drift(X,t)dt + diffusion(X,t)dW_t,
// W_t standard Wiener process
class GeneralDiffusor{
public:
    //overloads of constructor depending on input format of drift and diffusion
   
    GeneralDiffusor(
        double (*driftFct)(double,double),
        double (*diffusionFct)(double, double)
    );

    GeneralDiffusor(
        std::vector<double> (*driftFct)(const std::vector<double>&, double),
        std::vector<double> (*diffusionFct)(const std::vector<double>&,double)
    );

    
    GeneralDiffusor(
        const diffusion::FuncHelper& driftFct,
        const diffusion::FuncHelper& diffusionFct
    );

    //copy constructor needs to be custom to call modifiedClone FuncHelper
    //since FuncHelper has deleted copy constructor
    GeneralDiffusor(
        const GeneralDiffusor& other
    );
    
    
   
    
    virtual ~GeneralDiffusor(void); //create vtable

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
    std::unique_ptr<FuncHelper> driftFct_; 
    std::unique_ptr<FuncHelper> diffusionFct_;
};

/*
//
//END GeneralDiffusor
//
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
        const ScalarFuncHelper& varianceScheduleFct,
        const ScalarFuncHelper& varianceScheduleIntegralFct
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
protected:
    //only used to construct LinearDiffusor
    OUDiffusor(
        double betaMin,
        double betaMax,
        double timeMax
    );
private:
    std::unique_ptr<FuncHelper> varianceScheduleIntegralFct_;
};




//separated into own class purely for clarity
//diffusor with linear variance schedule derived from GeneralDiffusor
//beta(t)=betaMin_+(betaMax_-betaMin_)*t/timeMax_


class LinearDiffusor : public OUDiffusor{
public:
    LinearDiffusor(
        double betaMin,
        double betaMax,
        double timeMax
    );    
};






}
//end of namespace diffusion





