//diffusion.h
#pragma once

#include <random>
#include "../functional_expressions/FunctionalWrapper.hpp"

//TODO
/*
-) additional constructors for more user friendly interface
*/

namespace diffusion{

//base class for diffusion along
// dX = drift(X,t)dt + diffusion(X,t)dW_t,
// W_t standard Wiener process

template<typename T>
requires funcExpr::numerical<T>
using vec = funcExpr::ReturnVector<T>;

//for the following to typedefs note that any function expression eventually evaluating
//to vec<T> inherits from these abstract types
//this means that the members can actually be complicated function expressions 
//this has two consequences
//we are fully relying on lazy evaluation yielding performance gain
//when passing these to the constructor they can be any function expression with BaseReturnExpressionType being vec<T>
template<typename T>
requires funcExpr::numerical<T>
using driftFuncExpr = funcExpr::AbstractFunction<
    vec<T>,
    const vec<T>&, //X
    T, //t
    T   //dt
>;
//fully evaluating drift : drift = \mu(X,t)*dt and 


template<typename T>
requires funcExpr::numerical<T>
using diffusionFuncExpr = funcExpr::AbstractFunction<
    vec<T>,
    const vec<T>&, //X
    T,              //t
    vec<T>&       //dW
>;
//fully evaluating diffusion = \sigma(X,t)*dW_t respectively


template<typename T>
requires funcExpr::numerical<T>
class GeneralDiffusor{
public:
    
    GeneralDiffusor(
        const driftFuncExpr<T>& driftFct,
        const diffusionFuncExpr<T>& diffusionFct,
        int noiseDim=-1
    );

    //copy constructor needs to be custom to call modifiedClone FuncHelper
    //since FuncHelper has deleted copy constructor
    GeneralDiffusor(const GeneralDiffusor& other); //clone polymorphic members

    //not default because of unique_ptr member variables
    GeneralDiffusor& operator=(const GeneralDiffusor& other);
    
    
    //default ok because we use smart pointers for member variables
    virtual ~GeneralDiffusor(void) = default; 

    //clone for polymorphic copying
    virtual std::unique_ptr<GeneralDiffusor> clone(void) const;

    /// @brief returns pseudo random of type T number ~ N(0,1)
    T getRandomNormal(void);

    /// @brief returns pseudo random number of type T ~ Unif[low,high]
    T getRandomUnif(T low, T high);
    
    /// @brief returns pseudo random vector of size noiseDim
    vec<T> getRandomVector(int noiseDim, T scale = 1);



    /// @brief diffusion along dY = \mu(Y,t)*dt +  \sigma(Y,t)*dW_t
    /// @param inputState X(time)
    /// @param time 
    /// @param timeStep compute X(time+timestep)
    /// @return Y(time+timestep)
    vec<T> diffusionStep(
        const vec<T>& inputState,
        T time, 
        T timeStep
    );

    //functions to sample from process
    //virtual because this can be sped up significantly for more specific processes
    
    /// @brief returns state of sample path at time starting from inputState
    /// @param inputState 
    /// @param time 
    /// @param timeStepSize accuracy in sample path
    virtual vec<T> sample(
        const vec<T>& inputState,
        T time,
        T dt=1e-2
    );

    /// @brief samples a path up to time starting from inputState
    /// @param inputState 
    /// @param time 
    /// @param timeStepSize accuracy in sample path
    /// @return vector of pairs (time, state at time)
    std::vector<std::pair<T,vec<T>>> samplePath(
        const vec<T>& inputState,
        T time,
        T timeStepSize=1e-2
    );

   
protected:

    std::mt19937 rng_;
    std::unique_ptr<driftFuncExpr<T>> driftFct_;  //polymophic type
    std::unique_ptr<diffusionFuncExpr<T>> diffusionFct_; //polymophic type
    int noiseDim_; 
    //noise is given by diffusion matrix multiplied with noiseDim_ dimensional brownian motion
    //if default value -1 uses dimension of states for noiseDim_ 
};


//
//END GeneralDiffusor
//



//
//BEGIN OUDiffusor
//


//derived class for Ornstein Uhlenbeck like processes
// dX = -0.5*beta(t)*X(t)*dt + \sqrt(beta(t))*dW_t,
// beta(t) is fixed variance schedule


template<typename T>
requires funcExpr::numerical<T>
class OUDiffusor : public GeneralDiffusor<T>{
public:
    OUDiffusor(
        T (* const varianceScheduleFct)(T), //beta
        T (* const varianceScheduleIntegralFct)(T),  //int_0^x \beta,
        int dim
    );


    vec<T> sample(
        const vec<T>& inputState,
        T time,
        T dt = 1e-2
    ) override;

 

private:

    //std::unique_ptr<funcExpr::AbstractFunction<funcExpr::ReturnScalar<T>,T>>;
    T (*const varianceScheduleIntegralFct_)(T);
};


} //end of namespace diffusion


#include "generalDiffusion.hpp"
#include "OUDiffusion.hpp"

