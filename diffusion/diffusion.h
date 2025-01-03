//diffusion.h
#pragma once

#include <random>
#include <math.h>
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

template<typename F>
requires funcExpr::isFunctionExpression<F,const vec<typename F::DataType>&, typename F::DataType, typename F::DataType>
using driftFuncExpr = funcExpr::FunctionExpression<
    F, //delegates calls to F
    typename F::ReturnExpressionType,
    const vec<typename F::DataType>&, //X
    typename F::DataType, //t
    typename F::DataType   //dt
>;
//fully evaluating drift : drift = \mu(X,t)*dt as a return expression

template<typename F>
requires funcExpr::isFunctionExpression<F,const vec<typename F::DataType>&, typename F::DataType,  vec<typename F::DataType>&>
using diffusionFuncExpr = funcExpr::FunctionExpression<
    F, //delegates calls to F
    typename F::ReturnExpressionType,
    const vec<typename F::DataType>&, //X
    typename F::DataType, //t
    vec<typename F::DataType>&   //dW
>;
//fully evaluating diffusion = \sigma(X,t)*dW_t respectively

template<typename F1, typename F2>
concept diffusorPair =
    funcExpr::isFunctionExpression<F1, const vec<typename F1::DataType>&, typename F1::DataType, typename F1::DataType>
    &&
    funcExpr::isFunctionExpression<F2, const vec<typename F1::DataType>&, typename F1::DataType,  vec<typename F1::DataType>&>
;
//note that this also ensures that the numerical data type is the same!

//could do this:
//template<typename T, funcExpr::isFunctionExpr driftFct, funcExpr::isFunctionExpr diffusionFct>
//couple with factory method that does type deduction
template<typename F1, typename F2>
requires  diffusorPair<F1,F2>
class GeneralDiffusor{
public:

    using numericalType = typename F1::DataType;
    
    GeneralDiffusor(
        const driftFuncExpr<F1>& driftFct,
        const diffusionFuncExpr<F2>& diffusionFct,
        int noiseDim=-1
    );

    //copy constructor needs to be custom to call modifiedClone FuncHelper
    //since FuncHelper has deleted copy constructor
    GeneralDiffusor(const GeneralDiffusor& other) = default; //clone polymorphic members

    //not default because of unique_ptr member variables
    GeneralDiffusor& operator=(const GeneralDiffusor& other) = default;
    
    
    //default ok because we use smart pointers for member variables
    virtual ~GeneralDiffusor(void) = default;

    //clone for polymorphic copying
    //virtual std::unique_ptr<GeneralDiffusor> clone(void) const;

    /// @brief returns pseudo random of type T number ~ N(0,1)
    numericalType getRandomNormal(void);

    /// @brief returns pseudo random number of type T ~ Unif[low,high]
    numericalType getRandomUnif(numericalType low, numericalType high);
    
    /// @brief returns pseudo random vector of size noiseDim
    vec<numericalType> getRandomVector(int noiseDim, numericalType scale = 1.0);



    /// @brief diffusion along dY = \mu(Y,t)*dt +  \sigma(Y,t)*dW_t
    /// @param inputState X(time)
    /// @param time
    /// @param timeStep compute X(time+timestep)
    /// @return Y(time+timestep)
    vec<numericalType> diffusionStep(
        const vec<numericalType>& inputState,
        numericalType time,
        numericalType timeStep
    );

    //functions to sample from process
    //virtual because this can be sped up significantly for more specific processes
    
    /// @brief returns state of sample path at time starting from inputState
    /// @param inputState
    /// @param time
    /// @param timeStepSize accuracy in sample path
    virtual vec<numericalType> sample(
        const vec<numericalType>& inputState,
        numericalType time,
        numericalType dt=1e-2
    );

    /// @brief samples a path up to time starting from inputState
    /// @param inputState
    /// @param time
    /// @param timeStepSize accuracy in sample path
    /// @return vector of pairs (time, state at time)
    std::vector<std::pair<numericalType,vec<numericalType>>> samplePath(
        const vec<numericalType>& inputState,
        numericalType time,
        numericalType timeStepSize=1e-2
    );

   
protected:

    std::mt19937 rng_;
    //std::unique_ptr<driftFuncExpr<T>> driftFct_;  //polymophic type
    //std::unique_ptr<diffusionFuncExpr<T>> diffusionFct_; //polymophic type
    F1 driftFct_;
    F2 diffusionFct_;
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

//
// Ultilities for OUDiffusor
//


//drift

// -0.5*beta(t)*X(t)*dt
template<typename T>
using OUDriftExpr= std::remove_reference_t<
    decltype(
        funcExpr::ReturnScalar<T>(T(-0.5))
        * funcExpr::ReturnScalar<T>(T(1.0))
        *(vec<T>({T(1.0)}))
        *funcExpr::ReturnScalar<T>(T(1.0))
    )
>;

template<typename T>
requires funcExpr::numerical<T>
class OUDriftHelperClass{
public:
    OUDriftHelperClass(T(* const varianceScheduleFct)(T)):varianceScheduleFct_(varianceScheduleFct){}
    
    OUDriftExpr<T> operator()(
        const vec<T>& X,
        T t,
        T dt
    ) const {
        return
            funcExpr::ReturnScalar<T>(T(-0.5))
            * funcExpr::ReturnScalar<T>(varianceScheduleFct_(t))
            *(vec<T>(X))*
            funcExpr::ReturnScalar(dt)
        ;
    }

private:
    T (* const varianceScheduleFct_)(T);
};


//we can remove one indirection by implementing a custom wrapper inherting from FunctionExpression
template<typename T>
using OUDriftFuncExpr = funcExpr::FunctionObjectWrapper<
    OUDriftHelperClass<T>, //object we delegate to
    OUDriftExpr<T>,       //type of return expression it returns
    const vec<T>&, T, T
>;

//diffusion

// \sqrt(beta(t))*dW_t,
template<typename T>
using OUDiffusionExpr = std::remove_reference_t<
    decltype(
        funcExpr::ReturnScalar<T>(T(1.0))*funcExpr::ReturnVector<T>({T(1.0)})
    )
>;

template<typename T>
requires funcExpr::numerical<T>
class OUDiffusionHelperClass{
public:
    OUDiffusionHelperClass(T(* const varianceScheduleFct)(T)):varianceScheduleFct_(varianceScheduleFct){}
    
    OUDiffusionExpr<T> operator()(
        const vec<T>& X,
        T t,
        vec<T>& dW
    ) const {
        return
            funcExpr::ReturnScalar<T>(sqrt(varianceScheduleFct_(t)))
            * dW
        ;
    }

private:
    T (* const varianceScheduleFct_)(T);
};

//we can remove one indirection by implementing a custom wrapper inherting from FunctionExpression
template<typename T>
using OUDiffusionFuncExpr = funcExpr::FunctionObjectWrapper<
    OUDiffusionHelperClass<T>, //object we delegate to
    OUDiffusionExpr<T>,       //type of return expression it returns
    const vec<T>&, T, vec<T>&
>;


//
// END of Ultilities for OUDiffusor
//


//derived class for Ornstein Uhlenbeck like processes
// dX = -0.5*beta(t)*X(t)*dt + \sqrt(beta(t))*dW_t,
// beta(t) is fixed variance schedule


template<typename T>
requires funcExpr::numerical<T>
class OUDiffusor : public GeneralDiffusor<OUDriftFuncExpr<T>,OUDiffusionFuncExpr<T>>{
public:

    using BaseDiffusor = GeneralDiffusor<OUDriftFuncExpr<T>,OUDiffusionFuncExpr<T>>;
    
    OUDiffusor(
            T (* const varianceScheduleFct)(T), //beta
            T (* const varianceScheduleIntegralFct)(T),  //int_0^x \beta,
            int dim // stored in GeneralDiffusor::noiseDim
    );

    vec<T> sample(
        const vec<T>& inputState,
        T time,
        T dt = 1e-2
    ) override;


private:
    T (*const varianceScheduleIntegralFct_)(T);
};

//
//END GeneralDiffusor
//

//
//BEGIN LinearDiffusor
//




//
//END LinearDiffusor
//



} //end of namespace diffusion


#include "generalDiffusion.hpp"
#include "OUDiffusion.hpp"

