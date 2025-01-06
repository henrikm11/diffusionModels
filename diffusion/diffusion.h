//diffusion.h
#pragma once

#include <random>
#include <math.h>
#include "../functional_expressions/FunctionalWrapper.hpp"

//TODO
/*
-) add requirements for template parameter F in OUDiffusor
-) remove indirections in OUDiffusor?
*/

namespace diffusion{

//base class for diffusion along
// dX = drift(X,t)dt + diffusion(X,t)dW_t,s
// W_t standard Wiener process

template<typename T>
requires funcExpr::numerical<T>
using vec = funcExpr::ReturnVector<T>;


//this looks complicated but just says that the drift in SDE solver is evaluated by
// (X,t,dt) -> mu(X,t)*dt
// everything is a return expression
// and this is evaluated by delegating the call to F
template<typename F>
requires funcExpr::isFunctionExpression<F,const vec<typename F::DataType>&, typename F::DataType, typename F::DataType>
using driftFuncExpr = funcExpr::FunctionExpression<
    F, //delegates calls to F
    typename F::ReturnExpressionType, //whatever return expression we build in F
    const vec<typename F::DataType>&, //X
    typename F::DataType, //t
    typename F::DataType   //dt
>;


//same as for drift, this time evaluating diffusion
// (X,t,dW_t) \to \sigma(X,t)*dW_t
template<typename F>
requires funcExpr::isFunctionExpression<F,const vec<typename F::DataType>&, typename F::DataType,  vec<typename F::DataType>&>
using diffusionFuncExpr = funcExpr::FunctionExpression<
    F, //delegates calls to F
    typename F::ReturnExpressionType,
    const vec<typename F::DataType>&, //X
    typename F::DataType, //t
    vec<typename F::DataType>&   //dW
>;


//ensures that F1 and F2 have correct signatures
//note that this also ensures that the numerical data type is the same!
template<typename F1, typename F2>
concept diffusorPair =
    funcExpr::isFunctionExpression<F1, const vec<typename F1::DataType>&, typename F1::DataType, typename F1::DataType>
    &&
    funcExpr::isFunctionExpression<F2, const vec<typename F1::DataType>&, typename F1::DataType,  vec<typename F1::DataType>&>
;



//
//BEGIN GeneralDiffusor
//

//general class for diffusion along
// dX_t = mu(X_t,t)dt + sigma(X_t,t)dW_t
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

    GeneralDiffusor(const GeneralDiffusor& other) = default;
    GeneralDiffusor& operator=(const GeneralDiffusor& other) = default;
    
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


template<typename F, typename T> //F is callable T->T
requires funcExpr::numerical<T>
class OUDriftHelperClass{
public:
    OUDriftHelperClass(F varianceScheduleFct):varianceScheduleFct_(varianceScheduleFct){}
    
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
    F varianceScheduleFct_;
};

//we can remove one indirection by implementing a custom wrapper inherting from FunctionExpression
template<typename F, typename T>
using OUDriftFuncExpr = funcExpr::FunctionObjectWrapper<
    OUDriftHelperClass<F,T>, //object we delegate to
    OUDriftExpr<T>,       //type of return expression it returns
    const vec<T>&, T, T //arguments
>;

//diffusion

// \sqrt(beta(t))*dW_t,
template<typename T>
using OUDiffusionExpr = std::remove_reference_t<
    decltype(
        funcExpr::ReturnScalar<T>(T(1.0))*funcExpr::ReturnVector<T>({T(1.0)})
    )
>;

template<typename F, typename T>
requires funcExpr::numerical<T>
class OUDiffusionHelperClass{
public:
    OUDiffusionHelperClass(F varianceScheduleFct):varianceScheduleFct_(varianceScheduleFct){}
    
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
    F varianceScheduleFct_;
};

//we can remove one indirection by implementing a custom wrapper inherting from FunctionExpression
template<typename F, typename T>
using OUDiffusionFuncExpr = funcExpr::FunctionObjectWrapper<
    OUDiffusionHelperClass<F,T>, //object we delegate to
    OUDiffusionExpr<T>,       //type of return expression it returns
    const vec<T>&, T, vec<T>&
>;



//
// END of Ultilities for OUDiffusor
//


//derived class for Ornstein Uhlenbeck like processes
// dX = -0.5*beta(t)*X(t)*dt + \sqrt(beta(t))*dW_t,
// beta(t) is fixed variance schedule

//template this more generally
//to have callable F varianceScheduleIntegralFct_
// so that we can simply overload constructor
//type erasure would add yet another indirection
//template everything above to F
template<typename T, typename F1 = T(*)(T), typename F2 = T(*)(T)>
class OUDiffusor : public GeneralDiffusor<OUDriftFuncExpr<F1,T>,OUDiffusionFuncExpr<F2,T>> {
public:
    using BaseDiffusor = GeneralDiffusor<OUDriftFuncExpr<F1,T>,OUDiffusionFuncExpr<F2,T>>;
    OUDiffusor(
            F1 varianceScheduleFct, //beta
            F2  varianceScheduleIntegralFct,  //int_0^x \beta,
            int dim // stored in GeneralDiffusor::noiseDim
    );

    vec<T> sample(
        const vec<T>& inputState,
        T time,
        T dt = 1e-2
    ) override;

private:

    F2 varianceScheduleIntegralFct_; //needs to be callable T -> T, can be lambda too
};

template<typename T>
struct OUDiffusorFactory{
    template<typename F1 = T(*)(T), typename F2 = T(*)(T)>
    static OUDiffusor<T,F1,F2> createDiffusor(
        F1 varianceScheduleFct, //beta
        F2  varianceScheduleIntegralFct,  //int_0^x \beta,
        int dim // stored in GeneralDiffusor::noiseDim
    );
    
    //creates diffusor with vatianceSchedule beta(t)= betaMin+(betaMax-betaMin)*t/timeMax
    static decltype(auto) createLinearDiffusor(
        T betaMin,
        T betaMax,
        T timeMax,
        int dim
    );

};

template<typename T>
decltype(auto) createLinearDiffusor(
    T betaMin,
    T betaMax,
    T timeMax,
    int dim
);





//
//END OUDiffusor
//


} //end of namespace diffusion


#include "generalDiffusion.hpp"
#include "OUDiffusion.hpp"

