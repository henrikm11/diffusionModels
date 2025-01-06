//OUDiffusion.hpp

#pragma once

#include <type_traits>
#include "diffusion.h"

namespace diffusion{

//
//BEGIN OUDiffusor
//

template<typename T, typename F1, typename F2>
OUDiffusor<T,F1,F2>::OUDiffusor(
    F1 varianceScheduleFct, //beta
    F2  varianceScheduleIntegralFct,  //int_0^x \beta,
    int dim
    )
    :BaseDiffusor(
        OUDriftFuncExpr<F1,T>(OUDriftHelperClass<F1,T>(varianceScheduleFct)),
        OUDiffusionFuncExpr<F2,T>(OUDiffusionHelperClass<F2,T>(varianceScheduleIntegralFct)),
        dim
    ),
    varianceScheduleIntegralFct_(varianceScheduleIntegralFct)
{}


template<typename T, typename F1, typename F2>
vec<T> OUDiffusor<T,F1, F2>::sample(
    const vec<T>& inputState,
    T time,
    T dt //unused, but required for override
)
{
    
    vec<T> inputState_copy = inputState;
    auto factor1 = funcExpr::ReturnScalar(std::exp(T(-0.5)*(varianceScheduleIntegralFct_)(time)));
    auto factor2 = funcExpr::ReturnScalar(T(1.0)-std::exp(-(varianceScheduleIntegralFct_)(time))); //two copies so we can move them
    return std::move(factor1)*std::move(inputState_copy)+std::move(factor2)*(BaseDiffusor::getRandomVector(BaseDiffusor::noiseDim_));
}

template<typename T>
template<typename F1, typename F2>
OUDiffusor<T,F1,F2> OUDiffusorFactory<T>::createDiffusor(
    F1 varianceScheduleFct, //beta
    F2  varianceScheduleIntegralFct,  //int_0^x \beta,
    int dim // stored in GeneralDiffusor::noiseDim
){
    return OUDiffusor<T,F1,F2>(varianceScheduleFct, varianceScheduleIntegralFct, dim);
}

template<typename T>
decltype(auto) OUDiffusorFactory<T>::createLinearDiffusor(
    T betaMin,
    T betaMax,
    T timeMax,
    int dim
){
    return createDiffusor(
        [=](T t){return betaMin+(betaMax-betaMin)*t/timeMax;},
        [=](T t){return betaMin*t+(betaMax-betaMin)*t*t/(2*timeMax);},
        dim
    );
}

template<typename T>
decltype(auto) createLinearDiffusor(
    T betaMin,
    T betaMax,
    T timeMax,
    int dim
){
    return OUDiffusorFactor<T>::createLinearDiffusor(betaMin, betaMax, timeMax, dim);
}

} //end of namespace diffusion
