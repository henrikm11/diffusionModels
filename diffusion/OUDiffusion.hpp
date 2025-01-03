//OUDiffusion.hpp

#pragma once

#include <type_traits>
#include "diffusion.h"

namespace diffusion{

//
//BEGIN OUDiffusor
//

template<typename T>
requires funcExpr::numerical<T>
OUDiffusor<T>::OUDiffusor(
    T (* const varianceScheduleFct)(T), //beta
    T (* const varianceScheduleIntegralFct)(T),  //int_0^x \beta
    int dim
    )
    :BaseDiffusor(
        OUDriftFuncExpr<T>(varianceScheduleFct),
        OUDiffusionFuncExpr<T>(varianceScheduleFct),
        dim
    ),
    varianceScheduleIntegralFct_(varianceScheduleIntegralFct)
{}


template<typename T>
requires funcExpr::numerical<T>
vec<T> OUDiffusor<T>::sample(
    const vec<T>& inputState,
    T time,
    T dt //unused, but required for override
)
{
    
    vec<T> inputState_copy = inputState;
    auto factor1 = funcExpr::ReturnScalar(std::exp(T(-0.5)*(*varianceScheduleIntegralFct_)(time)));
    auto factor2 = funcExpr::ReturnScalar(T(1.0)-std::exp(-(*varianceScheduleIntegralFct_)(time))); //two copies so we can move them
    return std::move(factor1)*std::move(inputState_copy)+std::move(factor2)*(BaseDiffusor::getRandomVector(BaseDiffusor::noiseDim_));
}




} //end of namespace diffusion
