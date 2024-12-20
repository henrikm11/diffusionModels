//OUDiffusion.hpp

#pragma once

#include <type_traits>

namespace diffusion{

template<typename T>
using driftExpr= std::remove_reference_t<
    decltype(
        funcExpr::ReturnScalar<T>(T(-0.5)) 
        * funcExpr::ReturnScalar<T>(T(1.0))
        *(vec<T>({T(1.0)}))
        *funcExpr::ReturnScalar<T>(T(1.0))
    )
>;

template<typename T>
requires funcExpr::numerical<T>
class driftHelperClass{
public:
    driftHelperClass(T(* const varianceScheduleFct)(T)):varianceScheduleFct_(varianceScheduleFct){}
    
    driftExpr<T> operator()(
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




template<typename T>
decltype(auto) driftHelper(T (* const varianceScheduleFct)(T)){
    return funcExpr::FunctionObjectWrapper<
        driftHelperClass<T>,
        driftExpr<T>,
        const vec<T>&,
        T,
        T
        >
        (driftHelperClass<T>(varianceScheduleFct))
    ;
};

template<typename T>
using diffusionExpr =  funcExpr::ReturnOp<funcExpr::ReturnScalar<T>,vec<T>,funcExpr::multiply>;

template<typename T> 
requires funcExpr::numerical<T>
class diffusionHelperClass{
public:
    diffusionHelperClass(T (* const varianceScheduleFct)(T)):varianceScheduleFct_(varianceScheduleFct){}

    diffusionExpr<T> operator()(
        const vec<T>& X,
        T t,
        vec<T>& dW
    ) const {
        auto diff = funcExpr::ReturnScalar<T>(std::sqrt(varianceScheduleFct_(t)))
            *vec<T>(dW)
        ;
        return diff;
            
        
    }
private:
    T (*varianceScheduleFct_)(T);
};


template<typename T>
funcExpr::FunctionObjectWrapper<
        diffusionHelperClass<T>,
        diffusionExpr<T>,
        const vec<T>&,
        T,
        vec<T>&
        > 
diffusionHelper(T (* const varianceScheduleFct)(T)){
    return funcExpr::FunctionObjectWrapper<
        diffusionHelperClass<T>,
        diffusionExpr<T>,
        const vec<T>&,
        T,
        vec<T>&
        >
        (diffusionHelperClass<T>(varianceScheduleFct))
    ;
};




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
    :GeneralDiffusor<T>(
        driftHelper<T>(varianceScheduleFct),
        diffusionHelper<T>(varianceScheduleFct),
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
    return std::move(factor1)*std::move(inputState_copy)+std::move(factor2)*(GeneralDiffusor<T>::getRandomVector(GeneralDiffusor<T>::noiseDim_));
}




} //end of namespace diffusion