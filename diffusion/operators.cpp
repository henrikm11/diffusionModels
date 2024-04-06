//operators.cpp
//implementation of operators +,* for FuncHelper


#include "FuncHelper.h"


namespace diffusion{

std::unique_ptr<SumFuncHelper> operator+(
    const std::unique_ptr<FuncHelper>& lhs,
    const std::unique_ptr<FuncHelper>& rhs
){
    return lhs->add(*rhs);
}

std::unique_ptr<SumFuncHelper> operator+(
    const FuncHelper& lhs,
    const FuncHelper& rhs
){
    return lhs.add(rhs);
}



//product operations

std::unique_ptr<ProductFuncHelper> operator*(
    const FuncHelper& lhs,
    const FuncHelper& rhs
){
    return lhs.multiply(rhs);

}

std::unique_ptr<ProductFuncHelper> operator*(
    const std::unique_ptr<FuncHelper>& lhs,
    const std::unique_ptr<FuncHelper>& rhs
)
{
    return lhs->multiply(*rhs);
}

}