//VectorFuncHelper.cpp
//implementation of diffusion::VectorFuncHelper


#include "FuncHelper.h"

namespace diffusion{

VectorFuncHelper::VectorFuncHelper(
    std::vector<double> (*funcVec)(const std::vector<double>&, double), 
    bool multiply,
    bool integral,
    double factor,
    double power
)
    :FuncHelper(
        multiply,
        integral,
        factor,
        power
    ),
    funcVec_(funcVec)
{
    if(funcVec==nullptr) throw std::invalid_argument("VectorFuncHelper needs non null pointer for instantiation");
}


//copy and modify

//copy and modify
VectorFuncHelper::VectorFuncHelper(
    const VectorFuncHelper& other,
    double factor,
    double power
)
    :FuncHelper(
        other.multiply_,
        other.integral_,
        other.factor_*factor,
        other.power_*power
    ),
    funcVec_(other.funcVec_)
{}

VectorFuncHelper::VectorFuncHelper(
    const VectorFuncHelper& other,
    bool multiply,
    bool integral,
    double factor,
    double power
)
    :FuncHelper(
        multiply,
        integral,
        other.factor_*factor,
        other.power_*power
    ),
    funcVec_(other.funcVec_)
{}







//clone methods
std::unique_ptr<FuncHelper> VectorFuncHelper::clone(void) const {
    return std::unique_ptr<FuncHelper>(new VectorFuncHelper(*this));
}

std::unique_ptr<FuncHelper> VectorFuncHelper::modifiedClone(
    double factor, 
    double power
) const {
    return std::unique_ptr<FuncHelper>(
        new VectorFuncHelper(
            *this,
            factor, 
            power
        )
    );
}

std::unique_ptr<FuncHelper> VectorFuncHelper::modifiedClone(
    bool multiply,
    bool integral, 
    double factor, 
    double power
) const {
    return std::unique_ptr<FuncHelper>(
        new VectorFuncHelper(
            *this,
            multiply,
            integral,
            factor,
            power
        )
    );
}



//operators()
double VectorFuncHelper::operator()(double x, double t) const {
    std::vector<double> X = {x};
    std::vector<double> resVec = funcVec_(X,t);
    return adjustResult_(resVec[0],x,t);
    //return resVec[0]; 
}

double VectorFuncHelper::operator()(double t) const {
    double res= operator()(1.0,t);
    return adjustResult_(res);
}


std::vector<double> VectorFuncHelper::operator()(const std::vector<double>& X, double t) const {
    std::vector<double> res = funcVec_(X,t);
    for(size_t i=0; i<res.size(); i++){
        res[i]=adjustResult_(res[i],X[i],t);
    }
    return res;
}


} //end of namespace diffusion
