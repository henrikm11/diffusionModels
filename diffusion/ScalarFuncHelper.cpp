//ScalarFuncHelper.cpp
//non inline functions from diffusion::ScalarFuncHelper

#include "FuncHelper.h"



namespace diffusion{
/*
//ScalarFuncHelper
*/

//function in one argument
ScalarFuncHelper::ScalarFuncHelper(
    double (* const func1)(double),
    bool multiply, 
    bool integral, 
    double factor, 
    double power
) 
    :FuncHelper(multiply, integral, factor, power),
    func1_(func1),
    func2_(nullptr)
{
    if(func1==nullptr) throw std::invalid_argument("FuncHelper needs non null pointer for instantiation");
}


ScalarFuncHelper::ScalarFuncHelper(
    double (*func2)(double,double),
    bool multiply, 
    bool integral, 
    double factor, 
    double power
)
    :FuncHelper(multiply, integral, factor, power),
    func1_(nullptr),
    func2_(func2)
{ 
    if(func2==nullptr) throw std::invalid_argument("FuncHelper needs non null pointer for instantiation");
}

ScalarFuncHelper::ScalarFuncHelper(
    const ScalarFuncHelper& other,
    double factor,
    double power
)
    :FuncHelper(
        other.multiply_,
        other.integral_,
        other.factor_*factor,
        other.power_*power
    ),
    func1_(other.func1_),
    func2_(other.func2_)
{}

ScalarFuncHelper::ScalarFuncHelper(
    const ScalarFuncHelper& other,
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
    func1_(other.func1_),
    func2_(other.func2_)
{}

ScalarFuncHelper& ScalarFuncHelper::operator=(const ScalarFuncHelper& other){
        FuncHelper::operator=(other); 
        func1_=other.func1_;
        func2_=other.func2_;
        return *this;
}

std::unique_ptr<FuncHelper> ScalarFuncHelper::clone(void) const {
    return std::unique_ptr<FuncHelper>(new ScalarFuncHelper(*this));
}

std::unique_ptr<FuncHelper> ScalarFuncHelper::modifiedClone (
    double factor,
    double power
) const {
    return std::unique_ptr<FuncHelper>(
        new ScalarFuncHelper(
            *this,
            factor,
            power
        )
    );
}


std::unique_ptr<FuncHelper> ScalarFuncHelper::modifiedClone(
    bool multiply,
    bool integral, 
    double factor, 
    double power
) const {
    return std::unique_ptr<FuncHelper>(
        new ScalarFuncHelper(
            *this,
            multiply,
            integral,
            factor,
            power
        )
    );
}


double ScalarFuncHelper::operator()(double t) const {
    if(func2_!=nullptr) return func2_(1.0,t);
    double res = func1_(t);
    res = adjustResult_(res);
    return res;
}

double ScalarFuncHelper::operator()(double x, double t) const {
    if(func2_!=nullptr) return func2_(x,t); 
    double res = func1_(t);
    res = adjustResult_(res,x,t);
    return res;
}

//apply operator() elementwise
std::vector<double> ScalarFuncHelper::operator()(const std::vector<double>& X, double t) const {
    std::vector<double> res(X.size(),0);
    for(size_t i=0; i<res.size(); i++) res[i]=operator()(X[i],t);
    return res;
}


} //end of namespace diffusion