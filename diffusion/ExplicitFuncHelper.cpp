//ExlicitFuncHelper.cpp
//implementation of diffusion::ExplicitFuncHelper

#include "FuncHelper.h"


namespace diffusion{

//constructors
ExplicitFuncHelper::ExplicitFuncHelper(
    double betaMin,
    double betaMax,
    double timeMax,
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
    betaMin_(betaMin),
    betaMax_(betaMax),
    timeMax_(timeMax)
{
    if(betaMin>=betaMax) throw std::invalid_argument("ExplicitFuncHelper needs betaMin<betaMax");
    if(timeMax<=0) throw std::invalid_argument("ExplicitFuncHelper needs timeMax>0");
    if(integral){
        betaMax_+=betaMin_;
        betaMin_*=2;
    }
    return;
}

//copy and modify
ExplicitFuncHelper::ExplicitFuncHelper(
    const ExplicitFuncHelper& other,
    bool multiply, 
    bool integral,
    double factor,
    double power
)
    :FuncHelper(
        multiply,
        integral,
        factor*other.factor_,
        power*other.power_
    ),
    betaMin_(other.betaMin_),
    betaMax_(other.betaMax_),
    timeMax_(other.timeMax_)
{}


    //clone methods

std::unique_ptr<FuncHelper> ExplicitFuncHelper::clone(void) const {
    return std::unique_ptr<FuncHelper>(new ExplicitFuncHelper(*this));
}

std::unique_ptr<FuncHelper> ExplicitFuncHelper::modifiedClone(
    double factor, 
    double power
) const {
    return std::unique_ptr<FuncHelper>(
        new ExplicitFuncHelper(
            *this,
            false,
            false,
            factor,
            power
        )
    );
}
    
std::unique_ptr<FuncHelper> ExplicitFuncHelper::modifiedClone(
    bool multiply,
    bool integral,
    double factor,
    double power
) const {
    return std::unique_ptr<FuncHelper> (
        new ExplicitFuncHelper(
            *this,
            multiply, 
            integral,
            factor,
            power
        )
    );
}


    //operators()

double ExplicitFuncHelper::operator()(double t) const {
    double res = explicitFctEval_(t);
    res = adjustResult_(res);
    return res;
}

double ExplicitFuncHelper::operator()(double x, double t) const {
    double res = explicitFctEval_(t);
    res = adjustResult_(res,x,t);
    return res;
}

std::vector<double> ExplicitFuncHelper::operator()(const std::vector<double>& X, double t) const {
    std::vector<double> res(X.size(),0);
    for(size_t i=0; i<X.size(); i++) res[i]=operator()(X[i],t);
    return res;
}

double ExplicitFuncHelper::explicitFctEval_(double t) const {
    double res = 0;
    res = betaMin_ + (betaMax_-betaMin_)*t/timeMax_;
    return res;
}


} //end of namespace diffusion