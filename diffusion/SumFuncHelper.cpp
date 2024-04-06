//SumFuncHelper.cpp


#include "FuncHelper.h"

namespace diffusion{


//makes copies of summands now owned by it
SumFuncHelper::SumFuncHelper(const std::vector<const FuncHelper*>& summands){
    for(const FuncHelper* f : summands) summands_.emplace_back(f->clone());
}




//copy 
SumFuncHelper::SumFuncHelper(
    const SumFuncHelper& other,
    bool multiply,
    bool integral,
    double factor,
    double power
):FuncHelper(
    multiply,
    integral,
    other.factor_*factor,
    other.power_*power
)
{
    for(const auto& f : other.summands_) summands_.emplace_back(f->clone());
}

SumFuncHelper::SumFuncHelper(
    const SumFuncHelper& other,
    double factor,
    double power
):FuncHelper(
    other.multiply_,
    other.integral_,
    other.factor_*factor,
    other.power_*power
){
    for(const auto& f : other.summands_) summands_.emplace_back(f->clone());
}



    
//copy assignment
SumFuncHelper& SumFuncHelper::operator=(const SumFuncHelper& other){
    if(this==&other) return *this;
    summands_.clear();
    for(const auto& f : other.summands_) summands_.emplace_back(f->clone());
    return *this;
}
    




   

//clone methods

std::unique_ptr<FuncHelper> SumFuncHelper::clone(void) const{
    return std::unique_ptr<FuncHelper>(new SumFuncHelper(*this));
}

std::unique_ptr<FuncHelper> SumFuncHelper::modifiedClone(
    double factor,
    double power
) const {
    return std::unique_ptr<FuncHelper>(new SumFuncHelper(*this, factor, power));
}

std::unique_ptr<FuncHelper> SumFuncHelper::modifiedClone(
    bool multiply, 
    bool integral,
    double factor, 
    double power
) const {
    return std::unique_ptr<FuncHelper>(new SumFuncHelper(*this, multiply, integral, factor, power));
}


//operators()
double SumFuncHelper::operator()(double x, double t) const {
    double res=0;
    for(const auto& f : summands_) res+=(f->operator()(x,t));
    return adjustResult_(res,x,t);
}


double SumFuncHelper::operator()(double t) const {
    double res=0;
    for(const auto& f : summands_) res+=(f->operator()(t));
    return adjustResult_(res);
}



std::vector<double> SumFuncHelper::operator()(const std::vector<double>& X, double t) const {
    std::vector<double> res(X.size(),0);
    for(const auto& f : summands_){
        std::vector<double> term = f->operator()(X,t);
        for(size_t i=0; i<X.size(); i++){
            res[i]+=term[i];
        }
    }
        for(size_t i=0; i<X.size(); i++){
            res[i]=adjustResult_(res[i],X[i],t);
    }
    return res;
}




SumFuncHelper& SumFuncHelper::addToSum(const FuncHelper& newSummand){
    const SumFuncHelper* casted = dynamic_cast<const SumFuncHelper*>(&newSummand);
    if(casted){
        //keep things shallow, copy summands of newSummand
        for(const auto& f : casted->summands_){
            summands_.emplace_back(f->clone());
        }
        return *this;
    }
    summands_.emplace_back(newSummand.clone());
    return *this;
}

} //end of namespace diffusion
