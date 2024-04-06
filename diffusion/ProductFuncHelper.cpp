//ProductFuncHelper.cpp

#include "FuncHelper.h"

namespace diffusion{

//makes copies of factors now owned by this
ProductFuncHelper::ProductFuncHelper(const std::vector<const FuncHelper*>& factors){
    for(const FuncHelper* f : factors) factors_.emplace_back(f->clone());
}


//copy
ProductFuncHelper::ProductFuncHelper(
    const ProductFuncHelper& other,
    double factor,
    double power
)
    :FuncHelper(
        other.multiply_,
        other.integral_,
        other.factor_*factor,
        other.power_*power
    )
{
    for(const auto& f : other.factors_) factors_.emplace_back(f->clone());
}

ProductFuncHelper::ProductFuncHelper(
    const ProductFuncHelper& other,
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
    )
{
    for(const auto& f : other.factors_) factors_.emplace_back(f->clone());
}



//copy assignment
ProductFuncHelper& ProductFuncHelper::operator=(const ProductFuncHelper& other){
    if(this==&other) return *this;
    factors_.clear();
    for(const auto& f : other.factors_) factors_.emplace_back(f->clone());
    return *this;
}


//clone methods

std::unique_ptr<FuncHelper> ProductFuncHelper::clone(void) const{
    return std::unique_ptr<FuncHelper>(new ProductFuncHelper(*this));
}

std::unique_ptr<FuncHelper> ProductFuncHelper::modifiedClone(
    double factor,
    double power
) const {
    return std::unique_ptr<FuncHelper>(new ProductFuncHelper(*this, factor, power));
}

std::unique_ptr<FuncHelper> ProductFuncHelper::modifiedClone(
    bool multiply,
    bool integral,
    double factor,
    double power
) const {
    return std::unique_ptr<FuncHelper>(new ProductFuncHelper(*this, multiply, integral, factor, power));
}

//operators()

double ProductFuncHelper::operator()(double x, double t) const {
    double res=1;
    for(const auto& f : factors_) res*=f->operator()(x,t);
    return adjustResult_(res,x,t);
}

double ProductFuncHelper::operator()(double t) const {
    double res=1;
    for(const auto& f : factors_) res*=f->operator()(t);
    return adjustResult_(t);
}

std::vector<double> ProductFuncHelper::operator()(const std::vector<double>& X, double t) const {
    std::vector<double> res(X.size(),0);
    for(const auto& f : factors_){
        std::vector<double> term = f->operator()(X,t);
        for(size_t i=0; i<X.size(); i++){
            res[i]*=term[i];
        }
    }
        for(size_t i=0; i<X.size(); i++){
        res[i]=adjustResult_(res[i],X[i],t);
    }
    return res;
}

//add a new factor

ProductFuncHelper& ProductFuncHelper::addToProduct(const FuncHelper& newFactor){
    const ProductFuncHelper* casted = dynamic_cast<const ProductFuncHelper*>(&newFactor);
    if(casted){
        //keep things shallow, add factors of other to factors_
        for(const auto& f : casted->factors_) factors_.emplace_back(f->clone());
        return *this;
    }
    factors_.emplace_back(newFactor.clone());
    return *this;
}

}//end of namespace diffusion
