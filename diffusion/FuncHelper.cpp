//FuncHelper.cpp

#include "FuncHelper.h"


namespace diffusion{


FuncHelper::FuncHelper(
    bool multiply,
    bool integral,
    double factor,
    double power
)
    :multiply_(multiply),
    integral_(integral),
    factor_(factor),
    power_(power)
{}

double FuncHelper::adjustResult_(double res) const {
        res = std::pow(res,power_);
        res*=factor_;
        return res;
}

double FuncHelper::adjustResult_(double res, double x, double t) const {
    res = std::pow(res, power_);
    res*=factor_;
    if(multiply_) res*=x;
    if(integral_) res*=t;
    return res;
}



//casts to SumFuncHelper
//if this is SumFuncHelper creates a copy
//otherwise creates a SumFuncHelper with summands_ containing only this
std::unique_ptr<SumFuncHelper> FuncHelper::convertToSum() const {
    const SumFuncHelper* casted = dynamic_cast<const SumFuncHelper*>(this);
    if(casted){
        return std::unique_ptr<SumFuncHelper>(new SumFuncHelper(*casted));
    }
    const std::vector<const FuncHelper*> summands={this};
    return std::unique_ptr<SumFuncHelper>(new SumFuncHelper(summands));
}


//casts to ProductFuncHelper
//if this is SumFuncHelper creates a copy
//otherwise creates a SumFuncHelper with factors_ containing only this
std::unique_ptr<ProductFuncHelper> FuncHelper::convertToProduct(void) const {
    const ProductFuncHelper* casted = dynamic_cast<const ProductFuncHelper*>(this);
    if(casted){
        return std::unique_ptr<ProductFuncHelper>(new ProductFuncHelper(*casted));
    }
    const std::vector<const FuncHelper*> factors={this};
    return std::unique_ptr<ProductFuncHelper>(new ProductFuncHelper(factors));
}



std::unique_ptr<SumFuncHelper> FuncHelper::add(const FuncHelper& other) const {
    std::unique_ptr<SumFuncHelper> res = this->convertToSum(); //this checks if this was a sum before
    res->addToSum(other);
    return res;
}

std::unique_ptr<ProductFuncHelper> FuncHelper::multiply(const FuncHelper& other) const {
    std::unique_ptr<ProductFuncHelper> res = this->convertToProduct();
    res->addToProduct(other);
    return res;
}


} //end of namespace diffusion