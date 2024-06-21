//operators.cpp
//implementation of operators +,* for FuncHelper


#include "FuncHelper.h"


namespace diffusion{

//change return types???!!! why are these unique ptrs?
//return type is polymorphic, but we know exact type in all these cases


//casts to SumFuncHelper
//if this is SumFuncHelper creates a copy
//otherwise creates a SumFuncHelper with summands_ containing only this
std::unique_ptr<FuncHelper> convertToSum(const FuncHelper& other){
    const SumFuncHelper* casted = dynamic_cast<const SumFuncHelper*>(&other);
    if(casted){
        return std::unique_ptr<SumFuncHelper>(new SumFuncHelper(*casted));
    }
    const std::vector<const FuncHelper*> summands={&other};
    return std::unique_ptr<FuncHelper>(new SumFuncHelper(summands));
}


//this minimizes the overhead by checking if any of the summands is already of type SumFuncHelper
//in that case other summand is added to summands, or, if also of type SumFuncHelper, summands are added
std::unique_ptr<FuncHelper> operator+(
    const FuncHelper& summand1,
    const FuncHelper& summand2
){
    const SumFuncHelper* casted1 = dynamic_cast<const SumFuncHelper*>(&summand1);
    if(casted1){
        std::unique_ptr<FuncHelper> copy = casted1->clone();
        SumFuncHelper* copyCasted = static_cast<SumFuncHelper*>(copy.get()); 
        copyCasted->addToSum(summand2); //this checks if summand2 is SumFuncHelper and adds summands in that case 
        return copy;
    }
    else if(dynamic_cast<const SumFuncHelper*>(&summand2)) return operator+(summand2,summand1);
    
   std::unique_ptr<FuncHelper> converted = convertToSum(summand1);
   SumFuncHelper* convertedCasted = static_cast<SumFuncHelper*>(converted.get());
   convertedCasted->addToSum(summand2);
   return converted;
}

std::unique_ptr<FuncHelper> operator+(
    const std::unique_ptr<FuncHelper>& summand1,
    const std::unique_ptr<FuncHelper>& summand2
){
    //check for nullpointer dereferences!
    if(summand1.get()==nullptr && summand2.get()==nullptr) throw(
        std::invalid_argument(
            "std::unique_ptr<FuncHelper> operator+(const std::unique_ptr<FuncHelper>& ,const std::unique_ptr<FuncHelper>&) needs at least on non null argument"
        )
    );
    if(summand1.get()==nullptr) return convertToSum(*summand2);
    if(summand2.get()==nullptr) return convertToSum(*summand1);
    return operator+(*summand1,*summand2);
}


std::unique_ptr<FuncHelper> convertToProduct(const FuncHelper& other){
    const ProductFuncHelper* casted = dynamic_cast<const ProductFuncHelper*>(&other);
    if(casted){
        return std::unique_ptr<FuncHelper>(new ProductFuncHelper(*casted));
    }
    const std::vector<const FuncHelper*> factors={&other};
    return std::unique_ptr<FuncHelper>(new ProductFuncHelper(factors));
}

std::unique_ptr<FuncHelper> operator*(
    const FuncHelper& factor1,
    const FuncHelper& factor2
){
    const ProductFuncHelper* casted1 = dynamic_cast<const ProductFuncHelper*>(&factor1);
    if(casted1){
        std::unique_ptr<FuncHelper> copy = casted1->clone();
        ProductFuncHelper* copyCasted = static_cast<ProductFuncHelper*>(copy.get()); 
        copyCasted->addToProduct(factor2); 
        return copy;
    }
    else if(dynamic_cast<const ProductFuncHelper*>(&factor2)) return operator+(factor2,factor1);
    
    std::unique_ptr<FuncHelper> converted = convertToProduct(factor1);
    ProductFuncHelper* convertedCasted = static_cast<ProductFuncHelper*>(converted.get());
    convertedCasted->addToProduct(factor2);
    return converted;
}

std::unique_ptr<FuncHelper> operator*(
    const std::unique_ptr<FuncHelper>& factor1,
    const std::unique_ptr<FuncHelper>& factor2
){
    //check for nullpointer dereferences!
    if(factor1.get()==nullptr && factor2.get()==nullptr) throw(
        std::invalid_argument(
            "std::unique_ptr<FuncHelper> operator*(const std::unique_ptr<FuncHelper>& ,const std::unique_ptr<FuncHelper>&) needs at least on non null argument"
        )
    );
    if(factor1.get()==nullptr) return convertToProduct(*factor2);
    if(factor2.get()==nullptr) return convertToProduct(*factor1);
    return operator*(*factor1,*factor2);

}

}