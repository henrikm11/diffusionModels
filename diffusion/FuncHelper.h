//FuncHelper.h



#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <memory>
//#include "tensor_template/tensor_template.h"
//#include "neuralNetwork.h"

//namespace diffusion{

//purely virtual base class
class FuncHelper{
public:
    FuncHelper(double factor=1.0, double power=1.0, bool integral=false): factor_(factor), power_(power), integral_(integral){};


    virtual double operator()(double) const =0;
    virtual double operator()(double, double) const =0;
    virtual std::vector<double> operator()(const std::vector<double>&, double) const =0;
    virtual ~FuncHelper(void)=default;


protected:
    //adjust res to factor_*res^power_
    double adjustResult_(double res) const {
        res = std::pow(res,power_);
        res*=factor_;
        return res;
    }

    //adjust res to factor_*x*res^power_ 
    //multiply by t to integrate
    double adjustResult_(double res, double x, double t) const {
        res = std::pow(res, power_);
        res*=x;
        res*=factor_;
        if(integral_) res*=t;
        return res;
    }


private:
    double factor_; //multiply input function by factor_
    double power_;  //raise input function by power_
    bool integral_; //integrate input function when explicit
};


//
//end of purely virtual base class
//


//version for functions double -> double and (double,double) -> double
class ScalarFuncHelper : public FuncHelper{
public:
    
    //function in one argument
    ScalarFuncHelper(double (*const func1)(double), double factor=1.0, double power = 1.0, bool integral=false)
        :FuncHelper(factor,power,integral),
        func1_(func1),
        func2_(nullptr)
    {
        if(func1==nullptr) throw std::invalid_argument("FuncHelper needs non null pointer for instantiation");
    };

    //function in two arguments
    //function in one argument
    ScalarFuncHelper(double (*const func2)(double,double), double factor=1.0, double power = 1.0, bool integral=false)
        :FuncHelper(factor,power,integral),
        func1_(nullptr),
        func2_(func2)
    {
        if(func2==nullptr) throw std::invalid_argument("FuncHelper needs non null pointer for instantiation");
    };




    virtual double operator()(double t) const {
        if(func2_!=nullptr) return func2_(1.0,t);
        double res = func1_(t);
        res = adjustResult_(res);
        return res;
    }

    virtual double operator()(double x, double t) const {
        if(func2_!=nullptr) return func2_(x,t); 
        double res = func1_(t);
        res = adjustResult_(res,x,t);
        return res;
    }

    //apply operator() elementwise
    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const {
        std::vector<double> res(X.size(),0);
        for(size_t i=0; i<res.size(); i++) res[i]=operator()(X[i],t);
        return res;
    }


private:
    double (*func1_)(double);
    double (*func2_)(double,double);
};

//
//end of ScalarFuncHelper
//


//version for functions vector<double> -> vector<double> and (vector<double>,double) -> vector<double>
class VectorFuncHelper : public FuncHelper{
public:
    VectorFuncHelper(std::vector<double> (*funcVec)(const std::vector<double>&, double), double factor=1.0, double power = 1.0, bool integral=false)
        :FuncHelper(factor, power, integral),
        funcVec_(funcVec)
    {
        if(funcVec==nullptr) throw std::invalid_argument("VectorFuncHelper needs non null pointer for instantiation");
    }

    virtual double operator()(double x, double t) const {
        std::vector<double> X = {x};
        std::vector<double> resVec = funcVec_(X,t);
        return resVec[0]; 
    }

    virtual double operator()(double t) const {
        return operator()(1.0,t);
    }


    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const {
        return funcVec_(X,t);
    }


private:
    std::vector<double> (*funcVec_)(const std::vector<double>&, double);

};

//
//end of VectorFuncHelper
//

//version for explicit variance schedule / evaluation of t -> betaMin_ + (betaMax_-betaMin_)*t/timeMax_;
class ExplicitFuncHelper : public FuncHelper{
public:
    ExplicitFuncHelper(double betaMin, double betaMax, double timeMax, double factor=1, double power=1, bool integral=false)
        :FuncHelper(factor,power,integral),
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

    virtual double operator()(double t) const {
        double res = explicitFctEval_(t);
        res = adjustResult_(res);
        return res;
    }

    virtual double operator()(double x, double t) const {
        double res = explicitFctEval_(t);
        res = adjustResult_(res,x,t);
        return res;
    }

    virtual std::vector<double> operator()(const std::vector<double>& X, double t){
        std::vector<double> res(X.size(),0);
        for(size_t i=0; i<X.size(); i++) res[i]=operator()(X[i],t);
        return res;
    }


private:

    double explicitFctEval_(double t) const {
        double res = 0;
        res = betaMin_ + (betaMax_-betaMin_)*t/timeMax_;
        return res;
    };

    //member variables
    double betaMin_;
    double betaMax_;
    double timeMax_;
};

//
//end of ExplicitFuncHelper
//

/*
//version to use predict of neural network
class NeuralNetFuncHelper : public FuncHelper{
public:
    NeuralNetFuncHelper(vanillaNeuralNet::neuralNetworkWeightedMSE* neuralNet):neuralNet_(neuralNet){};

    virtual double operator()(double x, double t){
        std::vector<double> X = {x};
        return operator()(X,t)[0];
    }

    virtual double operator()(double t){
        return operator()(1.0,t);
    }

    virtual std::vector<double> operator(const std::vector<double>& X, double t){
        Tensor<double,1> X_tensor(X);
        Tensor<double,1> res_tensor = neuralNet_->predict(X_tensor,t,true);
        return std::vector<double>(res_tensor);
    }


private:
    vanillaNeuralNet::neuralNetworkWeightedMSE* neuralNet_;
};

//
//end of NeuralNetFuncHelper
//
*/



//need to ensure that we use scalar/vector versions for all of them

//version to sum across vector of FuncHelpers
class SumFuncHelper : public FuncHelper{
public:
    SumFuncHelper(const std::vector<FuncHelper*> summands):summands_(summands){};

    virtual double operator()(double t) const {
        double res=0;
        for(const FuncHelper* f : summands_) res+=(*f)(t);
        return res;
    }

    virtual double operator()(double x, double t) const {
        double res=0;
        for(const FuncHelper* f : summands_) res+=(*f)(x,t);
        return res;
    }

    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const {
        std::vector<double> res(X.size(),0);
        for(const FuncHelper* f : summands_){
            std::vector<double> nextSummand = (*f)(X,t);
            for(size_t i=0; i<X.size(); i++){
                res[i]+=nextSummand[i];
            }
        }
        return res;
    }

private:
    std::vector<FuncHelper*> summands_;

};

//
//end of SumFuncHelper
//


//version to multiply accross vector of FuncHelpers
class ProductFuncHelper : public FuncHelper{
public:
    ProductFuncHelper(const std::vector<FuncHelper*> factors):factors_(factors){};

    virtual double operator()(double t) const {
        double res=1;
        for(const FuncHelper* f : factors_) res*=(*f)(t);
        return res;
    }

    virtual double operator()(double x, double t) const {
        double res=1;
        for(const FuncHelper* f : factors_) res*=(*f)(x,t);
        return res;
    }

    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const {
        std::vector<double> res(X.size(),1);
        for(const FuncHelper* f : factors_){
            std::vector<double> nextSummand = (*f)(X,t);
            for(size_t i=0; i<X.size(); i++){
                res[i]*=nextSummand[i];
            }
        }
        return res;
    }


private:
    std::vector<FuncHelper*> factors_;

};

//
//end of ProductFuncHelper
//



//} //end namespace diffusion