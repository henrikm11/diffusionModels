//FuncHelper.h

#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>

namespace diffusion{




//everything inlined automatically!
// helper class to deal with various signatures of functions simultaneously
// if f is in two variables returns f(a,b)
// if f is in one variable returns factor_*a*(f(b)**power) or integral of this
// if parameters betaMin, betaMax, timeMax are passed explicit f(t)= betaMin + (betaMax-betaMin)*t/timeMax
class FuncHelper{
public:

    //function in one argument
    FuncHelper(double (*const func1)(double), double factor=1.0, double power = 1.0)
    :func1_(func1), func2_(nullptr), func2Vec_(nullptr), factor_(factor), power_(power)
    {if(func1==nullptr) throw std::invalid_argument("FuncHelper needs non null pointer for instantiation");};
    
    //function in two arguments, one dimensional
    FuncHelper(double (*const func2)(double,double), double factor=1.0, double power = 1.0)
    :func1_(nullptr), func2_(func2), func2Vec_(nullptr), factor_(factor), power_(power)
    {if(func2==nullptr) throw std::invalid_argument("FuncHelper needs non null pointer for instantiation");};

    //function in two arguments, any dimensional
    FuncHelper(std::vector<double> (*const func2Vec)(const std::vector<double>&,double), double factor=1.0, double power = 1.0)
    :func1_(nullptr), func2_(nullptr), func2Vec_(func2Vec), factor_(factor), power_(power)
    {if(func2Vec==nullptr) throw std::invalid_argument("FuncHelper needs non null pointer for instantiation");};
    
	//copy type constructor that can modify
    FuncHelper(const FuncHelper& other, double factor=1.0, double power = 1.0)
    :func1_(other.func1_), func2_(other.func2_), func2Vec_(other.func2Vec_), factor_(factor), power_(power)
    {};
    
	
	//constructor for explicit funcion (x,t)->factor*(betaMin + (betaMax-betaMin)*t/timeMax)**power*x or its integral
    FuncHelper(double betaMin, double betaMax, double timeMax, double factor=1.0, double power=1.0, bool integral=false)
    :func1_(nullptr), func2_(nullptr), func2Vec_(nullptr), factor_(factor), power_(power), betaMax_(betaMax), betaMin_(betaMin), integral_(integral)
    {
        if(betaMin>=betaMax) throw std::invalid_argument("FuncHelper needs betaMin<betaMax");
        if(timeMax<=0) throw std::invalid_argument("FuncHelper needs timeMax>0");
        if(integral_){
            betaMax_+=betaMin_;
            betaMin_*=2;
        }
        return;
    };
	   
	//overloads for operator()

    double operator()(double x, double t){
        if(func2_!=nullptr) return func2_(x,t); 
        double res=0;
        if(func1_!=nullptr) res = func1_(t);
        else res = explicitFctEval_(t);
        res = adjustResult_(res,x,t);
        return res;
    };

    std::vector<double> operator()(const std::vector<double>& X, double t){
        if(func2Vec_!=nullptr) return func2Vec_(X,t);
        std::vector<double> res(X.size(),0);
        for(size_t i=0; i<res.size(); i++) res[i]=operator()(X[i],t);
        return res;
    }

    double operator()(double t){
        if(func2_!=nullptr) return func2_(1,t);
        double res = 0;
        if(func1_!=nullptr) res = func1_(t);
        else double res = explicitFctEval_(t);
        res = adjustResult_(res);
        return res;
    }
private:
    //evaluation of function betaMin_ + (betaMax_-betaMin_)*t/timeMax_;
    double explicitFctEval_(double t){
        double res = 0;
        res = betaMin_ + (betaMax_-betaMin_)*t/timeMax_;
        return res;
    };

 
   //adjust computation of function in one variable for power and factor
    double adjustResult_(double res){
        res = std::pow(res, power_);
        res*=factor_;
        return res;
    };

    //adjust comnputation of function in two variables for power, factor, integral
    double adjustResult_(double res, double x, double t){
        res = std::pow(res, power_);
        res*=x;
        res*=factor_;
        if(integral_) res*=t;
        return res;
    };

    double (*const func1_)(double);
    double (*const func2_)(double,double);

    //std::vector<double> (*const func1Vec_)(const std::vector<double>&);
    std::vector<double> (*const func2Vec_)(const std::vector<double>&, double);

    double factor_=1;
    double power_=1;

    double betaMin_=0;
    double betaMax_=0; 
    double timeMax_=0;
    bool integral_=false;
    
};

} //end namespace diffusion