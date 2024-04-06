//FuncHelper.h


/*
wrapper class to have a convient framework to manipulate functions
specifically tailored for functions required in score based diffusion models

supposed to be used in combination with std::unique_ptr
thus operators + and * accept const std::unique_ptr<FuncHelper> as arguments
even though it is generally preferable to pass by reference or raw pointer if membership is not transfered
this allows us to write expressions of type
std::unique_ptr<FuncHelper> f = (f1+f2)*f3
if f1,f2,f3 are all of type std::unique_ptr<FuncHelper>


//FuncHelper is purely virtual base to handle different types of functions polymorphically

//ScalarFuncHelper wraps functions of signature double fct (double t) and double fct(double x, double t)

//VectorFuncHelper wraps functions of signature std::vector<double> fct(const std::vector<double>& X, double t)

//SumFuncHelper wraps collections of FuncHelpers to evaluate their sum

//ProductFuncHelper wraps collections of FuncHelpers to evaluate their product

//TimeShiftFuncHelper precomposes with affine maps in t coordinate

//ExplicitFuncHelper wraps explict variance schedule function used in score based diffusion models

//PredictorFuncHelper wraps prediction models (eg linear regression, neural network)
//uses their member function predict to evaluate

//operations + and * are supposed to be 




*/



#pragma once

#include <vector>
#include <memory>
#include <type_traits>
#include <cmath>
#include <iostream>
#include "is_predictor.h"



namespace diffusion{


//forward declatations
class SumFuncHelper;
class ProductFuncHelper;

//purely virtual base class
class FuncHelper{
public:	
	FuncHelper(
        bool multiply=false,
        bool integral=false,
        double factor=1.0,
        double power=1.0
	);
	   			
    virtual std::unique_ptr<FuncHelper> clone(void) const =0;
    virtual std::unique_ptr<FuncHelper> modifiedClone(double factor, double power) const =0;
    virtual std::unique_ptr<FuncHelper> modifiedClone(bool multiply, bool integral, double factor=1.0, double power=1.0) const =0;


    virtual double operator()(double) const = 0;
    virtual double operator()(double, double) const =0;
    virtual std::vector<double> operator()(const std::vector<double>&, double) const =0;
    virtual ~FuncHelper(void)=default;


    //create copies of this casted to single term Sum/Product objects if not already of required type
    //in the later case we get unmodified copies
    std::unique_ptr<SumFuncHelper> convertToSum(void) const;
    std::unique_ptr<ProductFuncHelper> convertToProduct(void) const;



    //addition
    std::unique_ptr<SumFuncHelper> add(const FuncHelper& other) const;

    //multiplication
    std::unique_ptr<ProductFuncHelper> multiply(const FuncHelper& other) const;





protected:
    //adjust res to factor_*res^power_
    double adjustResult_(double res) const;

    //adjust res to factor_*x*res^power_ 
    //multiply by t to integrate
    double adjustResult_(double res, double x, double t) const;

    double factor_; //multiply input function by factor_
    double power_;  //raise input function by power_
    bool multiply_; //evaluate f(t)*x or just f(t) in case of one variable function applied to (x,t)
    bool integral_; //integrate input function when explicit
  
};

//
//end of purely virtual base class
//


//version for functions double -> double and (double,double) -> double
class ScalarFuncHelper : public FuncHelper{
public:	
    //constructors

   //function in one argument
    ScalarFuncHelper(
        double (* const func1)(double),
        bool multiply=false, 
        bool integral=false, 
        double factor=1.0, 
        double power = 1.0
    ); 
	
	
    //function in two arguments
    ScalarFuncHelper(
        double (*func2)(double,double),
        bool multiply=false, 
        bool integral=false, 
        double factor=1.0, 
        double power = 1.0
    );
       

    //copy and modify
    //includes ordinary copy constructor

    ScalarFuncHelper(
        const ScalarFuncHelper& other,
        double factor=1.0,
        double power=1.0
    );

    ScalarFuncHelper(
        const ScalarFuncHelper& other,
        bool multiply, 
        bool integral, 
        double factor=1.0, 
        double power = 1.0
    );
        


    ScalarFuncHelper& operator=(const ScalarFuncHelper& other);

    ~ScalarFuncHelper(void) = default;

    //clone methods
    virtual std::unique_ptr<FuncHelper> clone(void) const;

    virtual std::unique_ptr<FuncHelper> modifiedClone (
        double factor,
        double power
    ) const; 

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        bool integral, 
        double factor=1, 
        double power=1
    ) const;

    //operator()
    inline virtual double operator()(double t) const; 
    inline virtual double operator()(double x, double t) const;
    //apply operator() elementwise
    inline virtual std::vector<double> operator()(const std::vector<double>& X, double t) const; 

private:
    double (*func1_)(double);
    double(*func2_)(double,double);

};

/*
//end of ScalarFuncHelper
*/


//version for functions vector<double> -> vector<double> and (vector<double>,double) -> vector<double>
class VectorFuncHelper : public FuncHelper{
public:

    VectorFuncHelper(
        std::vector<double> (*funcVec)(const std::vector<double>&, double), 
        bool multiply=false,
        bool integral=false,
        double factor=1.0,
        double power = 1.0
    );

    //copy and modify
    VectorFuncHelper(
        const VectorFuncHelper& other,
        double factor=1.0,
        double power=1.0
    );
        

    VectorFuncHelper(
        const VectorFuncHelper& other,
        bool multiply,
        bool integral,
        double factor=1.0,
        double power=1.0
    );

    //here

    VectorFuncHelper& operator=(const VectorFuncHelper& other) = default;
    ~VectorFuncHelper(void) = default;


    //clone methods
    virtual std::unique_ptr<FuncHelper> clone(void) const;

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        double factor, 
        double power
    ) const;

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        bool integral, 
        double factor=1.0, 
        double power=1.0
    ) const;
  
    //operators()
    virtual double operator()(double x, double t) const;
    
    virtual double operator()(double t) const;

    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const;

private:
    std::vector<double> (*funcVec_)(const std::vector<double>&, double);
};


/*
//end of VectorFuncHelper
*/


//wrapper to hold sum of FuncHelpers
//stores owned copies and evaluated sum for operator()
class SumFuncHelper : public FuncHelper{
public:
    //makes copies of summands now owned by it
    SumFuncHelper(const std::vector<const FuncHelper*>& summands);

  
    //copy 
    
    SumFuncHelper(
        const SumFuncHelper& other,
        bool multiply,
        bool integral,
        double factor=1.0,
        double power=1.0
    );

    SumFuncHelper(
        const SumFuncHelper& other,
        double factor=1.0,
        double power=1.0
    );

    
    //copy assignment
    SumFuncHelper& operator=(const SumFuncHelper& other);
    

    //ok because summands_ contains std::unique_ptr
    ~SumFuncHelper(void){
        std::cout << "calling ~SumFuncHelper" << std::endl;
    }

    //clone methods

    virtual std::unique_ptr<FuncHelper> clone(void) const;

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        double factor,
        double power
    ) const;
    

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply, 
        bool integral,
        double factor=1.0, 
        double power=1.0
    ) const;
    
    //operators()
    virtual double operator()(double x, double t) const;
    virtual double operator()(double t) const;
    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const;
    

    SumFuncHelper& addToSum(const FuncHelper& newSummand);
    
private:
    std::vector<std::unique_ptr<const FuncHelper>> summands_;
};

//
//end of class SumFuncHelper
//


//wrapper to hold product of FuncHelpers
//stores owned copies and evaluated product for operator()
class ProductFuncHelper : public FuncHelper{
public:
    //makes copies of factors now owned by this
    ProductFuncHelper(const std::vector<const FuncHelper*>& factors);

    //copy
    ProductFuncHelper(
        const ProductFuncHelper& other,
        double factor=1.0,
        double power=1.0
    );

    ProductFuncHelper(
        const ProductFuncHelper& other,
        bool multiply,
        bool integral,
        double factor=1.0,
        double power=1.0
    );


    //copy assignment
    ProductFuncHelper& operator=(const ProductFuncHelper& other);

    //ok because factors_ has unique_ptr
    ~ProductFuncHelper(void) = default;


    //clone methods

    virtual std::unique_ptr<FuncHelper> clone(void) const;
   
    virtual std::unique_ptr<FuncHelper> modifiedClone(
        double factor,
        double power
    ) const;

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        bool integral,
        double factor,
        double power
    ) const;

    //operators()
    virtual double operator()(double x, double t) const;
    virtual double operator()(double t) const;
    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const;

    //add a new factor
    ProductFuncHelper& addToProduct(const FuncHelper& newFactor);

private:
    std::vector<std::unique_ptr<const FuncHelper>> factors_;

};

//
//end of ProductFuncHelper
//

//returns originalFct_(x,shift_-speed_*t)
class TimeShiftFuncHelper : public FuncHelper{
public:
    TimeShiftFuncHelper(
        const FuncHelper& original,
        double shift=0,
        double speed=1
    );

    TimeShiftFuncHelper(
        const std::unique_ptr<FuncHelper>& original,
        double shift=0,
        double speed=1
    );

    TimeShiftFuncHelper(
        const TimeShiftFuncHelper& other,
        double factor=1.0,
        double power=1.0
    );

    TimeShiftFuncHelper(
        const TimeShiftFuncHelper& other,
        bool multiply,
        bool integral,
        double factor=1.0,
        double power=1.0
    );

    //clone methods
    virtual std::unique_ptr<FuncHelper> clone(void) const;
    
    virtual std::unique_ptr<FuncHelper> modifiedClone(
        double factor,
        double power
    ) const;

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply, 
        bool integral, 
        double factor=1.0, 
        double power=1.0
    ) const;


    virtual double operator()(double t) const;

    virtual double operator()(double x, double t) const;
    
    virtual std::vector<double> operator()(
        const std::vector<double>& X, 
        double t
    ) const;   

private:
    std::unique_ptr<FuncHelper> originalFct_;
    double shift_;
    double speed_;
};

//
//end of TimeShiftFuncHelper
//

//version for explicit variance schedule / evaluation of t -> betaMin_ + (betaMax_-betaMin_)*t/timeMax_;
class ExplicitFuncHelper : public FuncHelper{
public:

    //constructors
    ExplicitFuncHelper(
        double betaMin,
        double betaMax,
        double timeMax,
        bool multiply=false, 
        bool integral=false,
        double factor=1,
        double power=1 
    );

    //copy and modify
    ExplicitFuncHelper(
        const ExplicitFuncHelper& other,
        bool multiply=false, 
        bool integral=false,
        double factor=1,
        double power=1
    );
    //copy assignment
    ExplicitFuncHelper& operator=(const ExplicitFuncHelper& other) = default;

    //destructor
    ~ExplicitFuncHelper(void) = default;

    //clone methods
    virtual std::unique_ptr<FuncHelper> clone(void) const;

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        double factor, 
        double power
    ) const;
   
    virtual std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        bool integral,
        double factor,
        double power
    ) const;

    //operators()
    virtual double operator()(double t) const;
    virtual double operator()(double x, double t) const;
    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const;

private:

    double explicitFctEval_(double t) const;

    //member variables
    double betaMin_;
    double betaMax_;
    double timeMax_;
};


//
//end of ExplicitFuncHelper
//


//wrapper template for models
//used for T a predcitor class (eg linear regression, neural network,..)
//uses T::predict to evaluate operator()
template<typename T, typename std::enable_if<is_predictor<T>::value,T>::type* = nullptr>
class PredictorFuncHelper : public FuncHelper{
public:
    PredictorFuncHelper(
        T& predictor,
        bool multiply=false,
        bool integral=false,
        double factor=1.0,
        double power=1.0
        )
        :FuncHelper(
            multiply,
            integral,
            factor,
            power
        ),
        predictor_(&predictor)
    {}

    PredictorFuncHelper(
        const PredictorFuncHelper& other,
        double factor=1.0,
        double power=1.0
    )
        :FuncHelper(
            other.multiply_,
            other.integral_,
            other.factor_*factor,
            other.power_*power
        ),
        predictor_(other.predictor_)
    {}

    PredictorFuncHelper(
        const PredictorFuncHelper& other,
        bool multiply,
        bool integral,
        double factor=1.0,
        double power=1.0
    )
        :FuncHelper(
            multiply,
            integral,
            other.factor_*factor,
            other.power_*power
        ),
        predictor_(other.predictor_)
    {}

    //assignment and destruct?

    virtual double operator()(double t) const {
        double res = predictor_->predict(t);
        return adjustResult_(res);
    }

    virtual double operator()(double x, double t) const {
        double res = predictor_->predict(x,t);
        return adjustResult_(res);
    }

    virtual std::vector<double> operator()(const std::vector<double>& X, double t) const {
        return predictor_->predict(X,t);
    }
    
	
    virtual std::unique_ptr<FuncHelper> clone(void) const{
        return std::unique_ptr<FuncHelper>(new PredictorFuncHelper(*this));
    }
    virtual std::unique_ptr<FuncHelper> modifiedClone(
        double factor,
        double power) 
    const{
        return std::unique_ptr<FuncHelper>(
            new PredictorFuncHelper(
                *this,
                factor,
                power
            )
        );
    }
    virtual std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        bool integral,
        double factor=1.0,
        double power=1.0) 
    const{
        return std::unique_ptr<FuncHelper>(
            new PredictorFuncHelper(
                *this,
                multiply,
                integral,
                factor,
                power
            )
        );
    }

private:
    typename std::enable_if<is_predictor<T>::value,T>::type* const predictor_;
};



//operator +
std::unique_ptr<SumFuncHelper> operator+(
    const std::unique_ptr<FuncHelper>& lhs,
    const std::unique_ptr<FuncHelper>& rhs
);

std::unique_ptr<SumFuncHelper> operator+(
    const FuncHelper& lhs,
    const FuncHelper& rhs
);


//operator *
std::unique_ptr<ProductFuncHelper> operator*(
    const FuncHelper& lhs,
    const FuncHelper& rhs
);

std::unique_ptr<ProductFuncHelper> operator*(
    const std::unique_ptr<FuncHelper>& lhs,
    const std::unique_ptr<FuncHelper>& rhs
);


//
//end of PredictorFuncHelper

}; //end of namespace diffusion

