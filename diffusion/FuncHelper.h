//FuncHelper.h
//author: Henrik Matthiesen



/*
wrapper class to have a convient framework to manipulate functions
specifically tailored for functions required in score based diffusion models

//diffusion are of type
//dX_t = \mu(X_t,t) dt + \sigma(X_t,t) dW_t with W_t the standard Wiener process
//for diffusion models the functions \mu and \sigma are typically of a more specific form ike f(t) or f(t)X or f(t)X^{1/2}
//note that for the diffusion part that if W_t is k-dimension Wiener process and X_t is n-dimensional then \sigma(X_t,t) must be a nxk matrix
//this is covered more generally in FuncHelper via the operator of signature 
//virtual std::vector<double> operator()(
//        const std::vector<double>&,
//        double,
//        const std::vector<double>&
//    ) const =0;
//with the first argument X_t, the second t, and the third argument being the k dimensional dW_t
//in a (not here provided) type wrapping matrix valued functions this has to be overloaded accordingly
//the specific types provided here and inheriting from FuncHelper wrap these types of functions


//The specific types are as follows, check their interfaces below including detailed comments for specific functionality


//FuncHelper is purely virtual base to handle different types of functions polymorphically
//FuncHelperScalarType purely virtual inherits from FuncHelper for classes where an overload double operator(double x, double t) makes sense
//all classes below inherit publicly from FuncHelper

//ScalarFuncHelper wraps functions of signature double fct (double t) and double fct(double x, double t)
//also inherits publicly from FuncHelperScalarType

//VectorFuncHelper wraps functions of signature std::vector<double> fct(const std::vector<double>& X, double t)

//SumFuncHelper wraps collections of FuncHelpers to evaluate their sum

//ProductFuncHelper wraps collections of FuncHelpers to evaluate their product

//TimeShiftFuncHelper precomposes with affine maps in t coordinate

//ExplicitFuncHelper wraps explict variance schedule function used in score based diffusion models
//also inherits from publicly from FuncHelperScalarType

//PredictorFuncHelper wraps prediction models (eg linear regression, neural network)
//uses their member function predict to evaluate


//operations + and * are supposed to be used in conjunction with std::unique_ptr

//while it is a bit unusual and generally not encouraged to pass std::unique_ptr 
//this interacts well with the memory model employed here as well as providing a syntactically simple interface
//here is an example of that:

//f,g,h all be of type std::unique_ptr<FuncHelper>
//any arithmetic expression involving +,* and paranthesis we would want to write in f,g,h are totally fine and do exactly what you expect them to do, e.g.
// std::unique_ptr<FuncHelper> arithmeticExpr = f*(g+h)+g*(f+h)


//some non trivial memory management necessitates from the fact that we need to store copies of function when incorporating them into a sum
//and cannot just evaluate/keep a reference which makes the problem unacessible for instance via crtp/expresssion templates
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
class TimeShiftFuncHelper;

//purely virtual base class
class FuncHelper{
public:	
	
	//stored function is evaluated and then adjusted to
	//factor_*res^power_ 
	//default arguments don't do anything to result
	FuncHelper(
        double factor=1.0,
        double power=1.0
	);
	
	
	//polymorphic copying		
    virtual std::unique_ptr<FuncHelper> clone(void) const = 0;
    virtual std::unique_ptr<FuncHelper> modifiedClone(double factor, double power) const =0;


    virtual ~FuncHelper(void)=default;


    //arguments X,t
    virtual std::vector<double> operator()(
        const std::vector<double>&,
        double
    ) const =0;


    //arguments X,t,Z with Z eg random normals
    virtual std::vector<double> operator()(
        const std::vector<double>&,
        double,
        const std::vector<double>&
    ) const =0;
   
   
    double getFactor(void) const;
    double getPower(void) const;


protected:
    //adjust res to factor_*res^power_
    double adjustResult_(double res) const;

    //void resetFactor(double factor);
    //void resetPower(double power);

    double factor_; //multiply input function by factor_
    double power_;  //raise input function by power_
};

//
//end of purely virtual base class
//


//second purely virtual base class for functions that also map double->double

class FuncHelperScalarType : public FuncHelper{
public:
    FuncHelperScalarType(
        double factor=1.0,
        double power=1.0
    );

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        double factor=1.0,
        double power=1.0
    ) const = 0;

    virtual double operator()(double) const = 0;
};


//version for functions double -> double
//operators on vectors are implemented elementwise
class ScalarFuncHelper : public FuncHelperScalarType{
public:	
    //constructors

   //function in one argument
	//multiply indicates if evaluation for (x,t) is func1(t)*x or just func1(t)
    ScalarFuncHelper(
        double (* const func1)(double),
        bool multiply=false,
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
    //keeps other.multiply_

    ScalarFuncHelper(
        const ScalarFuncHelper& other,
        bool multiply, 
        double factor=1.0, 
        double power = 1.0
    );
        


    ScalarFuncHelper& operator=(const ScalarFuncHelper& other) = default;

    ~ScalarFuncHelper(void) = default;

    //clone methods
    std::unique_ptr<FuncHelper> clone(void) const override;

    std::unique_ptr<FuncHelper> modifiedClone (
        double factor,
        double power
    ) const override; 

    std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        double factor,
        double power
    ) const override;



    //operator()
    //apply operator() elementwise
    std::vector<double> operator()(
        const std::vector<double>& X, 
        double t
    )const override; 


    //this needs X.size()==Z.size()
    //returns vector with entries func1_(t)*Z_i or adjusted accordingly
    std::vector<double> operator()(
        const std::vector<double>& X, 
        double t,
        const std::vector<double>& Z
    ) const override; 


    double operator()(double t) const override;

private:

    double evaluate_(double, double) const;

    double (*func1_)(double);
    bool multiply_; //indicates if evaluation for (x,t) is func1(t)*x or just func1(t)

};

//
//end of ScalarFuncHelper
//


//version for functions (vector<double>,double) -> vector<double>
class VectorFuncHelper : public FuncHelper{
public:
	
	
	//power and factor both applied elementwise, i.e. operator() evaluates to factor*funcVec(X,t)[i]**power
    VectorFuncHelper(
        std::vector<double> (*funcVec)(const std::vector<double>&, double), 
        double factor=1.0,
        double power = 1.0
    );

    //copy and modify
    VectorFuncHelper(
        const VectorFuncHelper& other,
        double factor=1.0,
        double power=1.0
    );
        

    //here

    VectorFuncHelper& operator=(const VectorFuncHelper& other) = default;
    ~VectorFuncHelper(void) = default;


    //clone methods
    std::unique_ptr<FuncHelper> clone(void) const override;

    virtual std::unique_ptr<FuncHelper> modifiedClone(
        double factor, 
        double power
    ) const override;

  
    //operators()

    //returns funcVec_(X,t) modifies with power and factor in each component
    virtual std::vector<double> operator()(
        const std::vector<double>& X,
        double t
    ) const override;

    //returns Z[0]*funcVec_(X,t)
    std::vector<double> operator()(
        const std::vector<double>& X,
        double t,
        const std::vector<double>& Z
    ) const override;

    

private:
    std::vector<double> (*funcVec_)(const std::vector<double>&, double);
};

//
//end of VectorFuncHelper
//




//operator() returns originalFct_(x,shift_-speed_*t) and then modified by factor and power
class TimeShiftFuncHelper : public FuncHelper{
public:
    TimeShiftFuncHelper(
        const FuncHelper& original,
        double shift,
        double speed,
        double factor=1.0,
        double power=1.0
    );

    //make deep copies of owned uniquje_ptr
    TimeShiftFuncHelper(const TimeShiftFuncHelper& other); 
    TimeShiftFuncHelper& operator=(const TimeShiftFuncHelper& other);
	
	
    ~TimeShiftFuncHelper(void) = default;

    //clone methods
    std::unique_ptr<FuncHelper> clone(void) const override;
    
    std::unique_ptr<FuncHelper> modifiedClone(
        double factor,
        double power
    ) const override;

    std::unique_ptr<FuncHelper> modifiedClone(
        double shift,
        double speed,
        double factor=1.0, 
        double power=1.0
    ) const;
    
    std::vector<double> operator()(
        const std::vector<double>& X, 
        double t
    ) const override;   

    std::vector<double> operator()(
        const std::vector<double>& X, 
        double t,
        const std::vector<double>& Z
    ) const override; 

private:
    std::unique_ptr<FuncHelper> originalFct_;
    double shift_;
    double speed_;
};

//
//end of TimeShiftFuncHelper
//




//version for explicit variance schedule / evaluation of t -> betaMin_ + (betaMax_-betaMin_)*t/timeMax_;
class ExplicitFuncHelper : public FuncHelperScalarType{
public:

    //constructors
	//write explicitFctEval_(t)=betaMin_ + (betaMax_-betaMin_)*t/timeMax_
	//multiply indicates if evaluation on (X,t) is just explicitFctEval_(t) or explicitFctEval_(t)*X
	//integral indicates if we return explicitFctEval_(t) or its integral
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
        double factor=1,
        double power=1
    );

    ExplicitFuncHelper(
        const ExplicitFuncHelper& other,
        bool multiply, 
        bool integral,
        double factor=1,
        double power=1
    );

    //copy assignment
    ExplicitFuncHelper& operator=(const ExplicitFuncHelper& other) = default;

    //destructor
    ~ExplicitFuncHelper(void) = default;

    //clone methods
    std::unique_ptr<FuncHelper> clone(void) const override;

    std::unique_ptr<FuncHelper> modifiedClone(
        double factor, 
        double power
    ) const override;
    
    std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        double factor=1.0,
        double power=1.0
    ) const  override;

    std::unique_ptr<FuncHelper> modifiedClone(
        bool multiply,
        bool integral,
        double factor,
        double power
    ) const;

    //operators()
    std::vector<double> operator()(
        const std::vector<double>& X, 
        double t
    ) const override;

    //needs X.size()==Z.size()
    //returns vector with entries explicitEval_(t)*Z_i up to adjustment
    std::vector<double> operator()(
        const std::vector<double>& X, 
        double t,
        const std::vector<double>& Z
    ) const override;

    double operator()(double t) const override;



private:

    double explicitFctEval_(double x, double t) const;

    //member variables
    bool multiply_;
    bool integral_;
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
//requires T to have a function
//std::vector<double> predict(const std::vector<double>&X, double t)

template<typename T, typename std::enable_if<is_predictor<T>::value,T>::type* = nullptr>
class PredictorFuncHelper : public FuncHelper{
public:
    PredictorFuncHelper(
        T& predictor,
        double factor=1.0,
        double power=1.0
        )
        :FuncHelper(
            factor,
            power
        ),
        predictor_(predictor)
    {}

    PredictorFuncHelper(
        const PredictorFuncHelper& other,
        double factor=1.0,
        double power=1.0
    )
        :FuncHelper(
            other.factor_*factor,
            other.power_*power
        ),
        predictor_(other.predictor_)
    {}

    PredictorFuncHelper& operator=(const PredictorFuncHelper& other) = delete;
    ~PredictorFuncHelper(void) = default;

    std::vector<double> operator()(
        const std::vector<double>& X,
        double t
    ) const override {
        std::vector<double> res = predictor_.predict(X,t);
        assert(res.size()==X.size());
        for(size_t i=0; i<res.size(); i++) res[i]=adjustResult_(res[i]);
        return res;
    }

    std::vector<double> operator()(
        const std::vector<double>& X,
        double t,
        const std::vector<double>& Z
    ) const override {
        std::vector<double> res = this->operator()(X,t);
        assert(res.size()==Z.size());
        for(size_t i=0; i<res.size(); i++) res[i]*=Z[i];
        return res;
    }

    
	
    std::unique_ptr<FuncHelper> clone(void) const override {
        return std::unique_ptr<FuncHelper>(new PredictorFuncHelper(*this));
    }
    virtual std::unique_ptr<FuncHelper> modifiedClone(
        double factor,
        double power
    ) const override {
        return std::unique_ptr<FuncHelper>(
            new PredictorFuncHelper(
                *this,
                factor,
                power
            )
        );
    }

private:
    typename std::enable_if<is_predictor<T>::value,T>::type& predictor_;
};

//
//end of PredictorFuncHelper
//



//
//end of non aggregate types
//


//
//begin of aggregate types to handle + and *
//





//takes elementwise sum of functions
//makes owned copies of all functions involved so that lifetime management or modification of summands is not an issue
//stores owned copies of summands and returns sum of these in all operators
class SumFuncHelper : public FuncHelper{
public:
    SumFuncHelper(const std::vector<const FuncHelper*>& summands);
    //makes copies of all summands which are then owned by this

    SumFuncHelper(
        const SumFuncHelper& other,
        double factor=1,
        double power=1
    );

    //not default because of unique_ptrs in summands_
	//creates (deep) copies of all summands
    SumFuncHelper& operator=(const SumFuncHelper& other);

    //ok because of unique ptrs in summands_
    ~SumFuncHelper(void)=default; 



    std::unique_ptr<FuncHelper> clone(void) const override;

    std::unique_ptr<FuncHelper> modifiedClone(
        double factor,
        double power
    ) const override;

    std::vector<double> operator()(
        const std::vector<double>& X,
        double t
    ) const override;

    std::vector<double> operator()(
        const std::vector<double>& X,
        double t,
        const std::vector<double>& Z
    ) const override;
		
	//minimizes potential overhead by checking if newSummand is of type SumFuncHelper
	//if it is summands of newSummand are added
    SumFuncHelper& addToSum(const FuncHelper& newSummand);

private:
    std::vector<std::unique_ptr<FuncHelper>> summands_;
};

//
//end of SumFuncHelper
//


//takes elementwise product of functions
//makes owned copies of all functions involved so that lifetime management or modification of factors is not an issue
//stores owned copies of summands and returns product of these in all operators=
class ProductFuncHelper : public FuncHelper{
public:
    ProductFuncHelper(const std::vector<const FuncHelper*>& factors);
    //makes copies of all factors owned by this

    ProductFuncHelper(
        const ProductFuncHelper& other,
        double factor=1,
        double power=1
    );

    //not default because of unique_ptrs in factors_
	//makes (deep) copies of all factors
    ProductFuncHelper& operator=(const ProductFuncHelper& other);

    //ok because of unique ptrs in factors_
    ~ProductFuncHelper(void)=default; 



    std::unique_ptr<FuncHelper> clone(void) const override;

    std::unique_ptr<FuncHelper> modifiedClone(
        double factor,
        double power
    ) const override;

    std::vector<double> operator()(
        const std::vector<double>& X,
        double t
    ) const override;

    std::vector<double> operator()(
        const std::vector<double>& X,
        double t,
        const std::vector<double>& Z
    ) const override;
		
	//minimizes potential overhead by checking if newFactor is of type ProductFuncHelper
	//if it is factors of newFactor are added
    ProductFuncHelper& addToProduct(const FuncHelper& newFactor);

private:
    std::vector<std::unique_ptr<FuncHelper>> factors_;
};


//
//end of ProductFuncHelper
//

//
//operators +,* in general case
//defines user defined cast from FuncHelper to Sum/Product and then uses addToSum/addToProduct
//



//cast to sum
//if other is SumFuncHelper returns a deep copy
//otherwise returns a SumFuncHelper with single summand other
std::unique_ptr<FuncHelper> convertToSum(const FuncHelper& other);

//creates a new SumFuncHelper with summands being the union of the summands of summand1 and summand2
//summands may be onle summand1 and summand2 themselves
std::unique_ptr<FuncHelper> operator+(
    const FuncHelper& summand1,
    const FuncHelper& summand2
);


//throws if both are null
//just calls above operator but allows us to write eg f+g+h
std::unique_ptr<FuncHelper> operator+(
    const std::unique_ptr<FuncHelper>& summand1,
    const std::unique_ptr<FuncHelper>& summand2
);


//cast to product
//if other is ProductFuncHelper returns a deep copy
//otherwise returns a ProductFuncHelper with single factor other
std::unique_ptr<FuncHelper> convertToProduct(const FuncHelper& other);


//creates a new ProductFuncHelper with factors being the union of the summands of summand1 and summand2
//factprs may be only factor1 and factor3 themselves
std::unique_ptr<FuncHelper> operator*(
    const FuncHelper& factor1,
    const FuncHelper& factor2
);

//throws if both are null
//just calls above operator but allows us to write eg f*f*h
std::unique_ptr<FuncHelper> operator*(
    const std::unique_ptr<FuncHelper>& factor1,
    const std::unique_ptr<FuncHelper>& factor2
);




}; //end of namespace diffusion

