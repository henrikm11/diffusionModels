//FunctionalWrapper.hpp

#pragma once
#include "ReturnExpression.hpp"


//TODO
/*
-) some version of perfect forwarding, if just thrown in one gets a ~40% performance hit in certain situations from overhead
-) overload resolution for FunctionWrapper and FunctionObjectWrapper, note that older versions of clang have a bug that prevents implementing this using concepts
*/

namespace funcExpr{
	
	
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// AbstractFunction
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////



//T is final return type should be numeric or std::vector<numeric>
//Args are types of arguments of encapsulated function
//BE is ReturnExpression of basic type ReturnScalar, ReturnWrapper, ReturnVector
template<typename BE, typename... Args>
class AbstractFunction{
public:
    using DataType = typename BE::DataType; 
	using BaseReturnExpressionType = BE;
	
	virtual std::unique_ptr<AbstractFunction<BE,Args...>> clone(void) const = 0;
	
	virtual BE operator()(Args...) const = 0;
    virtual ~AbstractFunction(void) = default;
};



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// FunctionExpression
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

//F is FunctionExpression to which calls are delegates
//need E general ReturnExpression type here
//Args are arguments of function calls
template<typename F, typename E, typename... Args>
requires isReturnExpression<E>
class FunctionExpression : public AbstractFunction<typename E::BaseReturnExpressionType, Args...>{
public:
    using DataType = typename E::DataType; 
	using BaseReturnExpressionType = typename E::BaseReturnExpressionType;
	using FunctionalExpressionType = F;
	
	BaseReturnExpressionType operator()(Args... args) const final{
		//return BaseReturnExpressionType(static_cast<const F*>(this)->evaluate(std::forward<decltype(args)>(args)...));
		return BaseReturnExpressionType(static_cast<const F*>(this)->evaluate(args...));
	}
	
	E evaluate(Args... args) const{
		return static_cast<const F*>(this)->evaluate(args...);
	}	
protected:
	virtual ~FunctionExpression(void) = default;
};

template<typename F, typename... Args>
concept isFunctionExpression  = std::derived_from<std::remove_reference_t<F>,FunctionExpression<F,typename F::ReturnExpressionType, Args...>>;


///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
//FunctionWrapper
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

template<typename E, typename... Args>
requires isReturnExpression<E>
class FunctionWrapper :  public FunctionExpression<FunctionWrapper<E,Args...>, E, Args...>{
public:
	using DataType = typename E::DataType;
	using BaseReturnExpressionType = typename E::BaseReturnExpressionType;
	using ReturnExpressionType = E;
	
	std::unique_ptr<AbstractFunction<BaseReturnExpressionType,Args...>> clone(void) const final{
		return std::make_unique<FunctionWrapper<E,Args...>>(*this);	
	}

	
	FunctionWrapper(E (*func)(Args...)):func_(func){}
	FunctionWrapper(const FunctionWrapper&) = default;
	~FunctionWrapper(void) final = default;
	
	
	E evaluate(Args... args) const{
		//return func_(std::forward<decltype(args)>(args)...);
		return func_(args...);
	}

private:
	E (*func_)(Args...);
};


/*
template<typename T, typename... Args>
requires numerical<T>
class FunctionWrapper :  public FunctionExpression<FunctionWrapper<T,Args...>, ReturnScalar<T>, Args...>{
public:
	using DataType = T;
	using BaseReturnExpressionType = ReturnScalar<T>;
	using ReturnExpressionType = ReturnScalar<T>;
	
	std::unique_ptr<AbstractFunction<BaseReturnExpressionType,Args...>> clone(void) const final{
		return std::make_unique<FunctionWrapper<T,Args...>>(*this);	
	}
	
	FunctionWrapper(T (*func)(Args...)):func_(func){}
	FunctionWrapper(const FunctionWrapper&) = default;
	~FunctionWrapper(void) final = default;
	
	
	E evaluate(Args... args) const{
		//return ReturnScalar<T>(func_(std::forward<decltype(args)>(args)...));
		return ReturnScalar<T>(func_(args...));
	}

private:
	T (*func_)(Args...);
};
*/

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
//FunctionObjectWrapper
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////


template<typename F, typename E, typename... Args>
requires isReturnExpression<E>
class FunctionObjectWrapper : public FunctionExpression<FunctionObjectWrapper<F,E,Args...> ,E, Args...>{
public:
	using DataType = typename E::DataType;
	using BaseReturnExpressionType = typename E::BaseReturnExpressionType;
	using ReturnExpressionType = E;
	
	
	std::unique_ptr<AbstractFunction<BaseReturnExpressionType,Args...>> clone(void) const final{
		return std::make_unique<FunctionObjectWrapper<F,E,Args...>>(*this);	
	}

	
	FunctionObjectWrapper(const F& funcOb): funcOb_(funcOb){}
	~FunctionObjectWrapper(void) final = default;
	
	E evaluate(Args... args) const{
		//return funcOb_(std::forward<decltype(args)>(args)...);
		return funcOb_(args...);
	}
	
	
private:
	F funcOb_;
};




/*
template<typename F, numerical T, typename... Args>
requires numerical<T> && requires(F f, Args... args){ f(args...) -> std::same_as<E>;}
class FunctionObjectWrapper : public FunctionExpression<FunctionObjectWrapper<F,T,Args...> ,ReturnScalar<T>, Args...>{
public:
	using DataType = T;
	using BaseReturnExpressionType = ReturnScalar<T>;
	using ReturnExpressionType = ReturnScalar<T>;
	
	std::unique_ptr<AbstractFunction<BaseReturnExpressionType,Args...>> clone(void) const final{
		return std::make_unique<FunctionObjectWrapper<F,T,Args...>>(*this);	
	}
	
	
	FunctionObjectWrapper(const F& funcOb): funcOb_(funcOb){}
	~FunctionObjectWrapper(void) final = default;
	
	ReturnScalar<T> evaluate(Args... args) const{
		//return ReturnScalar<T>(funcOb_(std::forward<decltype(args)>(args)...));
		return ReturnScalar<T>(funcOb_(args...));
	}
	
	
private:
	F funcOb_;
};
*/



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
//FunctionOp
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////


template<typename F1, typename F2, template<typename> class BinOp, typename... Args>
requires isFunctionExpression<F1, Args...>
	&& isFunctionExpression<F2, Args...>
	&& std::same_as<typename F1::DataType, typename F2::DataType>
class FunctionOp  : public  FunctionExpression<
	FunctionOp<F1,F2,BinOp,Args...>,
	ReturnOp<typename F1::ReturnExpressionType, typename F2::ReturnExpressionType, BinOp>,
	Args...
	>
{
public:
	using DataType = typename F1::DataType;
	using ReturnExpressionType = ReturnOp<typename F1::ReturnExpressionType, typename F2::ReturnExpressionType, BinOp>;
	using BaseReturnExpressionType = typename F1::BaseReturnExpressionType;
	
	
	FunctionOp(const F1& arg1, const F2& arg2)
		:fct1_(std::make_unique<F1>(arg1)),
		fct2_(std::make_unique<F2>(arg2))
	{}
		
	FunctionOp(const FunctionOp& other)
		:fct1_(std::make_unique<F1>(*other.fct1_)),
		fct2_(std::make_unique<F2>(*other.fct2_))
	{}
	
		
	std::unique_ptr<AbstractFunction<BaseReturnExpressionType,Args...>> clone(void) const final{
		return std::make_unique<FunctionOp<F1,F2, BinOp,Args...>>(*this);	
	}
		
	ReturnExpressionType evaluate(Args... args) const{	
		//return ReturnExpressionType(fct1_->evaluate(std::forward<decltype(args)>(args)...), fct2_->evaluate(std::forward<decltype(args)>(args)...));	
		return ReturnExpressionType(fct1_->evaluate(args...), fct2_->evaluate(args...)); //arguments are rvalue refs already!
	}
	
	
private:
	std::unique_ptr<F1> fct1_;
	std::unique_ptr<F2> fct2_;
	
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
//operators +,* using FunctionOp template
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

//multiply and add are defined in ReturnExpression.hpp

template<typename F1, typename F2, typename... Args>
FunctionOp<F1,F2,add,Args...> operator+(
	const FunctionExpression<F1,typename F1::ReturnExpressionType,Args...>& arg1,
	const FunctionExpression<F2,typename F2::ReturnExpressionType,Args...>& arg2
){
	return FunctionOp<F1,F2,add,Args...>(static_cast<const F1&>(arg1),static_cast<const F2&>(arg2));
}

template<typename F1, typename F2, typename... Args>
FunctionOp<F1,F2,multiply,Args...> operator*(
	const FunctionExpression<F1,typename F1::ReturnExpressionType,Args...>& arg1,
	const FunctionExpression<F2,typename F2::ReturnExpressionType,Args...>& arg2
){
	return FunctionOp<F1,F2,multiply,Args...>(static_cast<const F1&>(arg1),static_cast<const F2&>(arg2));
}

} //end namespace funcExp
