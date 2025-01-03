//ReturnExpression.hpp
#pragma once


#include <cassert>
#include<vector>
#include <concepts>
#include <ranges>
#include <initializer_list>

//TODO
/*
*/


namespace funcExpr{
	


///////////////////////////////////////////////////////////////
template<typename T>
concept numerical = std::integral<T> || std::floating_point<T>;
///////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// ReturnExpression
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

	
template<typename E, typename T>
requires numerical<T>
class ReturnExpression{
public:
	using DataType = T;
	using ExpressionType = E;
	
	DataType operator[](const size_t i) const {
		return static_cast<const E*>(this)->operator[](i);
	}
	
	constexpr size_t size(void) const {
		return static_cast<const E*>(this)->size();
	}
	
	//this is taking a ~20% performance hit
	//type is std::conditional_t<E::isScalar, DataType, std::vector<DataType>>
	decltype(auto) convertToStdType(void){
		if constexpr(E::isScalar) return *this[0];
		std::vector<DataType> converted(size());
		#pragma clang loop vectorize(enable) interleave(enable)
		for(size_t i=0; i<size(); i++){
			converted[i]=(*this)[i];
		}
		return converted;
	}
	
protected:
	virtual ~ReturnExpression(void) = default;
};



///////////////////////////////////////////////////////////////
template<typename E>
concept isReturnExpression = std::derived_from<E,ReturnExpression<E,typename E::DataType>>;
//note that this is not satisfied for ReturnExpression itself!

template<typename E>
concept retExprRef = isReturnExpression<std::remove_cvref_t<E>> || std::same_as<std::remove_cvref_t<E>,ReturnExpression<typename std::remove_cvref_t<E>::ExpressionType, typename std::remove_cvref_t<E>::DataType>>;

///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// ReturnScalar
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

template<typename T>
class ReturnScalar : public ReturnExpression<ReturnScalar<T>,T> {
public:
	using DataType = T;
	using BaseReturnExpressionType = ReturnScalar<T>;
	constexpr static bool isScalar = true;
	constexpr static bool isWrapper = false;
	
	
	ReturnScalar(DataType data): data_(data){}
	~ReturnScalar(void) final = default;
	//everything else is fully default
	
	
	template<typename E>
	ReturnScalar(const ReturnExpression<E,T>& expr)
		: data_(expr[0])
	{
		static_assert(E::isScalar);
	}
	
	DataType getValue(void){return data_;}
	//made for broadcasting
	DataType operator[](const size_t i) const {return data_;}
	DataType& operator[](const size_t i){return data_;}
	constexpr size_t size(void) const {return 1;}
	
	
private:
	DataType data_;
};



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// ReturnVector
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

template<typename T>
class ReturnVector : public ReturnExpression<ReturnVector<T>,T> {
public:
	using DataType = T;
	using BaseReturnExpressionType = ReturnVector<T>;
	constexpr static bool isScalar = false;
	constexpr static bool isWrapper = false;
	
	ReturnVector(size_t s): data_(new T[s]),size_(s){}
	ReturnVector(std::initializer_list<T> l)
		:data_(new T[l.size()]),
		size_(l.size())
	{
		size_t idx = 0;
		for(auto it=l.begin(); it!=l.end(); it++){
			data_[idx]=*it;
			idx++;
		}		
	}
	
	ReturnVector(const ReturnVector& other) 
		:data_(new T[other.size()]),
		size_(other.size())
	{
		#pragma clang loop vectorize(enable) interleave(enable)
		for(size_t i=0; i<size_; i++) data_[i]=other.data_[i];
	}
	
	
	ReturnVector& operator=(const ReturnVector& other){
		//no need to check against self assignment here as we don't deallocate anything
		assert(size_==other.size());
		#pragma clang loop vectorize(enable) interleave(enable)
		for(size_t i=0; i<size_; i++) data_[i]=other.data_[i];
		return *this;
	}
	
	ReturnVector(ReturnVector&& other)
		:data_(other.data_), 
		size_(other.size_)
	{
		other.data_=nullptr;
		other.size_=0;
	}
	
	ReturnVector& operator=(ReturnVector&& other){
		if(this==&other) return *this;
		delete[] data_;
		data_=other.data_;
		other.data_=nullptr;
		other.size_=0;
		return *this;
	}
	~ReturnVector(void) final {delete[] data_;}
	
	template<typename E>
	ReturnVector(const ReturnExpression<E,T>& expr)
		:data_(new T[expr.size()]), size_(expr.size())
	{
		#pragma clang loop vectorize(enable) interleave(enable)
		for(size_t i=0; i<size_; ++i) data_[i]=expr[i];
	}
	
	
	DataType operator[](const size_t i) const {return data_[i];}
	DataType& operator[](const size_t i){return data_[i];}
	
	constexpr size_t size(void) const {return size_;}
	
private:
	
	T* data_;
	size_t size_;

};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// ReturnWrapper
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

//right now this requires Container to have random access in range [0,size_)

template<typename T, template<typename, typename> class Container,  typename Allocator = std::allocator<T>>
requires std::random_access_iterator<Container<T,Allocator>>
class ReturnWrapper : public ReturnExpression<ReturnWrapper<T,Container, Allocator>,T> {
public:
	using DataType = T;
	using BaseReturnExpressionType = ReturnWrapper<T,Container,Allocator>;
	constexpr static bool isScalar = false;
	constexpr static bool isWrapper = true;
	
	//ctors
	ReturnWrapper(size_t size): data_(size), size_(size){}
	ReturnWrapper(const Container<DataType, Allocator>& data): data_(data), size_(data.size()){}
	//copy
	ReturnWrapper(const ReturnWrapper& other) = default;
	ReturnWrapper& operator=(const ReturnWrapper& other) = default;
	
	//move both from same type or container
	ReturnWrapper(Container<DataType,Allocator>&& data): data_(std::move(data)), size_(data.size()){}
	ReturnWrapper(ReturnWrapper&& other): data_(std::move(other.data_)), size_(other.size_){}
	ReturnWrapper& operator=(ReturnWrapper&& other){
		if(this==&other) return *this;
		data_=std::move(other.data_);
		size_=other.size_;
		return *this;
	}
	//dtor
	~ReturnWrapper(void) final = default;
	
	template<typename E>
	ReturnWrapper(const ReturnExpression<E,DataType>& expr)
		:data_(expr.size()), size_(expr.size())
	{
		#pragma clang loop vectorize(enable) interleave(enable)
		for(size_t i=0; i<size_; ++i) data_[i]=expr[i];
	}
	
	constexpr size_t size(void) const {return size_;}
	

	Container<DataType,Allocator> detach(void){
		Container<DataType,Allocator>&& res = std::move(data_);
		return res;
	}
	
	
	DataType operator[](const size_t i) const {return data_[i];}
	DataType& operator[](const size_t i){return data_[i];}
	
private:
	Container<DataType,Allocator> data_;
	size_t size_;
};




///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
// ReturnOp
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
//no reference and cv types allowed here
//but ok in call from operators
//casting is done there
template<typename E1, typename E2, template<typename> class BinOp>
requires 
	isReturnExpression<E1> 
	&& isReturnExpression<E2> 
	&& std::same_as<typename E1::DataType, typename E2::DataType>
class ReturnOp : public ReturnExpression<ReturnOp<E1,E2,BinOp>, typename E1::DataType>{
public:
	using DataType = typename E1::DataType; //guaranteed to be same as for E2 by requires  
	constexpr static bool isScalar = E1::isScalar && E2::isScalar;
	constexpr static bool isWrapper = E1::isWrapper || E2::isWrapper;
	using BaseReturnExpressionType = std::conditional_t<
		isScalar, 
		ReturnScalar<DataType>,
		std::conditional_t<
			E1::isScalar,
			typename E2::BaseReturnExpressionType,
			typename E1::BaseReturnExpressionType
			>
		>;
		

	//templated with aliases to have perfect forwarding
	template<typename E1_alias, typename E2_alias>
	requires 
		std::same_as<std::remove_cvref_t<E1_alias>, std::remove_cvref_t<E1>>
		&& std::same_as<std::remove_cvref_t<E2_alias>, std::remove_cvref_t<E2>>
	ReturnOp(E1_alias&& arg1, E2_alias&& arg2)
		:arg1_(std::forward<E1_alias>(arg1)),
		arg2_(std::forward<E2_alias>(arg2))
	{
		constexpr bool noWrapperVectorMix = isScalar || ((E1::isWrapper && E2::isWrapper) || (!E1::isWrapper && !E2::isWrapper));
		static_assert(noWrapperVectorMix);
		assert(arg1_.size()==arg2_.size() || arg1_.size()==1 || arg2_.size()==1);		
	}
	
	
	ReturnOp(const ReturnOp& other) = default;
	ReturnOp& operator=(const ReturnOp& other) = default;
	
	ReturnOp(ReturnOp&& other)
		:arg1_(std::move(other.arg1_)),
		arg2_(std::move(other.arg2_))
	{}
	
	
	ReturnOp& operator=(ReturnOp&& other){
		if(this==&other) return *this;
		//corresponding move assignments take care of ressource management
		arg1_=std::move(other.arg1_);
		arg2_=std::move(other.arg2_);
		return *this;
	}

		
	~ReturnOp(void) final = default;
	
	
	
	DataType operator[](const size_t i) const {return op(arg1_[i],arg2_[i]);}
	constexpr size_t size(void) const {return std::max(arg1_.size(),arg2_.size());}
	
	
private:
	E1 arg1_; //ownership gets transfered here
	E2 arg2_; //same here
	static constexpr BinOp<DataType> op;// = BinOp<DataType>();
};



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
////fowarding for static polymorphic types
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
//this is helpful for overload resolution and perfect forwarding in operators+* below


//this does not fully take into account cv qualifieres yet!


//overloads to get cv type	
template<typename Base, typename Derived>
struct cvType{
	using type = Derived;
};


template<typename Base, typename Derived>
struct cvType<const Base, Derived>{
	using type = const Derived;
};


template<typename Base, typename Derived>
struct cvType<volatile Base, Derived>{
	using type = volatile Derived;
};

template<typename Base, typename Derived>
struct cvType<const volatile Base, Derived>{
	using type = const volatile Derived;
};

template<typename Base, typename Derived>
using cvType_t = typename cvType<Base,Derived>::type;


//overloads to get reference type
template<typename Base, typename Derived>
struct refType{
	using type = Derived;
};

template<typename Base, typename Derived>
struct refType<Base&, Derived>{
	using type = Derived&;
};

template<typename Base, typename Derived>
struct refType<Base&&, Derived>{
	using type = Derived&&;
};

template<typename Base, typename Derived>
using refType_t = typename refType<Base,Derived>::type;


//combine them to get cv reference type
template<typename Base, typename Derived>
using cvRefType_t = refType_t<Base, cvType_t<std::remove_reference_t<Base>,std::remove_reference_t<Derived>>>;


//forwarding operator for statically polymorphic types
template<typename Base, typename Derived>
constexpr cvRefType_t<Base, Derived> static_cast_forward(std::remove_reference_t<Base>& d) noexcept {
	static_assert(std::is_base_of_v<std::remove_reference_t<Base>,std::remove_reference_t<Derived>>);
	return static_cast<cvRefType_t<Base,Derived>>(d);
}

template<typename Base, typename Derived>
constexpr cvRefType_t<Base, Derived> static_cast_forward(std::remove_reference_t<Base>&& d) noexcept{
	static_assert(std::is_base_of_v<std::remove_reference_t<Base>,std::remove_reference_t<Derived>>);
	return static_cast<cvRefType_t<Base,Derived>>(d);
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
//operators +,* using ReturnOp template
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

template<typename T>
requires requires (T x, T y) {x+y;} //these are fine without std::remove_reference
class add{
public:
	constexpr T operator()(const T& arg1, const T& arg2) const {return arg1+arg2;}
};

template<typename T>
requires requires (T x, T y) {x*y;}
class multiply{
public:
	constexpr T operator ()(const T& arg1, const T& arg2) const {return arg1*arg2;}
};


//operator+
//E1 and E2 can be ReturnExpression or derived from it
//in the former case we static_cast them to their full type preserving reference and cv type
//use the resulting types to call constructor of ReturnOP
template<typename E1, typename E2>
requires retExprRef<E1> && retExprRef<E2>
ReturnOp<
	typename std::remove_cvref_t<E1>::ExpressionType,
	typename std::remove_cvref_t<E2>::ExpressionType,
	add
	>
	operator+(
		E1&& arg1,
		E2&& arg2
	){
		return ReturnOp<
			typename std::remove_cvref_t<E1>::ExpressionType,
			typename std::remove_cvref_t<E2>::ExpressionType,
			add
			>(
				static_cast_forward<E1&&, typename std::remove_reference_t<E1>::ExpressionType>(arg1),
				static_cast_forward<E2&&, typename std::remove_reference_t<E2>::ExpressionType>(arg2)
			)
		;
}

//operator*
template<typename E1, typename E2>
requires retExprRef<E1> && retExprRef<E2>
ReturnOp<
	typename std::remove_reference_t<E1>::ExpressionType,
	typename std::remove_reference_t<E2>::ExpressionType,
	multiply
	>
	operator*(
		E1&& arg1,
		E2&& arg2
	){
		return ReturnOp<
			typename std::remove_reference_t<E1>::ExpressionType,
			typename std::remove_reference_t<E2>::ExpressionType,
			multiply
			>(
				static_cast_forward<E1&&, typename std::remove_reference_t<E1>::ExpressionType>(arg1),
				static_cast_forward<E2&&, typename std::remove_reference_t<E2>::ExpressionType>(arg2)
			)
		;
}



} //end of namespace funcExpr




