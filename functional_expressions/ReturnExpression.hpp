//ReturnExpression.hpp


#pragma once


#include<vector>
#include <concepts>
#include <ranges>
#include <initializer_list>


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
		
	
	ReturnOp(const E1& arg1, const E1& arg2)
		:arg1_(arg1),
		arg2_(arg2)
	{
		constexpr bool noWrapperVectorMix = isScalar || ((E1::isWrapper && E2::isWrapper) || (!E1::isWrapper && !E2::isWrapper));
		static_assert(noWrapperVectorMix);
		assert(arg1_.size()==arg2_.size() || arg1_.size()==1 || arg2_.size()==1);		
	}
	
	ReturnOp(E1&& arg1, E2&& arg2)
		:arg1_(std::move(arg1)),
		arg2_(std::move(arg2))
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

/*
template<typename E1, typename E2>
ReturnOp<removE1,E2,add> operator+(ReturnExpression<E1, typename E1::DataType>&& arg1, ReturnExpression<E2, typename E2::DataType>&& arg2){
	return ReturnOp<E1,E2,add>(static_cast<E1&&>(arg1),static_cast<E2&&>(arg2));
}
*/

template<typename E1, typename E2>
requires isReturnExpression<std::remove_reference_t<E1>> && isReturnExpression<std::remove_reference_t<E2>>
ReturnOp<std::remove_reference_t<E1>,std::remove_reference_t<E2>,add> operator+(E1&& arg1, E2&& arg2){
	return ReturnOp<E1,E2,add>(std::forward<E1>(arg1),std::forward<E2>(arg2));
}

/*
template<typename E1, typename E2>
ReturnOp<E1,E2,multiply> operator*(ReturnExpression<E1, typename E1::DataType>&& arg1, ReturnExpression<E2, typename E2::DataType>&& arg2){
	return ReturnOp<E1,E2,multiply>(static_cast<E1&&>(arg1),static_cast<E2&&>(arg2));
}
*/

template<typename E1, typename E2>
requires isReturnExpression<std::remove_reference_t<E1>> && isReturnExpression<std::remove_reference_t<E2>>
ReturnOp<E1,E2,multiply> operator*(E1&& arg1,E2&& arg2){
	return ReturnOp<E1,E2,multiply>(std::forward<E1>(arg1),std::forward<E2>(arg2));
}


} //end of namespace funcExpr





