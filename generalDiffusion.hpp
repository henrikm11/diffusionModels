//diffusion.hpp

#pragma once

#include <math.h>

namespace diffusion{

template<typename T>
requires funcExpr::numerical<T>
GeneralDiffusor<T>::GeneralDiffusor(
    const driftFuncExpr<T>& driftFct,
    const diffusionFuncExpr<T>& diffusionFct,
    int noiseDim
)  
    :rng_(std::random_device{}()),
    driftFct_(driftFct.clone()), //this keeps the original function
    diffusionFct_(diffusionFct.clone()),
    noiseDim_(noiseDim)
{}

template<typename T>
requires funcExpr::numerical<T>
GeneralDiffusor<T>::GeneralDiffusor(const GeneralDiffusor<T>& other)
    :rng_(std::random_device{}()),
    driftFct_(other.driftFct_->clone()),
    diffusionFct_(other.diffusionFct_->clone()), 
    noiseDim_(other.noiseDim_)
{}

template<typename T>
requires funcExpr::numerical<T>
GeneralDiffusor<T>& GeneralDiffusor<T>::operator=(const GeneralDiffusor<T>& other){
    if(this==&other) return *this;

    std::unique_ptr<driftFuncExpr<T>> newDrift = other.driftFct_->clone();
    driftFct_.reset(newDrift.release());

    std::unique_ptr<diffusionFuncExpr<T>> newDiffusion = other.diffusionFct_->clone();
    diffusionFct_.reset(newDiffusion.release());

    noiseDim_=other.noiseDim_;

    return *this;
}


//this is virtual
template<typename T>
requires funcExpr::numerical<T>
std::unique_ptr<GeneralDiffusor<T>> GeneralDiffusor<T>::clone(void) const {
    return std::make_unique<GeneralDiffusor<T>>(*this);
}

template<typename T>
requires funcExpr::numerical<T>
T GeneralDiffusor<T>::getRandomNormal() {
    std::normal_distribution<T> dist(0,1); 
    return dist(rng_);
}


template<typename T>
requires funcExpr::numerical<T>
T GeneralDiffusor<T>::getRandomUnif(T low, T high){
    assert(low<high);
    std::uniform_real_distribution<double> dist(low,high);
    return dist(rng_);
}

template<typename T>
requires funcExpr::numerical<T>
vec<T> GeneralDiffusor<T>::getRandomVector(int noiseDim, T scale){
    vec<T> randVec(noiseDim);
    for(size_t i=0; i<randVec.size(); i++) randVec[i]=getRandomNormal()*scale;
    return randVec;
}

template<typename T>
requires funcExpr::numerical<T>
vec<T> GeneralDiffusor<T>::diffusionStep(
    const vec<T>& inputState,
    T time,
    T timeStep
){
    
    vec<T> noise = getRandomVector(noiseDim_, sqrt(timeStep));
    return vec<T>(inputState)
        +std::move((*driftFct_)(inputState,time,timeStep))
        +std::move((*diffusionFct_)(inputState, time, noise ));
    ;
}

//virtual sop we can be more efficient for Markov processes
template<typename T>
requires funcExpr::numerical<T>
vec<T> GeneralDiffusor<T>::sample(
    const vec<T>& inputState,
    T time,
    T timeStepSize
){
    //if(time<=0) throw(std::invalid_argument("diffusion::GeneralDiffusor::sample(...)need time>0"));
    //if(timeStepSize<=0) throw(std::invalid_argument("diffusion::GeneralDiffusor::sample(...)need timeStepSize>0"));

    T currTime=0;
    vec<T> res(inputState);

    while(currTime+timeStepSize<=time){
        res=diffusionStep(res,currTime,timeStepSize);
        currTime+=timeStepSize;
    }
    //second condition to keep some numerical stability
    if(currTime<time && currTime-time>timeStepSize/10) res=diffusionStep(res,currTime,time-currTime);
    
    return res;
}

template<typename T>
requires funcExpr::numerical<T>
std::vector<std::pair<T,vec<T>>> samplePath(
    const vec<T>& inputState,
    T time,
    T timeStepSize
){
    std::vector<std::pair<T,vec<T>>> path;
    path.push_back({0,inputState});
    double currentTime=0;
    vec<T> currentState=inputState;
    while(currentTime+timeStepSize<time){
        currentState=diffusionStep(
            currentState,
            currentTime,
            timeStepSize
        );
        path.push_back({currentTime+timeStepSize,currentState});
        currentTime+=timeStepSize;
    }
    //second condition to keep some numerical stability
    if(currentTime<time && currentTime-time>timeStepSize/10){
        currentState=diffusionStep(
            currentState,
            currentTime,
            time-currentTime 
        );
        path.push_back({time,currentState});
    } 
    return path;
}




} //end of namespace diffusion