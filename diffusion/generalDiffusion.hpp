//diffusion.hpp

#pragma once

#include <math.h>
#include "diffusion.h"
#include "../functional_expressions/FunctionalWrapper.hpp"

namespace diffusion{

template<typename F1, typename F2>
requires diffusorPair<F1,F2>
GeneralDiffusor<F1,F2>::GeneralDiffusor(
        const driftFuncExpr<F1>& driftFct,
        const diffusionFuncExpr<F2>& diffusionFct,
        int noiseDim
)
    :rng_(std::random_device{}()),
    driftFct_(static_cast<const F1&>(driftFct)), //this keeps the original function
    diffusionFct_(static_cast<const F2&>(diffusionFct)),
    noiseDim_(noiseDim)
{}


template<typename F1, typename F2>
requires diffusorPair<F1,F2>
typename GeneralDiffusor<F1,F2>::numericalType GeneralDiffusor<F1,F2>::getRandomNormal() {
    std::normal_distribution<typename GeneralDiffusor<F1,F2>::numericalType> dist(0,1);
    return dist(rng_);
}



template<typename F1, typename F2>
requires diffusorPair<F1,F2>
typename GeneralDiffusor<F1,F2>::numericalType GeneralDiffusor<F1,F2>::getRandomUnif(typename GeneralDiffusor<F1,F2>::numericalType low, typename GeneralDiffusor<F1,F2>::numericalType high){
    assert(low<high);
    std::uniform_real_distribution<double> dist(low,high);
    return dist(rng_);
}

template<typename F1, typename F2>
requires diffusorPair<F1,F2>
vec<typename GeneralDiffusor<F1,F2>::numericalType> GeneralDiffusor<F1,F2>::getRandomVector(int noiseDim, typename GeneralDiffusor<F1,F2>::numericalType scale){
    vec<typename GeneralDiffusor<F1,F2>::numericalType> randVec(noiseDim);
    for(size_t i=0; i<randVec.size(); i++) randVec[i]=getRandomNormal()*scale;
    return randVec;
}

template<typename F1, typename F2>
requires diffusorPair<F1,F2>
vec<typename GeneralDiffusor<F1,F2>::numericalType> GeneralDiffusor<F1,F2>::diffusionStep(
    const vec<typename GeneralDiffusor<F1,F2>::numericalType>& inputState,
    typename GeneralDiffusor<F1,F2>::numericalType time,
    typename GeneralDiffusor<F1,F2>::numericalType timeStep
){
    
    vec<typename GeneralDiffusor<F1,F2>::numericalType> noise = getRandomVector(noiseDim_, sqrt(timeStep));
    return vec<typename GeneralDiffusor<F1,F2>::numericalType>(inputState)
        +driftFct_(inputState,time,timeStep)
        +diffusionFct_(inputState, time, noise );
    ;
}

//virtual sop we can be more efficient for Markov processes
template<typename F1, typename F2>
requires diffusorPair<F1,F2>
vec<typename GeneralDiffusor<F1,F2>::numericalType> GeneralDiffusor<F1,F2>::sample(
    const vec<typename GeneralDiffusor<F1,F2>::numericalType>& inputState,
    typename GeneralDiffusor<F1,F2>::numericalType time,
    typename GeneralDiffusor<F1,F2>::numericalType timeStepSize
){
    //if(time<=0) throw(std::invalid_argument("diffusion::GeneralDiffusor::sample(...)need time>0"));
    //if(timeStepSize<=0) throw(std::invalid_argument("diffusion::GeneralDiffusor::sample(...)need timeStepSize>0"));

    typename GeneralDiffusor<F1,F2>::numericalType currTime=0;
    vec<typename GeneralDiffusor<F1,F2>::numericalType> res(inputState);

    while(currTime+timeStepSize<=time){
        res=diffusionStep(res,currTime,timeStepSize);
        currTime+=timeStepSize;
    }
    //second condition to keep some numerical stability
    if(currTime<time && currTime-time>timeStepSize/10) res=diffusionStep(res,currTime,time-currTime);
    
    return res;
}

template<typename F1, typename F2>
requires diffusorPair<F1,F2>
std::vector<std::pair<typename GeneralDiffusor<F1,F2>::numericalType,vec<typename GeneralDiffusor<F1,F2>::numericalType>>> samplePath(
    const vec<typename GeneralDiffusor<F1,F2>::numericalType>& inputState,
    typename GeneralDiffusor<F1,F2>::numericalType time,
    typename GeneralDiffusor<F1,F2>::numericalType timeStepSize
){
    std::vector<std::pair<typename GeneralDiffusor<F1,F2>::numericalType,vec<typename GeneralDiffusor<F1,F2>::numericalType>>> path;
    path.push_back({0,inputState});
    double currentTime=0;
    vec<typename GeneralDiffusor<F1,F2>::numericalType> currentState=inputState;
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
