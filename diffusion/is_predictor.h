//is_predictor.h
#pragma once



template<typename T>
struct is_predictor{
    const static bool value = false;
};

namespace predictor{
class Model;
}

template<>
struct is_predictor<predictor::Model>{
    const static bool value = true;
};