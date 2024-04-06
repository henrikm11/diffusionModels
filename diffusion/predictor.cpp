//predictor.cpp

#include "predictor.h"


namespace predictor{

Model::Model(int val):val_(val){}
    
double Model::predict(double t) const {return val_*t;}
double Model::predict(double x, double t) const {return val_*x*t;}
std::vector<double> Model::predict(const std::vector<double>&  X, double t) const{
    std::vector<double> res(X);
    for(auto& e : res){e*=val_*t;}
    return res;
}


}


