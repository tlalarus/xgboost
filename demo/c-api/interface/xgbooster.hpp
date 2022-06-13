/**
* @name    : xgbcpp.hpp
* @author  : Sim MinKyung
* @brief   : Interface to import xgboost for classification
*/

#ifndef __XGBOOSTER_H__
#define __XGBOOSTER_H__

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <memory>

#include <xgboost/c_api.h>
#include <xgboost/parameter.h>
#include <xgboost/objective.h>
#include <xgboost/learner.h>
#include <xgboost/data.h>


namespace xgboost{

// Row-major matrix
struct _Matrix{
    float *data;
    size_t shape[2]; // (rows, cols)

    char _array_interface[256];
};

typedef struct _Matrix* Matrix;

class Booster{
public:
    Booster();
    void load_model(std::string model_path);
    void init_data(int rows_, int cols_);
    void init_model();
    void predict_spin(float* feat_);

    size_t get_nsamples(Matrix self);
    size_t get_nfeatures(Matrix self);
    float at_(Matrix self, size_t i, size_t j);

    void free_(Matrix self);
private:
    void create_matrix(Matrix* self, float const* data, size_t n_samples,
            size_t n_features);

public:
    Matrix X_pred;
    std::unique_ptr<Learner> xgb_learner;

    
    DMatrixHandle dtest_X;
    BoosterHandle booster;
    
};


};

#endif //__XGBOOSTER_H__