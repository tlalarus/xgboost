/**
* @name    : xgbcpp.cpp
* @author  : Sim MinKyung
* @brief   : interface to import xgboost for classification
*
* Copyright(c) Golfzon Co.. Ltd.
* All rights reserved.
*/

#include <assert.h>
#include <vector>
#include <chrono>
#include "xgbooster.hpp"

using namespace std;
using namespace std::chrono;
using Clock = std::chrono::high_resolution_clock;
using SYSTIME = std::chrono::time_point<Clock>;
using DURATION = std::chrono::high_resolution_clock::duration;

#define N_FEAT 10

namespace xgboost{

#define safe_xgboost(err)                                                      \
  if ((err) != 0) {                                                            \
    fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #err,      \
            XGBGetLastError());                                                \
    exit(1);                                                                   \
  }

#define safe_malloc(ptr)                                                       \
  if ((ptr) == NULL) {                                                         \
    fprintf(stderr, "%s:%d: Failed to allocate memory.\n", __FILE__,           \
            __LINE__);                                                         \
    exit(1);                                                                   \
  }


Booster::Booster()
{
  safe_xgboost(XGBoosterCreate(NULL, 0, &booster));
}
void Booster::load_model(std::string model_path)
{
  //xgb_learner = std::make_unique<Learner>(Learner::Create());
  safe_xgboost(XGBoosterLoadModel(booster, model_path.c_str()));
  cout << "model loaded " << model_path << endl;

  bst_ulong n_feat = 0;
  safe_xgboost(XGBoosterGetNumFeature(booster, &n_feat));
  cout << "num of feature: " << n_feat << endl;

}
void Booster::init_data(int rows_, int cols_)
{

}
void Booster::init_model()
{

}

void Booster::create_matrix(Matrix* self, float const* data, size_t n_samples,
            size_t n_features)
{

}

size_t Booster::get_nsamples(Matrix self){
  return self->shape[0];
}

size_t Booster::get_nfeatures(Matrix self){
  return self->shape[1];
}

float Booster::at_(Matrix self, size_t i, size_t j){
    return self->data[i*self->shape[1] + j];
}

void Booster::free_(Matrix self){
  if(self != NULL){
      if(self->data != NULL){
          self->shape[0] = 0;
          self->shape[1] = 0;
          free(self->data);
          self->data = NULL;
      }
      free(self);
  }
}

void Booster::predict_spin(float* feat_)
{
  cout << "Print features: ";
  for(int i=0; i<N_FEAT; i++){
    cout << feat_[i] << " ";
  }
  cout << endl;

  // convert feature array to DMatrix
  XGDMatrixCreateFromMat(reinterpret_cast<float*>(feat_),
      1, N_FEAT, NAN, &dtest_X);


  bst_ulong out_len;
  const float* f;

  SYSTIME start = std::chrono::high_resolution_clock::now();

  XGBoosterPredict(booster, dtest_X, 0, 0, 0, &out_len, &f);
  cout << "Result: " << *f << endl;

  DURATION dur = std::chrono::high_resolution_clock::now() - start;
  auto dur_ = duration_cast<std::chrono::microseconds>(dur);

  cout << "duration(us): "<< dur_.count() << endl;

  // c++ style
  //xgb_learner->Predict((DMatrix*)dtest_X, true, &preds);
}

};


using namespace xgboost;

int main(){

  float feat[N_FEAT] = {
    0.035579f,0.999826f,6.608611f,0.999760f,0.011427f,
    7.044868f,4.242640f,0.062649f,0.039692f,0.f
  };


  Booster xgbooster;
  xgbooster.load_model("./model.json");
  xgbooster.predict_spin(feat);

  

  return 0;
}