/*
 *   All functions related to cascade training
 */

#include "node-fann.h"

void NNet::CascadeTrainOnData(struct fann_train_data *traindata, unsigned int max_neurons, unsigned int neurons_between_reports, float desired_error)
{
  fann_cascadetrain_on_data(FANN, traindata, max_neurons, neurons_between_reports, desired_error);
}

NAN_METHOD(NNet::CascadeTrain)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  struct fann_train_data *traindata = NULL;

  if (info.Length() < 1)
    return Nan::ThrowError("No arguments supplied");

  if (!info[0]->IsArray())
    return Nan::ThrowError("First argument should be 2d-array (training data set)");

  Local<Array> dataset = info[0].As<Array>();
  net->MakeTrainData(dataset, &traindata);

  if (traindata == NULL)
    return Nan::ThrowError("Internal error");


  unsigned int max_neurons = 100000;
  unsigned int neurons_between_reports = 1000;
  float desired_error = 0.001;
  int scale = 0;
  if (info.Length() >= 2) {
    Local<Object> params = info[1].As<Object>();
    if (params->Has(Nan::New<String>("neurons").ToLocalChecked())) {
      max_neurons = params->Get(Nan::New<String>("neurons").ToLocalChecked())->IntegerValue();
    }
    if (params->Has(Nan::New<String>("neurons_between_reports").ToLocalChecked())) {
      neurons_between_reports = params->Get(Nan::New<String>("neurons_between_reports").ToLocalChecked())->IntegerValue();
    }
    if (params->Has(Nan::New<String>("error").ToLocalChecked())) {
      desired_error = params->Get(Nan::New<String>("error").ToLocalChecked())->NumberValue();
    }
    if (params->Has(Nan::New<String>("scale").ToLocalChecked())) {
      scale = params->Get(Nan::New<String>("scale").ToLocalChecked())->BooleanValue();
    }
  }
  if (scale) {
    fann_scale_train(net->FANN, traindata);
    net->scale_present = true;
  }
  net->CascadeTrainOnData(traindata, max_neurons, neurons_between_reports, desired_error);
  fann_destroy_train(traindata);
  return;
}

