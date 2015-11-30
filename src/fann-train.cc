/*
 *   All functions related to training a network
 */

#include "node-fann.h"

fann_type *dataset_raw;
void FANN_API _TrainCB_FromRAW(unsigned int num, unsigned int innum, unsigned int outnum, fann_type *in, fann_type *out) {
  for (unsigned c=0; c<innum; c++) {
    in[c] = dataset_raw[num*(innum+outnum) + c];
  }
  for (unsigned c=0; c<outnum; c++) {
    out[c] = dataset_raw[num*(innum+outnum) + innum + c];
  }
}

const char* NNet::MakeTrainData(Local<Array> dataset, struct fann_train_data **traindata)
{
  unsigned input_num = fann_get_num_input(FANN);
  unsigned output_num = fann_get_num_output(FANN);

  dataset_raw = new fann_type[(input_num+output_num) * dataset->Length()];
  fann_type *dataset_raw_ptr = dataset_raw;
  //fann_set_user_data(net->FANN, (void*)dataset_raw);
  for (unsigned i=0; i<dataset->Length(); i++) {
    Local<Value> t = dataset->Get(i);
    if (!t->IsArray()) {
      return "First argument should be 2d-array (training data set)";
    }
    Local<Array> data = t.As<Array>();
    if (data->Length() != 2) {
      return "Incorrect dataset";
    }
    Local<Value> tin = data->Get(0);
    Local<Value> tout = data->Get(1);
    if (!tin->IsArray() || !tout->IsArray()) {
      return "Incorrect dataset";
    }

    Local<Array> datain = tin.As<Array>();
    Local<Array> dataout = tout.As<Array>();
    if (datain->Length() != input_num) {
      return "Incorrect dataset (input length doesn't match input layer)";
    }
    if (dataout->Length() != output_num) {
      return "Incorrect dataset (output length doesn't match output layer)";
    }
    for (unsigned c=0; c<input_num; c++) {
      *dataset_raw_ptr = datain->Get(c)->NumberValue();
      dataset_raw_ptr++;
    }
    for (unsigned c=0; c<output_num; c++) {
      *dataset_raw_ptr = dataout->Get(c)->NumberValue();
      dataset_raw_ptr++;
    }
  }

  *traindata = fann_create_train_from_callback(dataset->Length(), input_num, output_num, _TrainCB_FromRAW);
  delete[] dataset_raw;

  return NULL;
}

void NNet::TrainOnData(struct fann_train_data *traindata, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error)
{
  fann_train_on_data(FANN, traindata, max_epochs, epochs_between_reports, desired_error);
}

NAN_METHOD(NNet::Train)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  struct fann_train_data *traindata = NULL;

  if (info.Length() < 1)
    return Nan::ThrowError("No arguments supplied");

  if (!info[0]->IsArray())
    return Nan::ThrowError("First argument should be 2d-array (training data set)");

  Local<Array> dataset = info[0].As<Array>();

  const char* error = net->MakeTrainData(dataset, &traindata);
  if (error != NULL)
    return Nan::ThrowError(error);

  if (traindata == NULL)
    return Nan::ThrowError("Internal error");

  unsigned int max_epochs = 100000;
  unsigned int epochs_between_reports = 1000;
  float desired_error = 0.001;
  int scale = 0;
  if (info.Length() >= 2) {
    Local<Object> params = info[1].As<Object>();
    if (params->Has(Nan::New<String>("epochs").ToLocalChecked())) {
      max_epochs = params->Get(Nan::New<String>("epochs").ToLocalChecked())->IntegerValue();
    }
    if (params->Has(Nan::New<String>("epochs_between_reports").ToLocalChecked())) {
      epochs_between_reports = params->Get(Nan::New<String>("epochs_between_reports").ToLocalChecked())->IntegerValue();
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
  net->TrainOnData(traindata, max_epochs, epochs_between_reports, desired_error);
  fann_destroy_train(traindata);
  return;
}

NAN_METHOD(NNet::TrainOnce)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  if (info.Length() < 2)
    return Nan::ThrowError("2 arguments needed");

  if (!info[0]->IsArray())
    return Nan::ThrowError("First argument should be array (training input)");
  if (!info[1]->IsArray())
    return Nan::ThrowError("Second argument should be array (training output)");

  Local<Array> datain = info[0].As<Array>();
  Local<Array> dataout = info[1].As<Array>();
  fann_type *dataset_in = new fann_type[datain->Length()];
  fann_type *dataset_out = new fann_type[dataout->Length()];
  for (unsigned i=0; i<datain->Length(); i++) {
    dataset_in[i] = datain->Get(i)->NumberValue();
  }
  for (unsigned i=0; i<dataout->Length(); i++) {
    dataset_out[i] = dataout->Get(i)->NumberValue();
  }

  fann_train(net->FANN, dataset_in, dataset_out);

  delete[] dataset_in;
  delete[] dataset_out;
  return;
}

