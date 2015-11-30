/*
 *   All functions related to creation and running a network
 */

#include "node-fann.h"

NNet::NNet()
{
  FANN = NULL;
  scale_present = false;
}

NNet::~NNet()
{
  if (FANN != NULL) {
    fann_destroy(FANN);
    FANN = NULL;
  }
}

int NNet::_GetLayersFromArray(unsigned int *&layers, Local<Array> a)
{
  Nan::HandleScope scope;
  int len = a->Length();
  if (len < 2)
    return 0;

  layers = new unsigned int[len];
  for (unsigned i=0; i<a->Length(); i++) {
    int n = a->Get(i)->IntegerValue();
    if (n < 1)
      return 0;
    layers[i] = n;
  }
  return len;
}

NAN_METHOD(NNet::NewStandard)
{
  Nan::HandleScope scope;
  NNet *net = new NNet();
  net->Wrap(info.This());
  net->CreateStandard(info);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(NNet::NewSparse)
{
  Nan::HandleScope scope;
  NNet *net = new NNet();
  net->Wrap(info.This());
  net->CreateSparse(info);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(NNet::NewShortcut)
{
  Nan::HandleScope scope;
  NNet *net = new NNet();
  net->Wrap(info.This());
  net->CreateShortcut(info);
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(NNet::NewFromFile)
{
  Nan::HandleScope scope;
  NNet *net = new NNet();
  net->Wrap(info.This());
  net->CreateFromFile(info);
  info.GetReturnValue().Set(info.This());
}

/* for FANN >= 2.2.0
NAN_METHOD(NNet::CloneNet)
{
    Nan::HandleScope scope;
//    if (!NNet::HasInstance(info[0]))
//      return Nan::ThrowError("First argument must be existing network.");
  NNet *net = new NNet();
  net->Wrap(info.This());
  net->CreateClone(info);
    info.GetReturnValue().Set(info.This());
}*/

NAN_METHOD(NNet::CreateStandard)
{
  unsigned int* layers = NULL;
  int len = 0;

  if (info.Length() < 1)
    return Nan::ThrowError("No arguments supplied");

  if (info[0]->IsArray()) {
    len = _GetLayersFromArray(layers, info[0].As<Array>());
  } else {
    Local<Array> arr = Nan::New<Array>();
    for (int i=0; i<info.Length(); i++) {
      arr->Set(i, info[i]);
    }
    len = _GetLayersFromArray(layers, arr);
  }
  if (len <= 0) {
    if (layers != NULL) delete[] layers;
    return Nan::ThrowError("Wrong arguments supplied");
  }

  FANN = fann_create_standard_array(len, layers);
  if (FANN == NULL)
    return Nan::ThrowError("Failed to create neural network");

/*const float desired_error = (const float) 0.001;
const unsigned int max_epochs = 500000;
const unsigned int epochs_between_reports = 1000;
fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

fann_train_on_file(ann, "xor.data", max_epochs, epochs_between_reports, desired_error);

fann_save(ann, "xor_float.net");

fann_destroy(ann);*/
  delete[] layers;
  return;
}

NAN_METHOD(NNet::CreateSparse)
{
  unsigned int* layers = NULL;
  int len = 0;

  if (info.Length() < 1)
    return Nan::ThrowError("No arguments supplied");

  if (!info[0]->IsNumber())
    return Nan::ThrowError("First argument should be float");

  if (info[1]->IsArray()) {
    len = _GetLayersFromArray(layers, info[1].As<Array>());
  } else {
    Local<Array> arr = Nan::New<Array>();
    /* skip 1st argument here */
    for (int i=1; i<info.Length(); i++) {
      arr->Set(i-1, info[i]);
    }
    len = _GetLayersFromArray(layers, arr);
  }
  if (len <= 0) {
    if (layers != NULL) delete[] layers;
    return Nan::ThrowError("Wrong arguments supplied");
  }

  FANN = fann_create_sparse_array(info[0]->NumberValue(), len, layers);
  if (FANN == NULL)
    return Nan::ThrowError("Failed to create neural network");

  delete[] layers;
  return;
}

NAN_METHOD(NNet::CreateShortcut)
{
  unsigned int* layers = NULL;
  int len = 0;

  if (info.Length() < 1)
    return Nan::ThrowError("No arguments supplied");

  if (info[0]->IsArray()) {
    len = _GetLayersFromArray(layers, info[0].As<Array>());
  } else {
    Local<Array> arr = Nan::New<Array>();
    for (int i=0; i<info.Length(); i++) {
      arr->Set(i, info[i]);
    }
    len = _GetLayersFromArray(layers, arr);
  }
  if (len <= 0) {
    if (layers != NULL) delete[] layers;
    return Nan::ThrowError("Wrong arguments supplied");
  }

  FANN = fann_create_shortcut_array(len, layers);
  if (FANN == NULL)
    return Nan::ThrowError("Failed to create neural network");

  delete[] layers;
  return;
}

NAN_METHOD(NNet::CreateFromFile)
{
  Nan::HandleScope scope;
  if (info.Length() != 1 || !info[0]->IsString())
    return Nan::ThrowError("usage: new FANN.load(\"filename.nnet\")");

  String::Utf8Value name(info[0].As<String>());

  FANN = fann_create_from_file(*name);
  if (FANN == NULL)
    return Nan::ThrowError("Failed to create neural network");

  return;
}

NAN_METHOD(NNet::SaveToFile)
{
  Nan::HandleScope scope;
  if (info.Length() != 1 || !info[0]->IsString())
    return Nan::ThrowError("usage: net.save(\"filename.nnet\")");

  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  String::Utf8Value name(info[0].As<String>());

  fann_save(net->FANN, *name);
  return;
}

/* for FANN >= 2.2.0
NAN_METHOD(NNet::CreateClone)
{
  NNet *currnet = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  NNet *oldnet = Nan::ObjectWrap::Unwrap<NNet>(info[0]->ToObject());
  currnet->FANN = fann_copy(oldnet->FANN);
//  printf("!!!!!!!!!!!!!! %d %d\n", currnet->something, oldnet->something);
  return;
}*/

NAN_METHOD(NNet::Run)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  if (info.Length() < 1)
    return Nan::ThrowError("No arguments supplied");
  if (!info[0]->IsArray())
    return Nan::ThrowError("First argument should be array");

  Local<Array> datain = info[0].As<Array>();
  fann_type *dataset_in = new fann_type[datain->Length()];
  for (unsigned i=0; i<datain->Length(); i++) {
    dataset_in[i] = datain->Get(i)->NumberValue();
  }

  fann_type *result = fann_run(net->FANN, dataset_in);

  if (net->scale_present) {
    fann_descale_output(net->FANN, result);
  }

  int dim = fann_get_num_output(net->FANN);
  Local<Array> result_arr = Nan::New<Array>(dim);
  for (int i=0; i<dim; i++) {
    result_arr->Set(i, Nan::New<Number>(result[i]));
  }

  info.GetReturnValue().Set(result_arr);
}

