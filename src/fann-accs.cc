/*
 *   All setters, getters and other information providers
 */

#include <string.h>
#include <stdio.h>
#include "node-fann.h"

NAN_GETTER(NNet::GetTrainingAlgorithm)
{
  Nan::HandleScope scope;
  Local<Object> self = info.Holder();
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(self);
  int size = sizeof(FANN_TRAIN_NAMES)/sizeof(char*);
  enum fann_train_enum algo = fann_get_training_algorithm(net->FANN);

  if (algo >= 0 && algo < size) {
    info.GetReturnValue().Set(NormalizeName(FANN_TRAIN_NAMES[algo], TRAIN_PREFIX, sizeof(TRAIN_PREFIX)-1));
  } else {
    return;
  }
}

NAN_METHOD(NNet::GetNetworkType)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  int size = sizeof(FANN_NETTYPE_NAMES)/sizeof(char*);
  enum fann_nettype_enum ret = fann_get_network_type(net->FANN);

  if (ret >= 0 && ret < size) {
    info.GetReturnValue().Set(NormalizeName(FANN_NETTYPE_NAMES[ret], NETTYPE_PREFIX, sizeof(NETTYPE_PREFIX)-1));
  } else {
    return;
  }
}

NAN_SETTER(NNet::SetTrainingAlgorithm)
{
  Nan::HandleScope scope;
  Local<Object> self = info.Holder();
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(self);
  int size = sizeof(FANN_TRAIN_NAMES)/sizeof(char*);
  int num = -1;

  if (value->IsString()) {
    num = _SeekCharArray(value.As<String>(), FANN_TRAIN_NAMES, size, TRAIN_PREFIX);
  } else if (value->IsNumber()) {
    num = value->NumberValue();
  }

  if (num >= 0 && num < size) {
    fann_set_training_algorithm(net->FANN, fann_train_enum(num));
  }
}

NAN_GETTER(NNet::GetLearningRate)
{
  Nan::HandleScope scope;
  Local<Object> self = info.Holder();
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(self);

  float rate = fann_get_learning_rate(net->FANN);
  info.GetReturnValue().Set(Nan::New<Number>(rate));
}

NAN_GETTER(NNet::GetLearningMomentum)
{
  Nan::HandleScope scope;
  Local<Object> self = info.Holder();
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(self);

  float momentum = fann_get_learning_momentum(net->FANN);
  info.GetReturnValue().Set(Nan::New<Number>(momentum));
}

NAN_SETTER(NNet::SetLearningRate)
{
  Nan::HandleScope scope;
  Local<Object> self = info.Holder();
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(self);

  fann_set_learning_rate(net->FANN, value->NumberValue());
}

NAN_SETTER(NNet::SetLearningMomentum)
{
  Nan::HandleScope scope;
  Local<Object> self = info.Holder();
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(self);

  fann_set_learning_momentum(net->FANN, value->NumberValue());
}

NAN_METHOD(NNet::ActivationFunction)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  if (info.Length() < 2)
    return Nan::ThrowError("Usage: func = activation_function(layer, neuron) or activation_function(layer, neuron, newfunc)");

  int size = sizeof(FANN_ACTIVATIONFUNC_NAMES)/sizeof(char*);
  int layer = info[0]->IntegerValue();
  int neuron = info[1]->IntegerValue();

  if (info.Length() >= 3) {
    int num = -1;
    if (info[2]->IsString()) {
      num = _SeekCharArray(info[2].As<String>(), FANN_ACTIVATIONFUNC_NAMES, size, FANN_PREFIX);
    } else if (info[2]->IsNumber()) {
      num = info[2]->NumberValue();
    }

    if (num >= 0 && num < size) {
      fann_set_activation_function(net->FANN, fann_activationfunc_enum(num), layer, neuron);
    }
  }

  enum fann_activationfunc_enum func = fann_get_activation_function(net->FANN, layer, neuron);
  if (func >= 0 && func < size) {
    info.GetReturnValue().Set(NormalizeName(FANN_ACTIVATIONFUNC_NAMES[func], FANN_PREFIX, sizeof(FANN_PREFIX)-1));
  } else {
    return;
  }
}

NAN_METHOD(NNet::ActivationFunctionHidden)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  int size = sizeof(FANN_ACTIVATIONFUNC_NAMES)/sizeof(char*);

  if (info.Length() >= 1) {
    int num = -1;
    if (info[0]->IsString()) {
      num = _SeekCharArray(info[0].As<String>(), FANN_ACTIVATIONFUNC_NAMES, size, FANN_PREFIX);
    } else if (info[0]->IsNumber()) {
      num = info[0]->NumberValue();
    }

    if (num >= 0 && num < size) {
      fann_set_activation_function_hidden(net->FANN, fann_activationfunc_enum(num));
    }
  }

  enum fann_activationfunc_enum func = fann_get_activation_function(net->FANN, 1, 0);
  if (func >= 0 && func < size) {
    info.GetReturnValue().Set(NormalizeName(FANN_ACTIVATIONFUNC_NAMES[func], FANN_PREFIX, sizeof(FANN_PREFIX)-1));
  } else {
    return;
  }
}

NAN_METHOD(NNet::ActivationFunctionOutput)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  int size = sizeof(FANN_ACTIVATIONFUNC_NAMES)/sizeof(char*);

  if (info.Length() >= 1) {
    int num = -1;
    if (info[0]->IsString()) {
      num = _SeekCharArray(info[0].As<String>(), FANN_ACTIVATIONFUNC_NAMES, size, FANN_PREFIX);
    } else if (info[0]->IsNumber()) {
      num = info[0]->NumberValue();
    }

    if (num >= 0 && num < size) {
      fann_set_activation_function_output(net->FANN, fann_activationfunc_enum(num));
    }
  }

  enum fann_activationfunc_enum func = fann_get_activation_function(net->FANN, fann_get_num_layers(net->FANN)-1, 0);
  if (func >= 0 && func < size) {
    info.GetReturnValue().Set(NormalizeName(FANN_ACTIVATIONFUNC_NAMES[func], FANN_PREFIX, sizeof(FANN_PREFIX)-1));
  } else {
    return;
  }
}

NAN_METHOD(NNet::GetMse)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  float ret = fann_get_MSE(net->FANN);
  info.GetReturnValue().Set(Nan::New<Number>(ret));
}

NAN_METHOD(NNet::GetNumInput)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  unsigned int ret = fann_get_num_input(net->FANN);
  info.GetReturnValue().Set(Nan::New<Integer>(ret));
}

NAN_METHOD(NNet::GetNumOutput)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  unsigned int ret = fann_get_num_output(net->FANN);
  info.GetReturnValue().Set(Nan::New<Integer>(ret));
}

NAN_METHOD(NNet::GetTotalNeurons)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  unsigned int ret = fann_get_total_neurons(net->FANN);
  info.GetReturnValue().Set(Nan::New<Integer>(ret));
}

NAN_METHOD(NNet::GetTotalConnections)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  unsigned int ret = fann_get_total_connections(net->FANN);
  info.GetReturnValue().Set(Nan::New<Integer>(ret));
}

NAN_METHOD(NNet::GetConnectionRate)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  float ret = fann_get_connection_rate(net->FANN);
  info.GetReturnValue().Set(Nan::New<Number>(ret));
}

NAN_METHOD(NNet::GetNumLayers)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  unsigned int ret = fann_get_num_layers(net->FANN);
  info.GetReturnValue().Set(Nan::New<Integer>(ret));
}

NAN_GETTER(NNet::GetLayerArray)
{
  Nan::HandleScope scope;
  Local<Object> self = info.Holder();
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(self);
  info.GetReturnValue().Set(net->GetLayers());
}

NAN_METHOD(NNet::GetLayerArray)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  info.GetReturnValue().Set(net->GetLayers());
}

Local<Array> NNet::GetLayers()
{
  int size = fann_get_num_layers(FANN);
  unsigned int* layers = new unsigned int[size];
  fann_get_layer_array(FANN, layers);

  Local<Array> result_arr = Nan::New<Array>();
  for (int i=0; i<size; i++) {
    result_arr->Set(i, Nan::New<Number>(layers[i]));
  }

  delete[] layers;
  return result_arr;
}

NAN_METHOD(NNet::GetBiasArray)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  int size = fann_get_num_layers(net->FANN);
  unsigned int* layers = new unsigned int[size];
  fann_get_bias_array(net->FANN, layers);

  Local<Array> result_arr = Nan::New<Array>();
  for (int i=0; i<size; i++) {
    result_arr->Set(i, Nan::New<Number>(layers[i]));
  }

  delete[] layers;
  info.GetReturnValue().Set(result_arr);
}

NAN_METHOD(NNet::GetWeights)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  int size = fann_get_total_connections(net->FANN);
  struct fann_connection *conns = new struct fann_connection[size];
  fann_get_connection_array(net->FANN, conns);

  Local<Object> result_object = Nan::New<Object>();
  for (int i=0; i<size; i++) {
    Local<Object> obj;
    if (!result_object->Has(conns[i].from_neuron)) {
      obj = Nan::New<Object>();
      result_object->Set(conns[i].from_neuron, obj);
    } else {
      obj = result_object->Get(conns[i].from_neuron).As<Object>();
    }
    obj->Set(conns[i].to_neuron, Nan::New<Number>(conns[i].weight));
  }

  delete[] conns;
  info.GetReturnValue().Set(result_object);
}

NAN_METHOD(NNet::SetWeightsArr)
{
  Nan::HandleScope scope;
  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());

  if (!info[0]->IsObject())
    return Nan::ThrowError("First argument should be object");
  Local<Array> arg = info[0].As<Array>();
  Local<Array> keys = arg->GetOwnPropertyNames();

  struct fann_connection *conns = new struct fann_connection[fann_get_total_connections(net->FANN)];
  int counter = 0;
  for (unsigned i=0; i<keys->Length(); i++) {
    Local<Value> idx = keys->Get(i);
    if (!arg->Get(idx)->IsObject()) continue;
    Local<Object> obj = arg->Get(idx).As<Object>();
    Local<Array> keys2 = obj->GetOwnPropertyNames();
    for (unsigned j=0; j<keys2->Length(); j++) {
      conns[counter].from_neuron = idx->IntegerValue();
      conns[counter].to_neuron = keys2->Get(j)->IntegerValue();
      conns[counter].weight = obj->Get(keys2->Get(j))->NumberValue();
      counter++;
    }
  }
  fann_set_weight_array(net->FANN, conns, counter);

  delete[] conns;
  return;
}

NAN_METHOD(NNet::SetWeights)
{
  Nan::HandleScope scope;
  if (info[0]->IsObject())
    return SetWeightsArr(info);

  NNet *net = Nan::ObjectWrap::Unwrap<NNet>(info.This());
  if (info.Length() < 3)
    return Nan::ThrowError("Usage: set_weights(new_object) or set_weight(from_neuron, to_neuron, weight)");

  unsigned int from_neuron = info[0]->IntegerValue();
  unsigned int to_neuron = info[1]->IntegerValue();
  fann_type weight = info[2]->NumberValue();

  fann_set_weight(net->FANN, from_neuron, to_neuron, weight);
  return;
}

