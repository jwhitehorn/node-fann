/*
 *   All basic functions and primary connections to node.js
 */

#include <string.h>
#include "node-fann.h"

void NNet::PrototypeInit(Local<FunctionTemplate> t)
{
  t->InstanceTemplate()->SetInternalFieldCount(1);
  t->SetClassName(Nan::New<String>("FANN").ToLocalChecked());
  Nan::SetPrototypeMethod(t, "train", Train);
  Nan::SetPrototypeMethod(t, "cascadetrain", CascadeTrain);
  Nan::SetPrototypeMethod(t, "train_once", TrainOnce);
  Nan::SetPrototypeMethod(t, "run", Run);
  Nan::SetPrototypeMethod(t, "save", SaveToFile);

  // deprecated in favor of require('fann').get_...
  /*Nan::SetPrototypeMethod(t, "get_all_training_algorithms", GetTrainingAlgorithmList);
  Nan::SetPrototypeMethod(t, "get_all_activation_functions", GetActivationFunctionList);
  Nan::SetPrototypeMethod(t, "get_all_network_types", GetNetworkTypeList);
  Nan::SetPrototypeMethod(t, "get_all_stop_functions", GetStopFuncList);
  Nan::SetPrototypeMethod(t, "get_all_error_functions", GetErrorFuncList);*/

  Nan::SetPrototypeMethod(t, "activation_function", ActivationFunction);
  Nan::SetPrototypeMethod(t, "activation_function_hidden", ActivationFunctionHidden);
  Nan::SetPrototypeMethod(t, "activation_function_output", ActivationFunctionOutput);
  Nan::SetPrototypeMethod(t, "get_num_input", GetNumInput);
  Nan::SetPrototypeMethod(t, "get_MSE", GetMse);
  Nan::SetPrototypeMethod(t, "get_num_output", GetNumOutput);
  Nan::SetPrototypeMethod(t, "get_total_neurons", GetTotalNeurons);
  Nan::SetPrototypeMethod(t, "get_total_connections", GetTotalConnections);
  Nan::SetPrototypeMethod(t, "get_network_type", GetNetworkType);
  Nan::SetPrototypeMethod(t, "get_connection_rate", GetConnectionRate);

  // use net->layers instead ?
  Nan::SetPrototypeMethod(t, "get_num_layers", GetNumLayers);
  Nan::SetPrototypeMethod(t, "get_layer_array", GetLayerArray);

  Nan::SetPrototypeMethod(t, "get_bias_array", GetBiasArray);
  Nan::SetPrototypeMethod(t, "get_weight_array", GetWeights);
  Nan::SetPrototypeMethod(t, "set_weight_array", SetWeights);
  Nan::SetPrototypeMethod(t, "get_weight", GetWeights);
  Nan::SetPrototypeMethod(t, "set_weight", SetWeights);

  Nan::SetAccessor(t->InstanceTemplate(), Nan::New("training_algorithm").ToLocalChecked(), GetTrainingAlgorithm, SetTrainingAlgorithm);
  Nan::SetAccessor(t->InstanceTemplate(), Nan::New("learning_rate").ToLocalChecked(), GetLearningRate, SetLearningRate);
  Nan::SetAccessor(t->InstanceTemplate(), Nan::New("learning_momentum").ToLocalChecked(), GetLearningMomentum, SetLearningMomentum);
  Nan::SetAccessor(t->InstanceTemplate(), Nan::New("layers").ToLocalChecked(), GetLayerArray, 0);

  /*
  t->InstanceTemplate()->SetAccessor(Nan::New<String>("training_algorithm").ToLocalChecked(), GetTrainingAlgorithm, SetTrainingAlgorithm);
  t->InstanceTemplate()->SetAccessor(Nan::New<String>("learning_rate").ToLocalChecked(), GetLearningRate, SetLearningRate);
  t->InstanceTemplate()->SetAccessor(Nan::New<String>("learning_momentum").ToLocalChecked(), GetLearningMomentum, SetLearningMomentum);
  t->InstanceTemplate()->SetAccessor(Nan::New<String>("layers").ToLocalChecked(), GetLayerArray);
  */
}

void NNet::Initialize(Handle<Object> t)
{
  Nan::HandleScope scope;
  Local<FunctionTemplate> t1 = Nan::New<FunctionTemplate>(NewStandard);
  Local<FunctionTemplate> t2 = Nan::New<FunctionTemplate>(NewSparse);
  Local<FunctionTemplate> t3 = Nan::New<FunctionTemplate>(NewShortcut);
  Local<FunctionTemplate> t4 = Nan::New<FunctionTemplate>(NewFromFile);
//  Local<FunctionTemplate> t5 = Nan::New<FunctionTemplate>(CloneNet);
  PrototypeInit(t1);
  PrototypeInit(t2);
  PrototypeInit(t3);
  PrototypeInit(t4);
  t->Set(Nan::New<String>("standard").ToLocalChecked(), t1->GetFunction());
  t->Set(Nan::New<String>("sparse").ToLocalChecked(), t2->GetFunction());
  t->Set(Nan::New<String>("shortcut").ToLocalChecked(), t3->GetFunction());
  t->Set(Nan::New<String>("load").ToLocalChecked(), t4->GetFunction());
//  t->Set(String::NewSymbol("clone"), t4->GetFunction());

  Nan::SetMethod(t, "get_all_training_algorithms", GetTrainingAlgorithmList);
  Nan::SetMethod(t, "get_all_activation_functions", GetActivationFunctionList);
  Nan::SetMethod(t, "get_all_network_types", GetNetworkTypeList);
  Nan::SetMethod(t, "get_all_stop_functions", GetStopFuncList);
  Nan::SetMethod(t, "get_all_error_functions", GetErrorFuncList);
}

extern "C" void init (Handle<Object> target)
{
  Nan::HandleScope scope;
  NNet::Initialize(target);
}

#ifdef NODE_MODULE
NODE_MODULE(fann, init)
#endif

