#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include "Data.hpp"
#include "stdint.h"
#include <fstream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

class DataHandler {
  std::vector<Data *> *dataArray;
  std::vector<Data *> *trainingData;
  std::vector<Data *> *testData;
  std::vector<Data *> *validationData;

  int numClasses;
  int featureVectorSize;
  std::map<uint8_t, int> classMap;
  std::map<std::string, int> classMapCsv;

  const double TRAIN_SET_PERCENT = 0.75;
  const double TEST_SET_PERCENT = 0.20;
  const double VALIDATION_SET_PERCENT = 0.05;

public:
  DataHandler();
  ~DataHandler();

  void readCsv(std::string path, std::string delimiter);
  void readFeatureVector(std::string path);
  void readFeatureLabels(std::string path);
  void splitData();
  void countClasses();

  uint32_t convertToLittleEndian(const unsigned char *bytes);
  int getClassCounts();

  std::vector<Data *> *getTrainingData();
  std::vector<Data *> *getTestData();
  std::vector<Data *> *getValidationData();
};

#endif
