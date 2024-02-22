#ifndef __DATA_H
#define __DATA_H

#include "stdint.h"
#include "stdio.h"
#include <vector>

class Data {
  std::vector<uint8_t> *featureVector;
  std::vector<double> *doubleFeatureVector;
  std::vector<int> *classVector;

  uint8_t label;
  int enumLabel;
  double distance;

public:
  Data();
  ~Data();
  void setFeatureVector(std::vector<uint8_t> *);
  void setDoubleFeatureVector(std::vector<double> *);
  void appendToFeatureVector(uint8_t);
  void appendToDoubleFeatureVector(double);
  void setClassVector(int count);
  void setLabel(uint8_t);
  void setEnumeratedLabel(int);
  void setDistance(double val);
  void printVector();
  void printNormalizedVector();

  int getFeatureVectorSize();
  uint8_t getLabel();
  uint8_t getEnumeratedLabel();

  std::vector<uint8_t> *getFeatureVector();
  std::vector<double> *getDoubleFeatureVector();
  std::vector<int> *getClassVector();
  double getDistance();
};

#endif
