#include "../include/Data.hpp"

Data::Data() { featureVector = new std::vector<uint8_t>; }

Data::~Data() {}

void Data::setFeatureVector(std::vector<uint8_t> *vect) {
  featureVector = vect;
}

void Data::setDoubleFeatureVector(std::vector<double> *vect) {
  doubleFeatureVector = vect;
}

void Data::appendToFeatureVector(uint8_t val) { featureVector->push_back(val); }

void Data::appendToDoubleFeatureVector(double val) {
  doubleFeatureVector->push_back(val);
}

void Data::setLabel(uint8_t val) { label = val; }

void Data::setEnumeratedLabel(int val) { enumLabel = val; }

void Data::setDistance(double val) { distance = val; }

void Data::setClassVector(int count) {
  classVector = new std::vector<int>();
  for (int i = 0; i < count; i++) {
    if (i == label) {
      classVector->push_back(1);
    } else {
      classVector->push_back(0);
    }
  }
}

int Data::getFeatureVectorSize() { return featureVector->size(); }

uint8_t Data::getLabel() { return label; }
uint8_t Data::getEnumeratedLabel() { return enumLabel; }

std::vector<uint8_t> *Data::getFeatureVector() { return featureVector; }

std::vector<double> *Data::getDoubleFeatureVector() {
  return doubleFeatureVector;
}

double Data::getDistance() { return distance; }
