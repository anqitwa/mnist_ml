#include "../include/Knn.hpp"
#include "../../include/DataHandler.hpp"
#include "stdint.h"
#include <cmath>
#include <limits>
#include <map>

Knn::Knn(int val) { k = val; }

Knn::Knn() {
  // nothing to do
}

Knn::~Knn() {
  // nothing to do
}

void Knn::findKnearest(Data *queryPoint) {
  neighbors = new std::vector<Data *>;
  double min = std::numeric_limits<double>::max();
  double previousMin = min;
  int index = 0;
  for (int i = 0; i < k; i++) {
    if (i == 0) {
      for (int j = 0; j < trainingData->size(); j++) {
        double distance = calculateDistance(queryPoint, trainingData->at(j));
        trainingData->at(j)->setDistance(distance);
        if (distance < min) {
          min = distance;
          index = j;
        }
      }
      neighbors->push_back(trainingData->at(index));
      previousMin = min;
      min = std::numeric_limits<double>::max();
    } else {
      for (int j = 0; j < trainingData->size(); j++) {
        double distance = trainingData->at(j)->getDistance();
        if (distance > previousMin && distance < min) {
          min = distance;
          index = j;
        }
      }
      neighbors->push_back(trainingData->at(index));
      previousMin = min;
      min = std::numeric_limits<double>::max();
    }
  }
}

void Knn::setK(int val) { k = val; }

int Knn::predict() {
  std::map<uint8_t, int> classFreq;
  for (int i = 0; i < neighbors->size(); i++) {
    if (classFreq.find(neighbors->at(i)->getLabel()) == classFreq.end()) {
      classFreq[neighbors->at(i)->getLabel()] = 1;
    } else {
      classFreq[neighbors->at(i)->getLabel()]++;
    }
  }

  int best = 0;
  int max = 0;
  for (auto kv : classFreq) {
    if (kv.second > max) {
      max = kv.second;
      best = kv.first;
    }
  }
  neighbors->clear();
  return best;
}

double Knn::calculateDistance(Data *queryPoint, Data *input) {
  double distance = 0.0;
  if (queryPoint->getFeatureVectorSize() != input->getFeatureVectorSize()) {
    printf("Error: Vector Size Mismatch.\n");
    exit(1);
  }
#ifdef EUCLID
  for (unsigned i = 0; i < queryPoint->getFeatureVectorSize(); i++) {
    distance += pow(queryPoint->getFeatureVector()->at(i) -
                        input->getFeatureVector()->at(i),
                    2);
  }
  distance = sqrt(distance);
#elif defined MANHATTAN
// put manhattan implementation here later
#endif
  return distance;
}

double Knn::validatePerformance() {
  double currentPerformance = 0.0;
  int count = 0;
  int dataIndex = 0;
  for (Data *queryPoint : *validationData) {
    findKnearest(queryPoint);
    int prediction = predict();
    if (prediction == queryPoint->getLabel()) {
      count++;
    }
    dataIndex++;
    printf("Current Performance = %.3f %%\n",
           ((double)count * 100.0) / ((double)dataIndex));
  }
  currentPerformance =
      ((double)count * 100.0) / ((double)validationData->size());
  printf("Validation Performance for K = %d: %.3f %%\n", k, currentPerformance);
  return currentPerformance;
}

double Knn::testPerformance() {
  double currentPerformance = 0.0;
  int count = 0;
  for (Data *queryPoint : *testData) {
    findKnearest(queryPoint);
    int prediction = predict();
    if (prediction == queryPoint->getLabel()) {
      count++;
    }
  }
  currentPerformance = ((double)count * 100.0) / ((double)testData->size());
  printf("Tested Performance = %.3f %%\n", currentPerformance);
  return currentPerformance;
}

int main() {
  DataHandler *dh = new DataHandler();
  dh->readFeatureVector("../dataset/train-images-idx3-ubyte");
  dh->readFeatureLabels("../dataset/train-labels-idx1-ubyte");
  dh->splitData();
  dh->countClasses();
  Knn *knearest = new Knn();
  knearest->setTrainingData(dh->getTrainingData());
  knearest->setTestData(dh->getTestData());
  knearest->setValidationData(dh->getValidationData());

  double performance = 0.0;
  double bestPerformance = 0.0;
  int bestK = 1;
  for (int i = 0; i <= 4; i++) {
    if (i == 1) {
      knearest->setK(i);
      performance = knearest->validatePerformance();
      bestPerformance = performance;
    } else {
      knearest->setK(i);
      performance = knearest->validatePerformance();
      if (performance > bestPerformance) {
        bestPerformance = performance;
        bestK = i;
      }
    }
  }
  knearest->setK(bestK);
  knearest->testPerformance();
}
