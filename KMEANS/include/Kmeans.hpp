#ifndef __KMEANS_HPP
#define __KMEANS_HPP

#include "../../include/Common.hpp"
#include "../../include/DataHandler.hpp"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_set>

typedef struct Cluster {

  std::vector<double> *centroid;
  std::vector<Data *> *clusterPoints;
  std::map<int, int> classCounts;
  int mostFrequentClass;

  Cluster(Data *initialPoint) {
    centroid = new std::vector<double>;
    clusterPoints = new std::vector<Data *>;
    for (auto value : *(initialPoint->getFeatureVector())) {
      centroid->push_back(value);
    }
    clusterPoints->push_back(initialPoint);
    classCounts[initialPoint->getLabel()] = 1;
    mostFrequentClass = initialPoint->getLabel();
  }

  void addToCluster(Data *point) {
    int previousSize = clusterPoints->size();
    clusterPoints->push_back(point);
    for (int i = 0; i < centroid->size() - 1; i++) {
      double value = centroid->at(i);
      value *= previousSize;
      value += point->getFeatureVector()->at(i);
      value /= (double)clusterPoints->size();
      centroid->at(i) = value;
    }
    if (classCounts.find(point->getLabel()) == classCounts.end()) {
      classCounts[point->getLabel()] = 1;
    } else {
      classCounts[point->getLabel()]++;
    }
    setMostFrequentClass();
  }

  void setMostFrequentClass() {
    int bestClass, freq = 0;
    for (auto kv : classCounts) {
      if (kv.second > freq) {
        freq = kv.second;
        bestClass = kv.first;
      }
    }
    mostFrequentClass = bestClass;
  }

} cluster_t;

class Kmeans : public CommonData {
  int numClusters;
  std::vector<cluster_t *> *clusters;
  std::unordered_set<int> *usedIndexes;

public:
  Kmeans(int k);
  void initClusters();
  void initClustersForEachClass();
  void train();
  double euclideanDistance(std::vector<double> *, Data *);
  double validate();
  double test();
};

#endif
