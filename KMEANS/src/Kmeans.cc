#include "../include/Kmeans.hpp"

Kmeans::Kmeans(int k) {
  numClusters = k;
  clusters = new std::vector<cluster_t *>;
  usedIndexes = new std::unordered_set<int>;
}

void Kmeans::initClusters() {
  for (int i = 0; i < numClusters; i++) {
    int index = (rand() % trainingData->size());
    while (usedIndexes->find(index) != usedIndexes->end()) {
      index = (rand() % trainingData->size());
    }
    clusters->push_back(new Cluster(trainingData->at(index)));
    usedIndexes->insert(index);
  }
}

void Kmeans::initClustersForEachClass() {
  std::unordered_set<int> classesUsed;
  for (int i = 0; i < trainingData->size(); i++) {
    if (classesUsed.find(trainingData->at(i)->getLabel()) ==
        classesUsed.end()) {
      clusters->push_back(new cluster_t(trainingData->at(i)));
      classesUsed.insert(trainingData->at(i)->getLabel());
      usedIndexes->insert(i);
    }
  }
}

void Kmeans::train() {
  int index = 0;
  while (usedIndexes->size() < trainingData->size()) {
    while (usedIndexes->find(index) != usedIndexes->end()) {
      index++;
    }
    double minDist = std::numeric_limits<double>::max();
    int bestCluster = 0;
    for (int j = 0; j < clusters->size(); j++) {
      double currentDist =
          euclideanDistance(clusters->at(j)->centroid, trainingData->at(index));
      if (currentDist < minDist) {
        minDist = currentDist;
        bestCluster = j;
      }
    }
    clusters->at(bestCluster)->addToCluster(trainingData->at(index));
    usedIndexes->insert(index);
  }
}

double Kmeans::euclideanDistance(std::vector<double> *centroid, Data *point) {
  double dist = 0.0;
  for (int i = 0; i < centroid->size(); i++) {
    dist += pow(centroid->at(i) - point->getFeatureVector()->at(i), 2);
  }
  return sqrt(dist);
}

double Kmeans::validate() {
  double numCorrect = 0.0;
  for (auto queryPoint : *validationData) {
    double minDist = std::numeric_limits<double>::max();
    int bestCluster = 0;
    for (int j = 0; j < clusters->size(); j++) {
      double currentDist =
          euclideanDistance(clusters->at(j)->centroid, queryPoint);
      if (currentDist < minDist) {
        minDist = currentDist;
        bestCluster = j;
      }
    }
    if (clusters->at(bestCluster)->mostFrequentClass ==
        queryPoint->getLabel()) {
      numCorrect++;
    }
  }
  return 100.0 * (numCorrect) / (double)validationData->size();
}

double Kmeans::test() {
  double numCorrect = 0.0;
  for (auto queryPoint : *testData) {
    double minDist = std::numeric_limits<double>::max();
    int bestCluster = 0;
    for (int j = 0; j < clusters->size(); j++) {
      double currentDist =
          euclideanDistance(clusters->at(j)->centroid, queryPoint);
      if (currentDist < minDist) {
        minDist = currentDist;
        bestCluster = j;
      }
    }
    if (clusters->at(bestCluster)->mostFrequentClass ==
        queryPoint->getLabel()) {
      numCorrect++;
    }
  }
  return 100.0 * (numCorrect) / (double)testData->size();
}

int main() {
  DataHandler *dh = new DataHandler();
  dh->readFeatureVector("../dataset/train-images-idx3-ubyte");
  dh->readFeatureLabels("../dataset/train-labels-idx1-ubyte");
  dh->splitData();
  dh->countClasses();
  double performance = 0.0, best_performance = 0.0;
  int best_k = 1;
  for (int k = dh->getClassCounts(); k < dh->getTrainingData()->size() * 0.1;
       k++) {
    Kmeans *km = new Kmeans(k);
    km->setTrainingData(dh->getTrainingData());
    km->setValidationData(dh->getValidationData());
    km->setTestData(dh->getTestData());
    km->initClusters();
    km->train();
    performance = km->validate();
    printf("Current Performance @ k = %d: %.2f\n", k, performance);
    if (performance > best_performance) {
      best_performance = performance;
      best_k = k;
    }
  }

  Kmeans *km = new Kmeans(best_k);
  km->setTrainingData(dh->getTrainingData());
  km->setValidationData(dh->getValidationData());
  km->setTestData(dh->getTestData());
  km->initClusters();
  performance = km->test();
  printf("Tested Performance @ k = %d: %.2f\n", best_k, performance);
}
