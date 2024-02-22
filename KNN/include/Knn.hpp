#ifndef __KNN_H
#define __KNN_H

#include "../../include/Common.hpp"

class Knn : public CommonData {
  int k;
  std::vector<Data *> *neighbors;

public:
  Knn(int);
  Knn();
  ~Knn();
  void findKnearest(Data *queryPoint);
  void setK(int val);
  int predict();
  double calculateDistance(Data *queryPosition, Data *input);
  double validatePerformance();
  double testPerformance();
};

#endif
