#include "../include/DataHandler.hpp"

DataHandler::DataHandler() {
  dataArray = new std::vector<Data *>;
  trainingData = new std::vector<Data *>;
  testData = new std::vector<Data *>;
  validationData = new std::vector<Data *>;
}
DataHandler::~DataHandler() {
  // destructor later
}

void DataHandler::readCsv(std::string path, std::string delimiter) {
  numClasses = 0;
  std::ifstream dataFile(path.c_str());
  std::string line; // holds each line of csv
  while (std::getline(dataFile, line)) {
    if (line.length() == 0)
      continue;
    Data *d = new Data();
    d->setDoubleFeatureVector(new std::vector<double>());
    size_t position = 0;
    std::string token; // value in between delimiter
    while ((position = line.find(delimiter)) != std::string::npos) {
      token = line.substr(0, position);
      d->appendToDoubleFeatureVector(std::stod(token));
      line.erase(0, position + delimiter.length());
    }
    if (classMapCsv.find(line) != classMapCsv.end()) {
      d->setLabel(classMapCsv[line]);
    } else {
      classMapCsv[line] = numClasses;
      d->setLabel(classMapCsv[line]);
      numClasses++;
    }
    dataArray->push_back(d);
  }
  featureVectorSize = dataArray->at(0)->getDoubleFeatureVector()->size();
}

void DataHandler::readFeatureVector(std::string path) {
  uint32_t header[4]; // |Magic Number|Num. of images|Rowsize|Colsize|
  unsigned char bytes[4];
  FILE *f = fopen(path.c_str(), "r");
  if (f) {
    for (int i = 0; i < 4; i++) {
      if (fread(bytes, sizeof(bytes), 1, f)) {
        header[i] = convertToLittleEndian(bytes);
      }
    }
    printf("Input File Header has been retrieved.\n");
    int imageSize = header[2] * header[3];
    for (int i = 0; i < header[1]; i++) {
      Data *d = new Data();
      uint8_t elem[1];
      for (int j = 0; j < imageSize; j++) {
        if (fread(elem, sizeof(elem), 1, f)) {
          d->appendToFeatureVector(elem[0]);
        } else {
          printf("Error reading from the file\n");
          exit(1);
        }
      }
      dataArray->push_back(d);
    }
    printf("Successfully read and stored %lu feature vectors!\n",
           dataArray->size());
  } else {
    printf("Could not find the file.\n");
    exit(1);
  }
}
void DataHandler::readFeatureLabels(std::string path) {
  uint32_t header[2]; // |Magic Number|Num. of images|
  unsigned char bytes[4];
  FILE *f = fopen(path.c_str(), "r");
  if (f) {
    for (int i = 0; i < 2; i++) {
      if (fread(bytes, sizeof(bytes), 1, f)) {
        header[i] = convertToLittleEndian(bytes);
      }
    }
    printf("Label File Header has been retrieved.\n");
    for (int i = 0; i < header[1]; i++) {
      uint8_t elem[1];
      if (fread(elem, sizeof(elem), 1, f)) {
        dataArray->at(i)->setLabel(elem[0]);
        // printf("%d\t",elem[0]);
      } else {
        printf("Error reading from the file.\n");
        exit(1);
      }
    }
    printf("Successfully read and stored %lu labels!\n", dataArray->size());
  } else {
    printf("Could not find the file.\n");
    exit(1);
  }
}

void DataHandler::splitData() {
  std::unordered_set<int> usedIndexes;
  int trainSize = dataArray->size() * TRAIN_SET_PERCENT;
  int testSize = dataArray->size() * TEST_SET_PERCENT;
  int validSize = dataArray->size() * VALIDATION_SET_PERCENT;

  // Training Data
  int count = 0;
  while (count < trainSize) {
    int randomIndex =
        rand() % dataArray->size(); // number between 0 and dataSize-1
    if (usedIndexes.find(randomIndex) == usedIndexes.end()) {
      trainingData->push_back(dataArray->at(randomIndex));
      usedIndexes.insert(randomIndex);
      count++;
    }
  }

  // Test Data
  count = 0;
  while (count < testSize) {
    int randomIndex =
        rand() % dataArray->size(); // number between 0 and dataSize-1
    if (usedIndexes.find(randomIndex) == usedIndexes.end()) {
      testData->push_back(dataArray->at(randomIndex));
      usedIndexes.insert(randomIndex);
      count++;
    }
  }

  // Validation Data
  count = 0;
  while (count < validSize) {
    int randomIndex =
        rand() % dataArray->size(); // number between 0 and dataSize-1
    if (usedIndexes.find(randomIndex) == usedIndexes.end()) {
      validationData->push_back(dataArray->at(randomIndex));
      usedIndexes.insert(randomIndex);
      count++;
    }
  }

  printf("Training Data Size: %lu.\n", trainingData->size());
  printf("Test Data Size: %lu.\n", testData->size());
  printf("Validation Data Size: %lu.\n", validationData->size());
}

void DataHandler::countClasses() {
  int count = 0;
  for (unsigned i = 0; i < dataArray->size(); i++) {
    if (classMap.find(dataArray->at(i)->getLabel()) == classMap.end()) {
      classMap[dataArray->at(i)->getLabel()] = count;
      count++;
    }
  }
  numClasses = count;
  printf("Successfully extracted %d Unique Classes.\n", numClasses);
}

uint32_t DataHandler::convertToLittleEndian(const unsigned char *bytes) {
  return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) |
                    (bytes[3]));
}

int DataHandler::getClassCounts() { return numClasses; }

std::vector<Data *> *DataHandler::getTrainingData() { return trainingData; }

std::vector<Data *> *DataHandler::getTestData() { return testData; }
std::vector<Data *> *DataHandler::getValidationData() { return validationData; }
