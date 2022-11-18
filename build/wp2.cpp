#include "NetworkArchitectures.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SpatiallySparseDatasetOpenCV.h"

////////////////////////////

using namespace std;

////////////////////////////

// -------------------------------------------------------------------------

// Network definitions

// -------------------------------------------------------------------------

// We will allow for three types of convolutional network architecture:

// 1) A simple but effective, not-very-deep-by modern-standards network inspired

// by "Multi-column Deep Neural Networks for Image Classification", Dan Ciresan,

// Ueli Meier and Jurgen Schmidhuber.

// C3-MP2-C2-MP2-...-MP2-C2-AveragePooling

// If there are nLayers of MaxPooling, the network is optimised to detect

// features with size 2^nLayers

// 2) A deeper network with VGG-style C3C3 paired convolutional layers: "Very

// Deep Convolutional Networks for Large-Scale Visual Recognition", Karen

// Simonyan and Andrew Zisserman

// C3-C3-MP2-C3-C3-MP2-...-MP2-C3-C3-AveragePooling

// If there are nLayers of MaxPooling, the network is optimised to detect

// features with size 2^nLayers

// 3) A deeper network with fractional max-pooling. "Fractional Max-Pooling",

// Ben Graham

// C2-FMP-...-C2-FMP-C2-AveragePooling

// If there are nLayers of MaxPooling, the network is optimised to detect

// features with size 2^(nLayers/2), i.e. nLayers needs to be larger



class PlanktonSparseConvNet : public SparseConvNet {

public:

  PlanktonSparseConvNet(

      int nInputFeatures, int nClasses,

      int networkType, // 1, 2 or 3, as above

      int nLayers,     // number of layers of max-pooling (i.e. 5 or 6 for

                       // network types 1 and 2, 10 or 12 for type 2)

      int dropoutMultiplier = 0.375, // number between 0 and 0.5. Dropout can reduce

                                 // over-fitting when training data is limited.

                                 // An increasing amount of dropout is used

                                 // rising up through the network from 0 to

                                 // dropoutMultiplier.

      int cudaDevice =

          -1 // PCI Bus ID for the CUDA device to use, -1 for the default device

      )

      : SparseConvNet(2, nInputFeatures, nClasses, cudaDevice) {



    switch (networkType) {



    case 1: // Ciresan, et al, simple net

      for (int i = 0; i < nLayers; i++) {

        // convolution + max pooling

        addLeNetLayerMP(

            32 * (i + 1),     // number of filters in the i-th layer

            (i == 0) ? 3 : 2, // filter size, 3x3 convolution in

                              // layer 0, 2x2 convolution in higher

                              // layers

            1,                // filter stride (i.e. shift)

            2,                // max-pooling size

            2,                // max-pooling stride

            VLEAKYRELU,       // activation function

            dropoutMultiplier * i / (nLayers + 1) // dropout probability

            );

      }

      // convolution only (no pooling)

      addLeNetLayerMP(32 * (nLayers + 1), 2, 1, 1, 1, VLEAKYRELU,

                      dropoutMultiplier * nLayers / (nLayers + 1));

      break;



    case 2: // VGG-style net

      for (int i = 0; i < nLayers; i++) {

        // convolution over 3x3 px, no pooling

        addLeNetLayerMP(32 * (i + 1), 3, 1, 1, 1, VLEAKYRELU,

                        dropoutMultiplier * i / (nLayers + 1));

        // convolution over 3x3 and pooling over 2x2

        addLeNetLayerMP(32 * (i + 1), 3, 1, 2, 2, VLEAKYRELU,

                        dropoutMultiplier * i / (nLayers + 1));

      }

      // convolutions only

      addLeNetLayerMP(32 * (nLayers + 1), 3, 1, 1, 1, VLEAKYRELU,

                      dropoutMultiplier * nLayers / (nLayers + 1));

      addLeNetLayerMP(32 * (nLayers + 1), 3, 1, 1, 1, VLEAKYRELU,

                      dropoutMultiplier * nLayers / (nLayers + 1));

      break;



    case 3: // Fractional max-pooling net

      // fractional max-pooling ratio

      const float fmpShrink = powf(2, 0.5);

      for (int i = 0; i < nLayers; i++) {

        // convolution + fractional max pooling

        addLeNetLayerPOFMP(

            32 * (i + 1), // number of filters in the i-th layer

            2,            // filter size, 2x2

            1,            // filter stride (i.e. shift)

            2,            // fractional max-pooling window size 2x2

            fmpShrink,    // fractional max-pooling stride

            VLEAKYRELU,   // activation function

            dropoutMultiplier * i / (nLayers + 1) // dropout probability

            );

      }

      // convolution only (no pooling)

      addLeNetLayerMP(32 * (nLayers + 1), 2, 1, 1, 1, VLEAKYRELU,

                      dropoutMultiplier * nLayers / (nLayers + 1));

      break;

    }



    // final layer has spatial size 32x32, do average-pooling over active

    // sites

    addTerminalPoolingLayer(32);

    // fully connected layer

    addLeNetLayerMP(32 * (nLayers + 2), 1, 1, 1, 1, VLEAKYRELU,

                    dropoutMultiplier);

    // softmax for classification

    addSoftmaxLayer();

  }

};



// -------------------------------------------------------------------------

// Data augmentation

// -------------------------------------------------------------------------

// Method to distort and scale input images

// used for data augmentation and to limit GPU memory usage



float areaThreshold; // global variable, set below, threshold for reducing image

                     // size

int openCVflag;      // 0 for grayscale, >0 for color



Picture *OpenCVPicture::distort(RNG &rng, batchType type) {

  // load image

  OpenCVPicture *pic = new OpenCVPicture(*this);

  pic->loadDataWithoutScaling(openCVflag);



  int area = pic->area(); // number of active pixels

  float r;                // aspect ratio adjustment

  float s;                // scale

  float alpha;            // rotation

  float beta;             // shear



  // data augmentation

  if (type == TRAINBATCH) {

    // For training - rotate, shear and scale

    r = rng.uniform(-0.1, 0.1);

    s = 1 + rng.uniform(-0.1, 0.1);

    alpha = rng.uniform(0, 2 * 3.1415926535);

    beta = rng.uniform(-0.2, 0.2) + alpha;

  } else {

    // For testing

    // rotate only

    r = 0;

    s = 1;

    alpha = rng.uniform(0, 2 * 3.1415926535);

    beta = alpha;

  }



  // downscale large images while retaining most size information

  if (area > areaThreshold) {

    s *= powf(area / areaThreshold, -0.8);

  }



  // compute affine transformation matrix

  float c00 = (1 + r) * s * cos(alpha);

  float c01 = (1 + r) * s * sin(alpha);

  float c10 = -(1 - r) * s * sin(beta);

  float c11 = (1 - r) * s * cos(beta);



  // horizontal flip

  if (rng.randint(2) == 0) {

    c00 *= -1;

    c01 *= -1;

  }



  // distort image

  pic->affineTransform(c00, c01, c10, c11);



  // translate image within input field

  pic->jiggle(rng, 100);



  return pic;

}



//-------------------------------------------------------------------------

// main() function. Could be turned into a command line program or library

//-------------------------------------------------------------------------



int main(int argc, char *argv[]) {

  //-------------------------------------------------------------------------
  // Options.
  //-------------------------------------------------------------------------

  // Network specification
  int networkType = 3;
  int nLayers = 12;
  float dropoutMultiplier = 0.375;

  // address of CUDA card (PCI bus id)
  // use nvidia-smi to get it; -1 for default GPU
  int cudaDevice = -1;

  // Start and stop training epochs
  // Starting at epoch > 0 loads the weights from that epoch and continues
  // training until stop epoch
  // If startEpoch == stopEpoch then just classify the unlabeledDataset

  int startEpoch = 400;
  int stopEpoch = 400;
  int exemplarsPerClassPerEpoch = 1000;
  float initialLearningRate = 0.003;
  float learningRateDecay = 0.01;

  // number of images that are fed to the card at a time larger numbers will
  // (marginally) increase training speed; increase according to GPU memory
  // If batchSize is small, make momentum closer to 1. Momentum increases the
  // effective batch size to batchSize * 1/(1 - momentum); that should cover
  // the diversity of the training set
  int batchSize = 350;
  float momentum = 0.999;



  // Threshold, in terms of active pixels, at which to start scaling down images

  // in the OpenCVPicture::distort function

  areaThreshold = 4000; // global variable



  // Path to data directory.

  std::string trainDataDir =

      "Data/plankton/train";

  // There should be a file trainDataDir + "/classList" containing a list of

  // the classes, one per line, and a directory trainDataDir + className for

  // each

  // class.



  // Unlabeled data to classify. Empty if just training.

  std::string unlabeledDataDir =

      "Data/plankton/test";



  std::string wildcard = "*.*";

  // Look for files in trainDataDir + "/" + classname + "/" + wildcard

  // and unlabeledDataDir + "/" + classname + "/" + wildcard



  // Number of classes (-1 to calculate from classList file)

  int nClasses = -1;



  // OpenCV flag (1 for monochrome, 3 for RGB color images)

  int nFeatures = 1;

  openCVflag = (nFeatures == 1) ? 0 : 1; // global variable



  // Load training data into memory? May be faster. Needs sufficient RAM.

  bool loadImagesIntoMemory = true;



  // experiment name (path where weights are stored)

  std::string baseName = "weights/plankton";



  // Validation set percentage

  float validationSetPercentage = 0; // i.e. extract 20% for a validation set

  // Reserves a portion of the training set for monitoring training.

  // For best results, first run with a 20% validation set, and then when

  // you are happy with the other settings, re-train with

  // validationSetPercentage=0 to make full use of the trainingSet

  for (int i = 1; i < (argc-1); ++i) {

    std::string arg_current = argv[i];
    std::string arg_next = argv[i+1];

    if (arg_current == "-startEpoch" || arg_current == "-start" ) { 
      startEpoch = std::stoi(arg_next); 
    }
    else if (arg_current == "-stopEpoch" || arg_current == "-stop" ) { 
      stopEpoch = std::stoi(arg_next); 
    }
    else if (arg_current == "-batchSize" || arg_current == "-bs" ) { 
      batchSize = std::stoi(arg_next); 
    }
    else if (arg_current == "-trainDataDir" || arg_current == "-train" ) { 
      trainDataDir = arg_next; 
    }
    else if (arg_current == "-unlabeledDataDir" || arg_current == "-unl" ) { 
      unlabeledDataDir = arg_next; 
    }
    else if (arg_current == "-nClasses" || arg_current == "-nc" ) { 
      nClasses = std::stoi(arg_next); 
    }
    else if (arg_current == "-exemplarsPerClassPerEpoch" || arg_current == "-epcpe" ) { 
      exemplarsPerClassPerEpoch = std::stoi(arg_next); 
    }
    else if (arg_current == "-initialLearningRate" || arg_current == "-ilr" ) { 
      initialLearningRate = std::stof(arg_next); 
    }
    else if (arg_current == "-learningRateDecay" || arg_current == "-lrd" ) { 
      learningRateDecay = std::stof(arg_next); 
    }
    else if (arg_current == "-validationSetPercentage" || arg_current == "-vsp" ) { 
      validationSetPercentage = std::stof(arg_next); 
    }
    else if (arg_current == "-cudaDevice" || arg_current == "-cD" ) { 
      cudaDevice = std::stoi(arg_next); 
    }
}

  //-------------------------------------------------------------------------

  // Options summary

  //-------------------------------------------------------------------------



  switch (networkType) {

  case 1:

    std::cout << "Network type 1 - Ciresan-Schmidhuber-Meier style network" << std::endl;
    break;

  case 2:

    std::cout << "Network type 2 - VGG style network" << std::endl;

    break;

  case 3:

    std::cout << "Network type 3 - fractional max-pooling network" << std::endl;

    break;

  }

  std::cout << "Layers of pooling:       " << nLayers << std::endl;

  std::cout << "Dropout multiplier:      " << dropoutMultiplier << std::endl;

  std::cout << "Start epoch:             " << startEpoch << std::endl;

  std::cout << "Stop epoch:              " << stopEpoch << std::endl;

  std::cout << "Exemplars/class/epoch:   " << exemplarsPerClassPerEpoch

            << std::endl;

  std::cout << "Initial learning rate:   " << initialLearningRate << std::endl;

  std::cout << "Learning rate decay:     " << learningRateDecay << std::endl;

  std::cout << "Batch size:              " << batchSize << std::endl;

  std::cout << "Momentum:                " << momentum << std::endl;

  std::cout << "Area threshold:          " << areaThreshold << std::endl;

  std::cout << "Training data directory: " << trainDataDir << std::endl;

  std::cout << "Unlabeled data:          " << unlabeledDataDir << std::endl;

  std::cout << "Wildcard:                " << wildcard << std::endl;

  std::cout << "Cache training images:   " << (loadImagesIntoMemory ? "true" : "false") << std::endl;

  std::cout << "Experiment name:         " << baseName << std::endl;

  std::cout << "Validation set size:     " << validationSetPercentage * 100

            << "%" << std::endl;

  std::cout << std::endl;

  //-------------------------------------------------------------------------

  // Training and using the network

  //-------------------------------------------------------------------------

  SpatiallySparseDataset trainSet, validationSet;



  if (startEpoch < stopEpoch or nClasses == -1) {

    // Load training data and/or count classes

    std::cout << "Loading training set...\n";

    trainSet = OpenCVLabeledDataSet(

        "Data/plankton/classList",   // path to list of classes

        trainDataDir,		     // path to data

        wildcard,                    // wildcard for images

        TRAINBATCH,                  // type of dataset

        255, // background grey level (tolerance +/- 2 set in

             // OpenCVPicture.cpp)

        loadImagesIntoMemory,

        openCVflag // flag to OpenCV imread() function call

        );

    trainSet.summary();

    std::cout << std::endl;

    nClasses = trainSet.nClasses;

    if (validationSetPercentage > 0) {

      std::cout << "Extracting validation set ...\n";

      validationSet = trainSet.extractValidationSet(validationSetPercentage);

      trainSet.summary();

      validationSet.summary();

      std::cout << std::endl;

    }

  }



  // create network

  std::cout << "Create network:" << std::endl;

  PlanktonSparseConvNet cnn(nFeatures, nClasses, networkType, nLayers, dropoutMultiplier, cudaDevice);
  

  std::cout << std::endl;



  // load weights if continuing a previous run

  if (startEpoch > 0)

    cnn.loadWeights(baseName, startEpoch);



  // training loop

  for (int epoch = startEpoch; epoch < stopEpoch; epoch++) {

    std::cout << "epoch: " << epoch << std::endl;



    // Decrease learning rate with epoch

    // Start with large steps when the network is new.

    // If the initial learning rate is too big, the network will explode!

    // Decrease afterwards to fine tune the network

    // stopEpoch should be roughly 4 / learningRateDecay

    float learningRate = initialLearningRate * exp(-learningRateDecay * epoch);



    // train for one epoch: nClasses * exemplarsPerClassPerEpoch images

    // auto trainSample = trainSet.balancedSample(exemplarsPerClassPerEpoch);


    cnn.processDataset(

        trainSet, // subset of images to train on

        batchSize,   // number of images to feed to the GPU at a time

        learningRate, momentum);
      

    cnn.processDataset(trainSet, batchSize, learningRate, momentum);
      
    cnn.processDataset(trainSet, batchSize, learningRate, momentum);

      
    cnn.saveWeights(baseName, epoch + 1);



    // Perform validation on a balenced sample of the validation set

    if (validationSetPercentage > 0) {

      auto validationSample = validationSet.balancedSample(10);

      // Use multiple testing

      cnn.processDatasetRepeatTest(

          validationSample, // dataset to predict

          batchSize,        // number of images to feed to the GPU at a time

          3,                // number of repetitions of the prediction

          baseName + "/validation_predictions.csv", // file name for predictions

          baseName +

              "/validation_confusion.csv" // file name for confusion matrix

          );

    }

  }



  // load test set

  if (not unlabeledDataDir.empty()) {

    std::cout << "Loading testing set...\n";

    OpenCVUnlabeledDataSet testSet(

        "Data/plankton/classList", unlabeledDataDir, wildcard, 255,

        false,     // no point preloading images during testing

        openCVflag // flag to OpenCV imread() function call

        );

    testSet.summary();

    // predict test set with repeat testing

    std::string flattened_unlabeled = unlabeledDataDir.substr(unlabeledDataDir.find("results_images") + 30);
    for (int i =0; i < flattened_unlabeled.length(); i++){ 
      if (flattened_unlabeled[i] == '/')
        flattened_unlabeled[i] = '-';
    }

    std::string concat_var = "./Data/plankton/" + flattened_unlabeled + "_plankton_predictions.csv"; 

    cnn.processDatasetRepeatTest(testSet, batchSize / 2, 24, concat_var);

  }

}



// clang-format off

//-------------------------------------------------------------------------

// Interpreting the output of SparseConvNet

//-------------------------------------------------------------------------

/*



Starts with a description of network settings:



    Network type 1 - Ciresan-Schmidhuber-Meier style network

    Layers of pooling:       5

    Dropout multiplier:      0

    Start epoch:             0

    Stop epoch:              400

    Exemplars/class/epoch:   1000

    Initial learning rate:   0.003

    Learning rate decay:     0.1

    Batch size:              32

    Momentum:                0.999

    Area threshold:          4000

    Training data directory: /home/ben/Archive/Datasets/kagglePlankton/train/

    Unlabeled data:          /home/ben/Archive/Datasets/kagglePlankton/test

    Wildcard:                *.*

    Cache training images:   true

    Experiment name:         weights/plankton

    Validation set size:     20%



Then describe the training data (and the effect of removing a validation set)



    Loading training set...

    Name:           /home/jiho/cnn/test_2015-12-17/data

    nPictures:      204191

    nClasses:       46

    nFeatures:      1

    Type:           TRAINBATCH

    [...]



Then info about the available (and chosen: *) GPUs:



    *3 Tesla K20c 4799MB Compute capability: 3.5

     131 Quadro K620 2047MB Compute capability: 5.0



Then a description of the network



    Sparse CNN - dimension=2 nInputFeatures=1 nClasses=121

    0:Convolution 3^2x1->9

    1:Learn 9->32 dropout=0 VeryLeakyReLU

    2:MaxPooling 2 2

    3:Convolution 2^2x32->128

    4:Learn 128->64 dropout=0 VeryLeakyReLU

    5:MaxPooling 2 2

    6:Convolution 2^2x64->256

    7:Learn 256->96 dropout=0 VeryLeakyReLU

    8:MaxPooling 2 2

    9:Convolution 2^2x96->384

    10:Learn 384->128 dropout=0 VeryLeakyReLU

    11:MaxPooling 2 2

    12:Convolution 2^2x128->512

    13:Learn 512->160 dropout=0 VeryLeakyReLU

    14:MaxPooling 2 2

    15:Convolution 2^2x160->640

    16:Learn 640->192 dropout=0 VeryLeakyReLU

    17:TerminalPooling 32 1024

    18:Learn 192->224 dropout=0 VeryLeakyReLU

    19:Learn 224->0 dropout=0 Softmax Classification

    Spatially sparse CNN with layer sizes: 1-(TP)-32-(C2)-33-(MP2)-66-(C2)-67-...

    ...(MP2)-134-(C2)-135-(MP2)-270-(C2)-271-(MP2)-542-(C2)-543-(MP2)-1086-(C3)-1088

    Input-field dimensions = 1088x1088





TODO: add a description of this



And finally the training loop



    epoch: 1

    /home/jiho/cnn/test_2015-12-17/data minus Validation set subset

Mistakes:97.8237% NLL:3.82872 MegaMultiplyAdds/sample:12 time:2s

GigaMultiplyAdds/s:24 rate:1879/s

    /home/jiho/cnn/test_2015-12-17/data minus Validation set subset

Mistakes:97.7584% NLL:3.82861 MegaMultiplyAdds/sample:12 time:2s

GigaMultiplyAdds/s:26 rate:2068/s

    [...]



where a line contains



  Mistakes: percentage of wrong classification

  NLL: negative log likelihood (smaller = better ;-) )

  MegaMultiplyAdds/sample: number of (hundreds of) unit operations per pixel

  time: seconds elapsed for the current step

  GigaMultiplyAdds/s: number of (thousands of) ? Q? what exactly

  rate: number of images processed by second



*/

// clang-format on

