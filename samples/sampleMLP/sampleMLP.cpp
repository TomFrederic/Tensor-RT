#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <iterator>
#include <vector>

// constants that are known about the MNIST MLP network.
static const int32_t INPUT_H{28};                                                    // The height of the mnist input image.
static const int32_t INPUT_W{28};                                                    // The weight of the mnist input image.
static const int32_t HIDDEN_COUNT{2};                                                // The number of hidden layers for MNIST sample.
static const int32_t HIDDEN_SIZE{256};                                               // The size of the hidden state for MNIST sample.
static const int32_t FINAL_SIZE{10};                                                 // The size of the output state for MNIST sample.
static const int32_t MAX_BATCH_SIZE{1};                                              // The maximum default batch size for MNIST sample.
static const int32_t OUTPUT_SIZE{10};                                                // The output of the topK layer for MNIST sample.
static const int32_t ITER_COUNT{1};                                                  // The number of iterations to run the MNIST sample.
static const nvinfer1::ActivationType MNIST_ACT{nvinfer1::ActivationType::kSIGMOID}; // The MNIST sample uses a sigmoid for activation.
static const char* INPUT_BLOB_NAME{"input"};                                         // The default input blob name.
static const char* OUTPUT_BLOB_NAME{"output"};                                       // the default output blob name.
static const char* DEFAULT_WEIGHT_FILE{"sampleMLP.wts2"};                            // The weight file produced from README.txt

// The Args struct holds the arguments to generate the MLP structure, defaults to MNIST sample from README.txt.
struct Args
{
    int32_t inputSize{INPUT_H * INPUT_W};        // The input data size to use.
    int32_t hiddenCount{HIDDEN_COUNT};           // The number of hidden layers in this MLP.
    int32_t hiddenSize{HIDDEN_SIZE};             // The number of cells in the hidden layers.
    int32_t finalUnits{FINAL_SIZE};              // The number of cells in the final layer.
    int32_t maxBatchSize{MAX_BATCH_SIZE};        // The largest batch size for the given network.
    int32_t outputSize{OUTPUT_SIZE};             // The size of the topK results.
    int32_t iterCount{ITER_COUNT};               // The number of iterations to run the network.
    std::string inputName{INPUT_BLOB_NAME};      // The name of the input blob.
    std::string outputName{OUTPUT_BLOB_NAME};    // The name of the output blob.
    std::string weightFile{DEFAULT_WEIGHT_FILE}; // The name of the weight file to load
    nvinfer1::ActivationType actType{MNIST_ACT}; // The current activation type.
    bool enableINT8{false};                      // Enable ability to run in INT8 mode.
    bool enableFP16{false};                      // Enable ability to run in FP16 mode.
    bool enablePerf{false};                      // Enabling perf analysis disables verification.
    bool enableVerbose{false};                   // Enable verbose perf analysis.
    bool enableDebug{false};                     // Enable debug emission.
};                                               // struct args

static Logger gLogger;

class TimerBase
{
public:
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void accumulate() = 0;
    virtual float getElapsedTime() const final { return mMS; }
    virtual bool isCPU() const { return false; }
    virtual void addTimer(TimerBase* rhs) final { mMS += rhs->getElapsedTime(); }
protected:
    TimerBase() {}
    float mMS{0.0f};
}; // class TimerBase

class GpuTimer : public TimerBase
{
public:
    GpuTimer(cudaStream_t stream)
        : mStream(stream)
    {
        CHECK(cudaEventCreate(&mStart));
        CHECK(cudaEventCreate(&mStop));
    }
    virtual ~GpuTimer()
    {
        CHECK(cudaEventDestroy(mStart));
        CHECK(cudaEventDestroy(mStop));
    }
    void start() override final { CHECK(cudaEventRecord(mStart, mStream)); }
    void stop() override final { CHECK(cudaEventRecord(mStop, mStream)); }
    void accumulate() override final
    {
        float ms{0.0f};
        CHECK(cudaEventSynchronize(mStop));
        CHECK(cudaEventElapsedTime(&ms, mStart, mStop));
        mMS += ms;
    }

private:
    cudaEvent_t mStart, mStop;
    cudaStream_t mStream;
}; // class GpuTimer

class CpuTimer : public TimerBase
{
public:
    CpuTimer()
        : TimerBase()
    {
    }
    virtual ~CpuTimer() {}
    void start() override final { mBegin = std::clock(); }
    void stop() override final { mEnd = std::clock(); }
    void accumulate() override final { mMS += (mEnd - mBegin) / (float) CLOCKS_PER_SEC * 1000; }
    bool isCPU() const override final { return true; }
private:
    std::clock_t mBegin, mEnd;
}; // class CpuTimer

/**
 * \class ShapedWeights
 * \brief A combination of Dims and Weights to provide shape to a weight struct.
 */
struct ShapedWeights
{
    nvinfer1::Dims shape;
    nvinfer1::Weights data;
};
typedef std::map<std::string, ShapedWeights> WeightMap_t;

// The split function takes string and based on a set of tokens produces a vector of tokens
// tokenized by the tokens. This is used to parse the shape field of the wts format.
static void split(std::vector<std::string>& split, std::string tokens, const std::string& input)
{
    split.clear();
    std::size_t begin = 0, size = input.size();
    while (begin != std::string::npos)
    {
        std::size_t found = input.find_first_of(tokens, begin);
        // Handle case of two or more delimiters in a row.
        if (found != begin) split.push_back(input.substr(begin, found - begin));
        begin = found + 1;
        // Handle case of no more tokens.
        if (found == std::string::npos) break;
        // Handle case of delimiter being last or first token.
        if (begin >= size) break;
    }
}

// Read a blob based on the type passed in
template <typename T>
void* loadShapeData(std::ifstream& input, size_t numElements)
{
    void* tmp = malloc(sizeof(T) * numElements);
    input.read(reinterpret_cast<char*>(tmp), numElements * sizeof(T));
    assert(input.peek() == '\n');
    // Consume the newline at the end of the data blob.
    input.get();
    return tmp;
}

nvinfer1::Dims loadShape(std::ifstream& input)
{
    // Initial format is "(A, B, C,...,Y [,])"
    nvinfer1::Dims shape{};
    std::string shapeStr;

    // Convert to "(A,B,C,...,Y[,])"
    do
    {
        std::string tmp;
        input >> tmp;
        shapeStr += tmp;
    } while (*shapeStr.rbegin() != ')');
    assert(input.peek() == ' ');

    // Consume the space between the shape and the data buffer.
    input.get();

    // Convert to "A,B,C,...,Y[,]"
    assert(*shapeStr.begin() == '(');
    shapeStr.erase(0, 1); //
    assert(*shapeStr.rbegin() == ')');
    shapeStr.pop_back();

    // Convert to "A,B,C,...,Y"
    if (*shapeStr.rbegin() == ',')
        shapeStr.pop_back(); // Remove the excess ',' character

    std::vector<std::string> shapeDim;
    split(shapeDim, ",", shapeStr);
    // Convert to {A, B, C,...,Y}
    assert(shapeDim.size() <= shape.MAX_DIMS);
    assert(shapeDim.size() > 0);
    assert(shape.nbDims == 0);
    std::for_each(shapeDim.begin(),
                  shapeDim.end(),
                  [&](std::string& val) {
                      shape.d[shape.nbDims++] = std::stoi(val);
                  });
    return shape;
}

// Our weight files are in a very simple space delimited format.
// type is the integer value of the DataType enum in NvInfer.h.
// <number of buffers>
// for each buffer: [name] [type] [size] <data x size in hex>
WeightMap_t loadWeights(const std::string file)
{
    WeightMap_t weightMap;
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--)
    {
        ShapedWeights wt{};
        std::int32_t type;
        std::string name;
        input >> name >> std::dec >> type;
        wt.shape = loadShape(input);
        wt.data.type = static_cast<nvinfer1::DataType>(type);
        wt.data.count = std::accumulate(wt.shape.d, wt.shape.d + wt.shape.nbDims, 1, std::multiplies<int32_t>());
        if (wt.data.type == nvinfer1::DataType::kFLOAT)
        {
            wt.data.values = loadShapeData<uint32_t>(input, wt.data.count);
        }
        else
        {
            assert(wt.data.type == nvinfer1::DataType::kHALF);
            wt.data.values = loadShapeData<uint16_t>(input, wt.data.count);
        }
        weightMap[name] = wt;
    }
    return weightMap;
}

// We have the data files located in a specific directory. This
// searches for that directory format from the current directory.
std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/mnist/", "data/mnist/",
                                  "data/samples/mlp/", "data/mlp/"};
    return locateFile(input, dirs);
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(locateFile(filename), buffer, INPUT_H, INPUT_W);
}

// The addMLPLayer function is a simple helper function that creates the combination required for an
// MLP layer. By replacing the implementation of this sequence with various implementations, then
// then it can be shown how TensorRT optimizations those layer sequences.
nvinfer1::ILayer* addMLPLayer(nvinfer1::INetworkDefinition* network,
                              nvinfer1::ITensor& inputTensor,
                              int32_t hiddenSize,
                              nvinfer1::Weights wts,
                              nvinfer1::Weights bias,
                              nvinfer1::ActivationType actType,
                              int idx)
{
    std::string baseName("MLP Layer" + (idx == -1 ? "Output" : std::to_string(idx)));
    auto fc = network->addFullyConnected(inputTensor, hiddenSize, wts, bias);
    assert(fc != nullptr);
    std::string fcName = baseName + "FullyConnected";
    fc->setName(fcName.c_str());
    auto act = network->addActivation(*fc->getOutput(0), actType);
    assert(act != nullptr);
    std::string actName = baseName + "Activation";
    act->setName(actName.c_str());
    return act;
}

ShapedWeights generateRandomWeights(nvinfer1::DataType type, int64_t count)
{
    nvinfer1::DimsHW dim{1, static_cast<int>(count)};
    ShapedWeights wts{dim, nvinfer1::Weights{type, nullptr, count}};
    if (type == nvinfer1::DataType::kFLOAT)
        wts.data.values = new float[count];
    else
        wts.data.values = new uint16_t[count];
    return wts;
}

template <typename TYPE>
void transposeWeights(nvinfer1::Weights& wts, int hiddenSize)
{
        int d = 0;
        int dim0 = hiddenSize;                // 256 or 10
        int dim1 = wts.count / dim0;          // 784 or 256
        TYPE* trans_wts = new TYPE[wts.count];
        for (int d0=0; d0<dim0; ++d0) {
            for (int d1=0; d1<dim1; ++d1) {
                trans_wts[d]  = *((TYPE*)wts.values + d1*dim0 + d0);
                d++;
            }
        }

        for (int k=0; k<wts.count; ++k){
            *((TYPE*)wts.values + k) = trans_wts[k];
        }
}

static void setAllLayerOutputsToHalf(nvinfer1::INetworkDefinition* network)
{
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        nvinfer1::ILayer* layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            if (layer->getOutput(j)->isNetworkOutput())
                layer->getOutput(j)->setType(nvinfer1::DataType::kHALF);
        }
    }
}

// Create the Engine using only the API and not any parser.
nvinfer1::ICudaEngine* fromAPIToModel(Args& args, nvinfer1::IBuilder* builder)
{
    nvinfer1::DataType dt{args.enableFP16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT};
    WeightMap_t weightMap;
    if (args.enablePerf)
    {
        for (int i = 0; i < args.hiddenCount; ++i)
        {
            std::stringstream weightStr, biasStr;
            weightStr << "hiddenWeights" << i;
            biasStr << "hiddenBias" << i;
            int wtsSize{!i ? args.inputSize * args.hiddenSize : args.hiddenSize * args.hiddenSize};
            int biasSize{args.hiddenSize};
            weightMap[weightStr.str()] = generateRandomWeights(dt, wtsSize);
            weightMap[biasStr.str()] = generateRandomWeights(dt, biasSize);
        }
        weightMap["outputWeights"] = generateRandomWeights(dt, args.hiddenSize * args.finalUnits);
        weightMap["outputBias"] = generateRandomWeights(dt, args.finalUnits);
    }
    else
        weightMap = loadWeights(locateFile(args.weightFile));
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    // FC layers must still have 3 dimensions, so we create a {C, 1, 1,} matrix.
    // Currently the mnist example is only trained in FP32 mode.
    auto input = network->addInput(INPUT_BLOB_NAME, args.enablePerf ? dt : nvinfer1::DataType::kFLOAT, nvinfer1::DimsCHW{args.inputSize, 1, 1});
    assert(input != nullptr);

    for (int i = 0; i < args.hiddenCount; ++i)
    {
        std::stringstream weightStr, biasStr;
        weightStr << "hiddenWeights" << i;
        biasStr << "hiddenBias" << i;
        // Transpose hidden layer weights 
        if (dt == nvinfer1::DataType::kFLOAT)
            transposeWeights<uint32_t>(weightMap[weightStr.str()].data, args.hiddenSize);
        else
            transposeWeights<uint16_t>(weightMap[weightStr.str()].data, args.hiddenSize);
        auto mlpLayer = addMLPLayer(network, *input, args.hiddenSize, weightMap[weightStr.str()].data, weightMap[biasStr.str()].data, args.actType, i);
        input = mlpLayer->getOutput(0);
    }
    // Tranpose output layer weights 
    if (dt == nvinfer1::DataType::kFLOAT)
        transposeWeights<uint32_t>(weightMap["outputWeights"].data, args.finalUnits);
    else
        transposeWeights<uint16_t>(weightMap["outputWeights"].data, args.finalUnits);
    
    auto finalLayer = addMLPLayer(network, *input, args.finalUnits, weightMap["outputWeights"].data, weightMap["outputBias"].data, args.actType, -1);
    assert(finalLayer != nullptr);
    // Run softmax to get our probabilities.
    auto softMax = network->addSoftMax(*finalLayer->getOutput(0));
    assert(softMax != nullptr);
    softMax->setName("OutputSoftMax");
    softMax->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*softMax->getOutput(0));

    // Build the engine
    builder->setMaxBatchSize(args.maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setHalf2Mode(args.enableFP16);
    if (args.enableFP16) setAllLayerOutputsToHalf(network);

    auto engine = builder->buildCudaEngine(*network);
    // we don't need the network any more
    network->destroy();

    // Once we have built the cuda engine, we can release all of our held memory.
    for (auto& mem : weightMap)
    {
        free(const_cast<void*>(mem.second.data.values));
    }
    return engine;
}

void APIToModel(Args& args, // batch size - NB must be at least as large as the batch we want to run with)
                nvinfer1::IHostMemory** modelStream)
{
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

    // create the model to populate the network, then set the outputs and create an engine
    nvinfer1::ICudaEngine* engine = fromAPIToModel(args, builder);

    assert(engine != nullptr);

    // GIE-3533
    // serialize the engine, then close everything down
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

template <typename T>
void doInference(nvinfer1::IExecutionContext& context, uint8_t* inputPtr, uint8_t* outputPtr, Args& args)
{
    T* input = reinterpret_cast<T*>(inputPtr);
    T* output = reinterpret_cast<T*>(outputPtr);
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], args.maxBatchSize * args.inputSize * sizeof(T)));
    CHECK(cudaMalloc(&buffers[outputIndex], args.maxBatchSize * args.outputSize * sizeof(T)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    GpuTimer gpu(stream), enqueue(stream), iter(stream), inAsync(stream), outAsync(stream);

    CHECK(cudaMemcpy(buffers[inputIndex], input, args.maxBatchSize * args.inputSize * sizeof(T), cudaMemcpyHostToDevice));
    if (args.enablePerf)
    {
        cudaProfilerStart();
        gpu.start();
    }
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    int iterCount{args.enablePerf ? args.iterCount : 1};
    for (int begin = 0; begin < iterCount; ++begin)
    {
        if (args.enablePerf && args.enableVerbose)
        {
            iter.start();
            inAsync.start();
        }
        if (args.enablePerf && args.enableVerbose)
        {
            inAsync.stop();
            enqueue.start();
        }
        context.enqueue(args.maxBatchSize, buffers, stream, nullptr);
        if (args.enablePerf && args.enableVerbose)
        {
            enqueue.stop();
            outAsync.start();
        }
        if (args.enablePerf && args.enableVerbose)
        {
            outAsync.stop();
            iter.stop();
            iter.accumulate();
            inAsync.accumulate();
            outAsync.accumulate();
            enqueue.accumulate();
        }
    }
    if (args.enablePerf)
    {
        cudaProfilerStop();
        gpu.stop();
        gpu.accumulate();
    }
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], args.maxBatchSize * args.outputSize * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    if (args.enablePerf)
    {
        std::cout
            << args.maxBatchSize << ", "
            << args.inputSize << ", "
            << args.outputSize << ", "
            << args.hiddenCount << ", "
            << args.hiddenSize << ", "
            << args.finalUnits << ", "
            << gpu.getElapsedTime() << "s";
        if (args.enableVerbose)
        {
            std::cout << ", " << (enqueue.getElapsedTime() / iterCount);
            std::cout << ", " << (inAsync.getElapsedTime() / iterCount);
            std::cout << ", " << (outAsync.getElapsedTime() / iterCount);
            std::cout << ", " << (iter.getElapsedTime() / iterCount);
        }
        std::cout << std::endl;
    }
}
void printHelp(char* appName)
{
    std::cout << "Usage:\n"
                 "\t "
              << appName << "[-h]\n"
                            "\t-h      Display help information. All single dash optoins enable perf mode.\n"
                            "\t-i <#>  Specify the size of the input, defaults to 784(28x28).\n"
                            "\t-e <#>  Specify the number of iterations to execute the network.\n"
                            "\t-b <#>  Specify the batch size to use, defaults to 256.\n"
                            "\t-o <#>  Specify the size of the topK output, defaults to 10.Must be in range [1, finalLayerCell].\n"
                            "\t-n <#>  Specify the number of hidden layers, defaults to 2.\n"
                            "\t-u <#>  Specify the number of hidden layer cells, defaults to 256.\n"
                            "\t-f <#>  Specify the number of final layer cells, defaults to 10.\n"
                            "\t-a <#>  The activation to use in on the layers, defaults to 1. Valid values are 0[ReLU], 1[Sigmoid], and 2[TanH].\n"
                            "\t--perf  Enable performance mode.\n"
                            "\t--fp16  Enable FP16 support.\n"
                            "\t--int8  Enable Int8 support.\n"
                            "\t--debug Enable debug mode.\n"
                            "\t--verbose Enable verbose perf mode.\n"
                            "\n";
}
void emitPerfWarning()
{
    std::cout << "Default arguments changed, perf mode enabled.\n";
}
void reportFailure(char* name, const char* error)
{
    std::cerr << "Invalid argument: " << error << "." << std::endl;
    printHelp(name);
    exit(EXIT_FAILURE);
}
// Parse the arguments and return failure if arguments are incorrect
// or help menu is requested.
void parseArgs(Args& args, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string argStr(argv[i]);
        if (argStr == "-h")
        {
            printHelp(argv[0]);
            exit(EXIT_SUCCESS);
        }
        if (argStr == "-i")
        {
            if (++i >= argc) reportFailure(argv[0], "no integer specified for argument -i");
            args.inputSize = std::atoi(argv[i]);
            if (args.maxBatchSize <= 0)
                reportFailure(argv[0], "input size must be at least 1");
            if (!args.enablePerf)
            {
                emitPerfWarning();
                args.enablePerf = true;
            }
        }
        else if (argStr == "-e")
        {
            if (++i >= argc) reportFailure(argv[0], "no integer specified for argument -e");
            args.iterCount = std::atoi(argv[i]);
            if (args.iterCount <= 0)
                reportFailure(argv[0], "iteration count must be at least 1");
        }
        else if (argStr == "-b")
        {
            if (++i >= argc) reportFailure(argv[0], "no integer specified for argument -b");
            args.maxBatchSize = std::atoi(argv[i]);
            if (args.maxBatchSize <= 0)
                reportFailure(argv[0], "batch size must be at least 1");
            if (!args.enablePerf)
            {
                emitPerfWarning();
                args.enablePerf = true;
            }
        }
        else if (argStr == "-o")
        {
            if (++i >= argc) reportFailure(argv[0], "no integer specified for argument -o");
            args.outputSize = std::atoi(argv[i]);
            if (args.outputSize <= 0)
                reportFailure(argv[0], "number of outputs must be at least 1");
            if (!args.enablePerf)
            {
                emitPerfWarning();
                args.enablePerf = true;
            }
        }
        else if (argStr == "-n")
        {
            if (++i >= argc) reportFailure(argv[0], "no integer specified for argument -n");
            args.hiddenCount = std::atoi(argv[i]);
            if (args.hiddenCount < 0)
                reportFailure(argv[0], "number of hidden layers cannot be negative");
            if (!args.enablePerf)
            {
                emitPerfWarning();
                args.enablePerf = true;
            }
        }
        else if (argStr == "-u")
        {
            if (++i >= argc) reportFailure(argv[0], "no integer specified for argument -u");
            args.hiddenSize = std::atoi(argv[i]);
            if (args.hiddenSize <= 0)
                reportFailure(argv[0], "hidden size must be at least 1");
            if (!args.enablePerf)
            {
                emitPerfWarning();
                args.enablePerf = true;
            }
        }
        else if (argStr == "-f")
        {
            if (++i >= argc) reportFailure(argv[0], "no integer specified for argument -f");
            args.finalUnits = std::atoi(argv[i]);
            if (args.finalUnits <= 0)
                reportFailure(argv[0], "output size must be at least 1");
            if (!args.enablePerf)
            {
                emitPerfWarning();
                args.enablePerf = true;
            }
        }
        else if (argStr == "-a")
        {
            if (++i >= argc) reportFailure(argv[0], "no integer specified for argument -a");
            int val = std::atoi(argv[i]);
            if (val < -1)
                reportFailure(argv[0], "activation value must be positive");
            if (val >= nvinfer1::EnumMax<nvinfer1::ActivationType>())
                reportFailure(argv[0], "activation value must be one of specified values");
            args.actType = static_cast<nvinfer1::ActivationType>(val - 1);
            if (!args.enablePerf)
            {
                emitPerfWarning();
                args.enablePerf = true;
            }
        }
        else if (argStr == "--perf")
        {
            args.enablePerf = true;
        }
        else if (argStr == "--fp16")
        {
            args.enableFP16 = true;
        }
        else if (argStr == "--int8")
        {
            args.enableINT8 = true;
        }
        else if (argStr == "--debug")
        {
            args.enableDebug = true;
        }
        else if (argStr == "--verbose")
        {
            args.enableVerbose = true;
        }
        else
        {
            std::cerr << "Invalid argument: " << argStr << std::endl;
            printHelp(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (args.outputSize > args.finalUnits)
        reportFailure(argv[0], "The output size must be in the range [1, finalCellCount].");
    if (args.enableINT8 && args.enableFP16)
        reportFailure(argv[0], "Cannot enable both Int8 and FP16 modes.");
}
void printHeader(Args& args)
{
    if (args.enablePerf)
    {
        std::cout << "BatchSize, InputSize, OutputSize, HiddenLayers, HiddenCells, FinalCells, TotalTime";
        if (args.enableVerbose)
            std::cout << "NetworkTime, H2D, D2H, PerIteration";
        std::cout << std::endl;
    }
}
int main(int argc, char* argv[])
{
    Args args;
    parseArgs(args, argc, argv);

    // create a model using the API directly and serialize it to a stream.
    nvinfer1::IHostMemory* modelStream{nullptr};

    // Temporarily disable serialization path while debugging the layer.
    APIToModel(args, &modelStream);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
    if (modelStream) modelStream->destroy();

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    printHeader(args);

    srand(unsigned(time(nullptr)));
    bool pass{true};
    int num = rand() % 10;
    // Just for simplicity, allocations for memory use float,
    // even for fp16 data type.
    uint8_t* input = new uint8_t[args.maxBatchSize * args.inputSize * sizeof(float)];
    uint8_t* output = new uint8_t[args.maxBatchSize * args.outputSize * sizeof(float)];
    assert(input != nullptr);
    assert(output != nullptr);

    if (!args.enablePerf)
    {
        // read a random digit file from the data directory for use as input.
        auto fileData = new uint8_t[args.inputSize];
        readPGMFile(std::to_string(num) + ".pgm", fileData);

        // print the ascii representation of the file that was loaded.
        std::cout << "\n\n\n---------------------------"
                  << "\n\n\n"
                  << std::endl;
        for (int i = 0; i < args.inputSize; i++)
            std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

        // Normalize the data the same way TensorFlow does.
        for (int i = 0; i < args.inputSize; i++)
            reinterpret_cast<float*>(input)[i] = 1.0 - float(fileData[i])/255.0f;
        delete [] fileData;
    }

    // Run the inference
    if (args.enableFP16)
        doInference<uint16_t>(*context, input, output, args);
    else
        doInference<float>(*context, input, output, args);
    if (!args.enablePerf)
    {
        float* prob{reinterpret_cast<float*>(output)};
        std::cout << "\n\n";
        auto val = std::max_element(prob, prob + 10);
        auto idx = std::distance(prob, val);
        std::cout << "Algorithm chose " << idx << " with probability " << *val << std::endl;
        pass = (idx == num && *val > 0.2f);

    }
    delete[] input;
    delete[] output;
    // destroy the engine
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
