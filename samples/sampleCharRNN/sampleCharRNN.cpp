#include "NvInfer.h"
#include "NvUtils.h"
#include "cuda_runtime_api.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include "common.h"

static Logger gLogger;
// To train the model that this sample uses the dataset can be found here:
// http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
//
// The ptb_w model was created retrieved from:
// https://github.com/okuchaiev/models.git
//
// The tensorflow command used to train:
// python models/tutorials/rnn/ptb/ptb_word_lm.py --data_path=data --file_prefix=ptb.char --model=charlarge --save_path=charlarge/ --seed_for_sample='consumer rep'
//
// Epochs trained: 30 
// Test perplexity: 2.697
//
// Training outputs a params.p file, which contains all of the weights in pickle format.
// This data was converted via a python script that did the following.
// Cell0 and Cell1 Linear weights matrices were concatenated as rnnweight
// Cell0 and Cell1 Linear bias vectors were concatenated as rnnbias
// Embedded is added as embed.
// fc_w is added as rnnfcw
// fc_b is added as rnnfcb
//
// The floating point values are converted to 32bit integer hexadecimal and written out to char-rnn.wts.

// These mappings came from training with tensorflow 0.12.1
// and emitting the word to id and id to word mappings from
// the checkpoint data after loading it.
// The only difference is that in the data set that was used,
static std::map<char, int> char_to_id{{'#', 40},
    { '$', 31}, { '\'', 28}, { '&', 35}, { '*', 49},
    { '-', 32}, { '/', 48}, { '.', 27}, { '1', 37},
    { '0', 36}, { '3', 39}, { '2', 41}, { '5', 43},
    { '4', 47}, { '7', 45}, { '6', 46}, { '9', 38},
    { '8', 42}, { '<', 22}, { '>', 23}, { '\0', 24},
    { 'N', 26}, { '\\', 44}, { ' ', 0}, { 'a', 3},
    { 'c', 13}, { 'b', 20}, { 'e', 1}, { 'd', 12},
    { 'g', 18}, { 'f', 15}, { 'i', 6}, { 'h', 9},
    { 'k', 17}, { 'j', 30}, { 'm', 14}, { 'l', 10},
    { 'o', 5}, { 'n', 4}, { 'q', 33}, { 'p', 16},
    { 's', 7}, { 'r', 8}, { 'u', 11}, { 't', 2},
    { 'w', 21}, { 'v', 25}, { 'y', 19}, { 'x', 29},
    { 'z', 34}
};

// A mapping from index to character.
static std::vector<char> id_to_char{{' ', 'e', 't', 'a',
    'n', 'o', 'i', 's', 'r', 'h', 'l', 'u', 'd', 'c',
    'm', 'f', 'p', 'k', 'g', 'y', 'b', 'w', '<', '>',
    '\0', 'v', 'N', '.', '\'', 'x', 'j', '$', '-', 'q',
    'z', '&', '0', '1', '9', '3', '#', '2', '8', '5',
    '\\', '7', '6', '4', '/', '*'}};

// Information describing the network
static const int LAYER_COUNT = 2;
static const int BATCH_SIZE = 1;
static const int HIDDEN_SIZE = 512;
static const int SEQ_SIZE = 1;
static const int DATA_SIZE = HIDDEN_SIZE;
static const int VOCAB_SIZE = 50;
static const int OUTPUT_SIZE = 1;
static const int NUM_GATES = 8;

// We have 6 outputs for LSTM, this needs to be changed to 4 for any other RNN type
static const int NUM_BINDINGS = 6;
const char* INPUT_BLOB_NAME = "data";
const char* HIDDEN_IN_BLOB_NAME = "hiddenIn";
const char* CELL_IN_BLOB_NAME = "cellIn";
const char* HIDDEN_OUT_BLOB_NAME = "hiddenOut";
const char* CELL_OUT_BLOB_NAME = "cellOut";
const char* OUTPUT_BLOB_NAME = "pred";
static const int INPUT_IDX = 0;
static const int HIDDEN_IN_IDX = 1;
static const int CELL_IN_IDX = 2;
static const int HIDDEN_OUT_IDX = 3;
static const int CELL_OUT_IDX = 4;
static const int OUTPUT_IDX = 5;

const char *gNames[NUM_BINDINGS] = 
{
    INPUT_BLOB_NAME,
    HIDDEN_IN_BLOB_NAME,
    CELL_IN_BLOB_NAME,
    HIDDEN_OUT_BLOB_NAME,
    CELL_OUT_BLOB_NAME,
    OUTPUT_BLOB_NAME
};
const int gSizes[NUM_BINDINGS] = 
{
    BATCH_SIZE * SEQ_SIZE * DATA_SIZE,
    LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
    LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
    LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
    LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
    BATCH_SIZE * SEQ_SIZE * OUTPUT_SIZE
};

using namespace nvinfer1;

// Our weight files are in a very simple space delimited format.
// type is the integer value of the DataType enum in NvInfer.h.
// <number of buffers>
// for each buffer: [name] [type] [size] <data x size in hex> 
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t *val = static_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF)
        {
            uint16_t *val = static_cast<uint16_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

// We have the data files located in a specific directory. This 
// searches for that directory format from the current directory.
std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/char-rnn/", "data/char-rnn/"};
    return locateFile(input, dirs);
}
	
// TensorFlow weight parameters for BasicLSTMCell
// are formatted as:
// Each [WR][icfo] is hiddenSize sequential elements.
// CellN  Row 0: WiT, WcT, WfT, WoT
// CellN  Row 1: WiT, WcT, WfT, WoT
// ...
// CellN RowM-1: WiT, WcT, WfT, WoT
// CellN RowM+0: RiT, RcT, RfT, RoT
// CellN RowM+1: RiT, RcT, RfT, RoT
// ...
// CellNRow2M-1: RiT, RcT, RfT, RoT
//
// TensorRT expects the format to laid out in memory:
// CellN: Wf, Wi, Wc, Wo, Rf, Ri, Rc, Ro
Weights convertRNNWeights(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count));
    int indir[4]{ 1, 2, 0, 3 };
    int order[5]{ 0, 1, 4, 2, 3};
    int dims[5]{LAYER_COUNT, 2, 4, HIDDEN_SIZE, HIDDEN_SIZE};
    utils::reshapeWeights(input, dims, order, ptr, 5);
    utils::transposeSubBuffers(ptr, DataType::kFLOAT, LAYER_COUNT * 2, HIDDEN_SIZE * HIDDEN_SIZE, 4);
    int subMatrix = HIDDEN_SIZE * HIDDEN_SIZE;
    int layerOffset = 8 * subMatrix;
    for (int z = 0; z < LAYER_COUNT; ++z)
    {
        utils::reorderSubBuffers(ptr + z * layerOffset, indir, 4, subMatrix * sizeof(float));
        utils::reorderSubBuffers(ptr + z * layerOffset + 4 * subMatrix, indir, 4, subMatrix * sizeof(float));
    }
    return Weights{input.type, ptr, input.count};
}

// TensorFlow bias parameters for BasicLSTMCell
// are formatted as:
// CellN: Bi, Bc, Bf, Bo
//
// TensorRT expects the format to be:
// CellN: Wf, Wi, Wc, Wo, Rf, Ri, Rc, Ro
//
// Since tensorflow already combines U and W,
// we double the size and set all of U to zero.
Weights convertRNNBias(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count*2));
    std::fill(ptr, ptr + input.count*2, 0);
    const float* iptr = static_cast<const float*>(input.values);
    int indir[4]{ 1, 2, 0, 3 };
    for (int z = 0, y = 0; z < LAYER_COUNT; ++z)
        for (int x = 0; x < 4; ++x, ++y)
            std::copy(iptr + y * HIDDEN_SIZE , iptr + (y + 1) * HIDDEN_SIZE, ptr + (z * 8 + indir[x]) * HIDDEN_SIZE);
    return Weights{input.type, ptr, input.count*2};
}

// The fully connected weights from tensorflow are transposed compared to 
// the order that tensorRT expects them to be in.
Weights transposeFCWeights(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count));
    const float* iptr = static_cast<const float*>(input.values);
    assert(input.count == HIDDEN_SIZE * VOCAB_SIZE);
    for (int z = 0; z < HIDDEN_SIZE; ++z)
        for (int x = 0; x < VOCAB_SIZE; ++x)
            ptr[x * HIDDEN_SIZE + z] = iptr[z * VOCAB_SIZE + x];
    return Weights{input.type, ptr, input.count};
}

IRNNv2Layer * addRNNv2Layer(INetworkDefinition * network, std::map<std::string, Weights> &weightMap)
{
    // Initialize data, hiddenIn, and cellIn inputs into RNN Layer
    auto data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3(BATCH_SIZE, SEQ_SIZE, DATA_SIZE));
    assert(data != nullptr);

    auto hiddenIn = network->addInput(HIDDEN_IN_BLOB_NAME, DataType::kFLOAT, Dims3(BATCH_SIZE, LAYER_COUNT, HIDDEN_SIZE));
    assert(hiddenIn != nullptr);

    auto cellIn = network->addInput(CELL_IN_BLOB_NAME, DataType::kFLOAT, Dims3(BATCH_SIZE, LAYER_COUNT, HIDDEN_SIZE));
    assert(cellIn != nullptr);

    // create an RNN layer w/ 2 layers and 512 hidden states
    auto rnn = network->addRNNv2(*data, LAYER_COUNT, HIDDEN_SIZE, SEQ_SIZE, RNNOperation::kLSTM);
    assert(rnn != nullptr);

    // Set RNNv2 optional inputs
    rnn->getOutput(0)->setName("RNN output");
    rnn->setHiddenState(*hiddenIn);
    if (rnn->getOperation() == RNNOperation::kLSTM)
        rnn->setCellState(*cellIn);

    // convert tensorflow weight format to trt weight format
    auto tfwts = weightMap["rnnweight"];
    Weights rnnwts = convertRNNWeights(tfwts);
    auto tfbias = weightMap["rnnbias"];
    Weights rnnbias = convertRNNBias(tfbias);

    std::vector<nvinfer1::RNNGateType> gateOrder({  nvinfer1::RNNGateType::kFORGET, 
                                                    nvinfer1::RNNGateType::kINPUT, 
                                                    nvinfer1::RNNGateType::kCELL, 
                                                    nvinfer1::RNNGateType::kOUTPUT});
    const nvinfer1::DataType dataType = static_cast<nvinfer1::DataType>(rnnwts.type);
    const float * wts = static_cast<const float *>(rnnwts.values);
    const float * biases = static_cast<const float *>(rnnbias.values);
    size_t kernelOffset = 0, biasOffset = 0;
    for (int layerIndex = 0; layerIndex < LAYER_COUNT; layerIndex++)
    {
        for (int gateIndex = 0; gateIndex < NUM_GATES; gateIndex++)
        {           
            // extract weights and bias for a given gate and layer
            Weights gateWeight{.type = dataType,
                               .values = (void*)(wts + kernelOffset),
                               .count = DATA_SIZE * HIDDEN_SIZE};
            Weights gateBias{.type = dataType,
                             .values = (void*)(biases + biasOffset),
                             .count = HIDDEN_SIZE};
            
            // set weights and bias for given gate
            rnn->setWeightsForGate(layerIndex, gateOrder[gateIndex % 4], (gateIndex < 4), gateWeight);
            rnn->setBiasForGate(layerIndex, gateOrder[gateIndex % 4], (gateIndex < 4), gateBias);

            // Update offsets
            kernelOffset = kernelOffset + DATA_SIZE * HIDDEN_SIZE;
            biasOffset = biasOffset + HIDDEN_SIZE;
        }
    }

    // Store the transformed weights in the weight map so the memory can be properly released later.
    weightMap["rnnweight2"] = rnnwts;
    weightMap["rnnbias2"] = rnnbias;

    return rnn;
}

void APIToModel(std::map<std::string, Weights> &weightMap, IHostMemory **modelStream)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();
 
    // add RNNv2 layer and set its parameters
    auto rnn = addRNNv2Layer(network, weightMap);

    // Add a shuffle layer to reshape the ouput dims of RNN {N, S_max, H} to match the fully connected layer input dims {C, 1, 1}
    auto shuffle = network->addShuffle(*rnn->getOutput(0));
    assert(shuffle != nullptr);
    Dims rnnOutDims = rnn->getOutput(0)->getDimensions();
    assert(rnnOutDims.nbDims == 3);
    shuffle->setReshapeDimensions(Dims4(rnnOutDims.d[0] * rnnOutDims.d[1], rnnOutDims.d[2], 1, 1));
    shuffle->getOutput(0)->setName("Shuffle Output");

    // Add a second fully connected layer with 50 outputs.
    auto tffcwts = weightMap["rnnfcw"];
    auto fcwts = transposeFCWeights(tffcwts);
    auto bias = weightMap["rnnfcb"];
    auto fc = network->addFullyConnected(*shuffle->getOutput(0), VOCAB_SIZE, fcwts, bias);
    assert(fc != nullptr);
    fc->getOutput(0)->setName("FC output");
    weightMap["rnnfcw2"] = fcwts;

    // Add TopK layer to determine which character has highest probability.
    int reduceAxis = 0x2;  // reduce across vocab axis
    auto pred =  network->addTopK(*fc->getOutput(0), nvinfer1::TopKOperation::kMAX, 1, reduceAxis);
    assert(pred != nullptr);
    pred->getOutput(1)->setName(OUTPUT_BLOB_NAME);

    // Mark the outputs for the network
    network->markOutput(*pred->getOutput(1));
    pred->getOutput(1)->setType(DataType::kINT32);
    rnn->getOutput(1)->setName(HIDDEN_OUT_BLOB_NAME);
    network->markOutput(*rnn->getOutput(1));
    if (rnn->getOperation() == RNNOperation::kLSTM)
    {
        rnn->getOutput(2)->setName(CELL_OUT_BLOB_NAME);
        network->markOutput(*rnn->getOutput(2));
    }

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 25);
    auto engine = builder->buildCudaEngine(*network);
    assert(engine != nullptr);

    // serialize engine and clean up resources
    network->destroy();
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void allocateMemory(const ICudaEngine& engine, void** buffers, float** data, uint32_t*& output, int* indices)
{
    // Allocate memory for input, cell state, and hidden state
    for (int x = 0; x < NUM_BINDINGS - 1; ++x)
    {
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        indices[x] = engine.getBindingIndex(gNames[x]);
        assert(indices[x] < NUM_BINDINGS);
        if (indices[x] == -1) continue;

        // create GPU and CPU buffers and a stream
        CHECK(cudaMalloc(&buffers[indices[x]], gSizes[x] * sizeof(float)));
        data[x] = new float[gSizes[x]];
        std::fill(data[x], data[x] + gSizes[x], 0);
    }
    // Allocate memory for output
    indices[OUTPUT_IDX] = engine.getBindingIndex(gNames[OUTPUT_IDX]);
    assert(indices[OUTPUT_IDX] < NUM_BINDINGS);
    if (indices[OUTPUT_IDX] != -1)
    {
        CHECK(cudaMalloc(&buffers[indices[OUTPUT_IDX]], gSizes[OUTPUT_IDX] * sizeof(uint32_t)));
        output = new uint32_t[gSizes[OUTPUT_IDX]];
        std::fill(output, output + gSizes[OUTPUT_IDX], 0);
    }
}

void stepOnce(float **data, uint32_t * output, void **buffers, int *indices, cudaStream_t &stream, IExecutionContext &context)
{
    // DMA the input to the GPU
    CHECK(cudaMemcpyAsync(buffers[indices[INPUT_IDX]], data[INPUT_IDX], gSizes[INPUT_IDX] * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[indices[HIDDEN_IN_IDX]], data[HIDDEN_IN_IDX], gSizes[HIDDEN_IN_IDX] * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[indices[CELL_IN_IDX]], data[CELL_IN_IDX], gSizes[CELL_IN_IDX] * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Execute asynchronously
    context.enqueue(1, buffers, stream, nullptr);

    // DMA the output from the GPU
    CHECK(cudaMemcpyAsync(data[HIDDEN_OUT_IDX], buffers[indices[HIDDEN_OUT_IDX]], gSizes[HIDDEN_OUT_IDX] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(data[CELL_OUT_IDX], buffers[indices[CELL_OUT_IDX]], gSizes[CELL_OUT_IDX] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output, buffers[indices[OUTPUT_IDX]], gSizes[OUTPUT_IDX] * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
}

bool doInference(IExecutionContext& context, std::string input, std::string expected, std::map<std::string, Weights> &weightMap)
{
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == NUM_BINDINGS);
    void* buffers[NUM_BINDINGS];
    float* data[NUM_BINDINGS - 1];
    uint32_t* output = nullptr;
    int indices[NUM_BINDINGS];

    std::fill(buffers, buffers + NUM_BINDINGS, nullptr);
    std::fill(data, data + NUM_BINDINGS - 1, nullptr);
    std::fill(indices, indices + NUM_BINDINGS, -1);

    // allocate memory on host and device
    allocateMemory(engine, buffers, data, output, indices);

    // create stream for trt execution
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    auto embed = weightMap["embed"];
    std::string genstr;

    // Seed the RNN with the input.
    for (auto &a : input)
    {
        std::copy(static_cast<const float*>(embed.values) + char_to_id[a]*DATA_SIZE,
                static_cast<const float*>(embed.values) + char_to_id[a]*DATA_SIZE + DATA_SIZE,
                data[INPUT_IDX]);
        stepOnce(data, output, buffers, indices, stream, context);
        cudaStreamSynchronize(stream);

        // Copy Ct/Ht to the Ct-1/Ht-1 slots.
        std::memcpy(data[HIDDEN_IN_IDX], data[HIDDEN_OUT_IDX], gSizes[HIDDEN_IN_IDX] * sizeof(float));
        std::memcpy(data[CELL_IN_IDX], data[CELL_OUT_IDX], gSizes[CELL_IN_IDX] * sizeof(float));

        genstr.push_back(a);
    }

    // Generate predicted sequence of characters
    for (size_t x = 0, y = expected.size(); x < y; ++x)
    {
        std::copy(static_cast<const float*>(embed.values) + char_to_id[*genstr.rbegin()]*DATA_SIZE,
                static_cast<const float*>(embed.values) + char_to_id[*genstr.rbegin()]*DATA_SIZE + DATA_SIZE,
                data[INPUT_IDX]);

        stepOnce(data, output, buffers, indices, stream, context);
        cudaStreamSynchronize(stream);

        // Copy Ct/Ht to the Ct-1/Ht-1 slots.
        std::memcpy(data[HIDDEN_IN_IDX], data[HIDDEN_OUT_IDX], gSizes[HIDDEN_IN_IDX] * sizeof(float));
        std::memcpy(data[CELL_IN_IDX], data[CELL_OUT_IDX], gSizes[CELL_IN_IDX] * sizeof(float));

		uint32_t predIdx = *(output);
        genstr.push_back(id_to_char[predIdx]);
    }
    printf("Received: %s\n", genstr.c_str() + input.size());

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    for (int x = 0; x < NUM_BINDINGS - 1; ++x)
    {
        delete [] data[x];
        data[x] = nullptr;
    }
    CHECK(cudaFree(buffers[indices[OUTPUT_IDX]]));
    delete [] output;
    return genstr == (input + expected);
}

int main(int argc, char** argv)
{
    // BATCH_SIZE needs to equal one because the doInference() function
    // assumes that the batch size is one. To change this, one would need to add code to the 
    // doInference() function to seed BATCH_SIZE number of inputs and process the 
    // generation of BATCH_SIZE number of outputs. We leave this as an excercise for the user.
    assert(BATCH_SIZE == 1 && "This code assumes batch size is equal to 1.");

    // create a model using the API directly and serialize it to a stream
    IHostMemory *modelStream{nullptr};

    // Load weights and create model
    std::map<std::string, Weights> weightMap = loadWeights(locateFile("char-rnn.wts"));
    APIToModel(weightMap, &modelStream);

    // Input strings and their respective expected output strings
    const char* strings[10]{ "customer serv",
        "business plans",
        "help",
        "slightly under",
        "market",
        "holiday cards",
        "bring it",
        "what time",
        "the owner thinks",
        "money can be use"
    };
    const char* outs[10]{ "es and the",
        " to be a",
        "en and",
        "iting the company",
        "ing and",
        " the company",
        " company said it will",
        "d and the company",
        "ist with the",
        "d to be a"
    };

    // Select a random seed string.
    srand(unsigned(time(nullptr)));
    int num = rand() % 10;

    // Initialize engine, context, and other runtime resources
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
    if (modelStream) modelStream->destroy();
    IExecutionContext *context = engine->createExecutionContext();

    // Perform inference
    bool pass {false};
    std::cout << "\n---------------------------" << "\n";
    std::cout << "RNN Warmup: " << strings[num] << std::endl;
    std::cout << "Expect: " << outs[num] << std::endl;
    pass = doInference(*context, strings[num], outs[num], weightMap);
    if (!pass) std::cout << "Failure!" << std::endl;
    std::cout << "---------------------------" << "\n";

    // Clean up runtime resources
    for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return !pass ? EXIT_FAILURE : EXIT_SUCCESS;
}
