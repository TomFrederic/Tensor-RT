#include "lstm_encoder.h"
#include "trt_util.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{

    LSTMEncoder::LSTMEncoder(ComponentWeights::ptr weights)
        : mWeights(weights)
    {
        // please refer to chpt_to_bin.py for the details on the format
        assert(mWeights->mMetaData.size() >= 4);
        const nvinfer1::DataType dataType = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
        assert(dataType == nvinfer1::DataType::kFLOAT);
        mRNNKind = mWeights->mMetaData[1];
        mNumLayers = mWeights->mMetaData[2];
        mNumUnits = mWeights->mMetaData[3];

        size_t elementSize = inferTypeToBytes(dataType);
        // compute weights offsets
        size_t kernelOffset = 0;
        size_t biasStartOffset = ((4 * mNumUnits + 4 * mNumUnits) * mNumUnits * mNumLayers) * elementSize;
        size_t biasOffset = biasStartOffset;
        int numGates = 8;
        for (int layerIndex = 0; layerIndex < mNumLayers; layerIndex++)
        {
            for (int gateIndex = 0; gateIndex < numGates; gateIndex++)
            {           
                // encoder input size == mNumUnits
                int64_t inputSize = ((layerIndex == 0) && (gateIndex < 4))? mNumUnits : mNumUnits;
                nvinfer1::Weights gateKernelWeights { .type = dataType
                                                    , .values = (void*)(&mWeights->mWeights[0] + kernelOffset)
                                                    , .count = inputSize * mNumUnits };
                nvinfer1::Weights gateBiasWeights   { .type = dataType
                                                    , .values = (void*)(&mWeights->mWeights[0] + biasOffset)
                                                    , .count = mNumUnits };
                mGateKernelWeights.push_back(std::move(gateKernelWeights));
                mGateBiasWeights.push_back(std::move(gateBiasWeights));
                kernelOffset = kernelOffset + inputSize * mNumUnits * elementSize;
                biasOffset = biasOffset + mNumUnits * elementSize;
            }
        }
        assert(kernelOffset + biasOffset - biasStartOffset == mWeights->mWeights.size());
    }

    void LSTMEncoder::addToModel(
        nvinfer1::INetworkDefinition * network,
        int maxInputSequenceLength,
        nvinfer1::ITensor * inputEmbeddedData,
        nvinfer1::ITensor * actualInputSequenceLengths,
        nvinfer1::ITensor ** inputStates,
        nvinfer1::ITensor ** memoryStates,
        nvinfer1::ITensor ** lastTimestepStates)
    {
        nvinfer1::ITensor * shuffledInput;
        {
            auto shuffleLayer = network->addShuffle(*inputEmbeddedData);
            assert(shuffleLayer != nullptr);
            nvinfer1::Dims shuffleDims{3, {1, maxInputSequenceLength, mNumUnits}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kSEQUENCE, nvinfer1::DimensionType::kCHANNEL}};
            shuffleLayer->setReshapeDimensions(shuffleDims);
            shuffledInput = shuffleLayer->getOutput(0);
            assert(shuffledInput != nullptr);
        }
 
        nvinfer1::ITensor * shuffledHiddenState;
        {
            auto shuffleLayer = network->addShuffle(*inputStates[0]);
            assert(shuffleLayer != nullptr);
            nvinfer1::Dims shuffleDims{3, {1, mNumLayers, mNumUnits}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kSPATIAL, nvinfer1::DimensionType::kCHANNEL}};
            shuffleLayer->setReshapeDimensions(shuffleDims);
            shuffledHiddenState = shuffleLayer->getOutput(0);
            assert(shuffledHiddenState != nullptr);
        }

        nvinfer1::ITensor * shuffledCellState;
        {
            auto shuffleLayer = network->addShuffle(*inputStates[1]);
            assert(shuffleLayer != nullptr);
            nvinfer1::Dims shuffleDims{3, {1, mNumLayers, mNumUnits}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kSPATIAL, nvinfer1::DimensionType::kCHANNEL}};
            shuffleLayer->setReshapeDimensions(shuffleDims);
            shuffledCellState = shuffleLayer->getOutput(0);
            assert(shuffledCellState != nullptr);
        }

        auto encoderLayer = network->addRNNv2(
            *shuffledInput,
            mNumLayers,
            mNumUnits,
            maxInputSequenceLength,
            nvinfer1::RNNOperation::kLSTM);

        assert(encoderLayer != nullptr);

        encoderLayer->setSequenceLengths(*actualInputSequenceLengths);
        encoderLayer->setInputMode(nvinfer1::RNNInputMode::kLINEAR);
        encoderLayer->setDirection(nvinfer1::RNNDirection::kUNIDIRECTION);

        std::vector<nvinfer1::RNNGateType> gateOrder({  nvinfer1::RNNGateType::kFORGET, 
                                                        nvinfer1::RNNGateType::kINPUT, 
                                                        nvinfer1::RNNGateType::kCELL, 
                                                        nvinfer1::RNNGateType::kOUTPUT});
        for (size_t i = 0; i < mGateKernelWeights.size(); i++)
        {
            // we have 4 + 4 gates
            bool isW = ((i % 8) < 4);
            encoderLayer->setWeightsForGate(i / 8, gateOrder[i % 4], isW, mGateKernelWeights[i]);
            encoderLayer->setBiasForGate(i / 8, gateOrder[i % 4], isW, mGateBiasWeights[i]);
        }

        encoderLayer->setHiddenState(*shuffledHiddenState);
        encoderLayer->setCellState(*shuffledCellState);
        *memoryStates = encoderLayer->getOutput(0);
        assert(*memoryStates != nullptr);

        {
            auto shuffleLayer = network->addShuffle(**memoryStates);
            assert(shuffleLayer != nullptr);
            nvinfer1::Dims shuffleDims{2, {maxInputSequenceLength, mNumUnits}, {nvinfer1::DimensionType::kSEQUENCE, nvinfer1::DimensionType::kCHANNEL}};
            shuffleLayer->setReshapeDimensions(shuffleDims);
            auto shuffledOutput = shuffleLayer->getOutput(0);
            assert(shuffledOutput != nullptr);
            *memoryStates = shuffledOutput;
        }

        if (lastTimestepStates)
        {
            // Per layer hidden output
            lastTimestepStates[0] = encoderLayer->getOutput(1);
            assert(lastTimestepStates[0] != nullptr);
            
            {
                auto shuffleLayer = network->addShuffle(*lastTimestepStates[0]);
                assert(shuffleLayer != nullptr);
                nvinfer1::Dims shuffleDims{1, {mNumUnits * mNumLayers}, {nvinfer1::DimensionType::kCHANNEL}};
                shuffleLayer->setReshapeDimensions(shuffleDims);
                auto shuffledOutput = shuffleLayer->getOutput(0);
                assert(shuffledOutput != nullptr);
                lastTimestepStates[0] = shuffledOutput;
            }

            // Per layer cell output
            lastTimestepStates[1] = encoderLayer->getOutput(2);
            assert(lastTimestepStates[1] != nullptr);

            {
                auto shuffleLayer = network->addShuffle(*lastTimestepStates[1]);
                assert(shuffleLayer != nullptr);
                nvinfer1::Dims shuffleDims{1, {mNumUnits * mNumLayers}, {nvinfer1::DimensionType::kCHANNEL}};
                shuffleLayer->setReshapeDimensions(shuffleDims);
                auto shuffledOutput = shuffleLayer->getOutput(0);
                assert(shuffledOutput != nullptr);
                lastTimestepStates[1] = shuffledOutput;
            }
        }
    }

    int LSTMEncoder::getMemoryStatesSize()
    {
        return mNumUnits;
    }

    std::vector<int> LSTMEncoder::getStateSizes()
    {
        return std::vector<int>({mNumLayers * mNumUnits, mNumLayers * mNumUnits});
    }

    std::string LSTMEncoder::getInfo()
    {
        std::stringstream ss;
        ss << "LSTM Encoder, num layers = " << mNumLayers << ", num units = " << mNumUnits;
        return ss.str();
    }
}
