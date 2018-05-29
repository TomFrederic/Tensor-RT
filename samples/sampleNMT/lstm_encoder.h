#ifndef SAMPLE_NMT_LSTM_ENCODER_
#define SAMPLE_NMT_LSTM_ENCODER_

#include "encoder.h"

#include "component_weights.h"

namespace nmtSample
{
    /** \class LSTMEncoder
    *
    * \brief encodes input sentences into output states using LSTM
    *
    */
    class LSTMEncoder : public Encoder
    {
    public:
        LSTMEncoder(ComponentWeights::ptr weights);

        virtual void addToModel(
            nvinfer1::INetworkDefinition * network,
            int maxInputSequenceLength,
            nvinfer1::ITensor * inputEmbeddedData,
            nvinfer1::ITensor * actualInputSequenceLengths,
            nvinfer1::ITensor ** inputStates,
            nvinfer1::ITensor ** memoryStates,
            nvinfer1::ITensor ** lastTimestepStates);

        virtual int getMemoryStatesSize();

        virtual std::vector<int> getStateSizes();

        virtual std::string getInfo();

        virtual ~LSTMEncoder() = default;

    protected:
        ComponentWeights::ptr mWeights;
        std::vector<nvinfer1::Weights> mGateKernelWeights;
        std::vector<nvinfer1::Weights> mGateBiasWeights;
        bool mRNNKind;
        int mNumLayers;
        int mNumUnits;
    };
}

#endif // SAMPLE_NMT_LSTM_ENCODER_
