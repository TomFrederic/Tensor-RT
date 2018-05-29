#ifndef SAMPLE_NMT_LSTM_DECODER_
#define SAMPLE_NMT_LSTM_DECODER_

#include "decoder.h"

#include "component_weights.h"

namespace nmtSample
{
    /** \class LSTMDecoder
    *
    * \brief encodes single input into output states with LSTM
    *
    */
    class LSTMDecoder : public Decoder
    {
    public:
        LSTMDecoder(ComponentWeights::ptr weights);

        virtual void addToModel(
            nvinfer1::INetworkDefinition * network,
            nvinfer1::ITensor * inputEmbeddedData,
            nvinfer1::ITensor ** inputStates,
            nvinfer1::ITensor ** outputData,
            nvinfer1::ITensor ** outputStates);

        virtual std::vector<int> getStateSizes();

        virtual std::string getInfo();

        virtual ~LSTMDecoder() = default;

    protected:
        ComponentWeights::ptr mWeights;
        std::vector<nvinfer1::Weights> mGateKernelWeights;
        std::vector<nvinfer1::Weights> mGateBiasWeights;
        bool mRNNKind;
        int mNumLayers;
        int mNumUnits;
    };
}

#endif // SAMPLE_NMT_LSTM_DECODER_
