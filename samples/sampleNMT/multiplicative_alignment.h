#ifndef SAMPLE_NMT_MULTIPLICATIVE_ALIGNMENT_
#define SAMPLE_NMT_MULTIPLICATIVE_ALIGNMENT_

#include "alignment.h"

#include "component_weights.h"

namespace nmtSample
{
    /** \class MultiplicativeAlignment
    *
    * \brief alignment scores from Luong attention mechanism 
    *
    */
    class MultiplicativeAlignment : public Alignment
    {
    public:
        MultiplicativeAlignment(ComponentWeights::ptr weights);

        virtual void addToModel(
            nvinfer1::INetworkDefinition * network,
            nvinfer1::ITensor * attentionKeys,
            nvinfer1::ITensor * queryStates,
            nvinfer1::ITensor ** alignmentScores);

        virtual void addAttentionKeys(
            nvinfer1::INetworkDefinition * network,
            nvinfer1::ITensor * memoryStates,
            nvinfer1::ITensor ** attentionKeys);

        virtual int getSourceStatesSize();

        virtual int getAttentionKeySize();

        virtual std::string getInfo();

        virtual ~MultiplicativeAlignment() = default;

    protected:
        ComponentWeights::ptr mWeights;
        nvinfer1::Weights mKernelWeights;
        int mInputChannelCount;
        int mOutputChannelCount;
    };
}

#endif // SAMPLE_NMT_MULTIPLICATIVE_ALIGNMENT_
