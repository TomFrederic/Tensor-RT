#ifndef SAMPLE_NMT_SLP_PROJECTION_
#define SAMPLE_NMT_SLP_PROJECTION_

#include "projection.h"

#include "component_weights.h"

namespace nmtSample
{
    /** \class SLPProjection
    *
    * \brief Linear logits calculation
    *
    * Calculates logits vector by multiplying input vector with weight matrix  
    *
    */
    class SLPProjection : public Projection
    {
    public:
        SLPProjection(ComponentWeights::ptr weights);

        virtual void addToModel(
            nvinfer1::INetworkDefinition * network,
            nvinfer1::ITensor * input,
            nvinfer1::ITensor ** outputLogits);

        virtual int getOutputSize();

        virtual std::string getInfo();

        virtual ~SLPProjection() = default;

    protected:
        ComponentWeights::ptr mWeights;
        nvinfer1::Weights mKernelWeights;
        int mInputChannelCount;
        int mOutputChannelCount;
    };
}

#endif // SAMPLE_NMT_SLP_PROJECTION_
