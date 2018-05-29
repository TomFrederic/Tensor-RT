#ifndef SAMPLE_NMT_LIKELIHOOD_
#define SAMPLE_NMT_LIKELIHOOD_

#include <memory>

#include "NvInfer.h"
#include "likelihood_combination_operator.h"
#include "component.h"

namespace nmtSample
{
    /** \class Likelihood
    *
    * \brief calculates likelihood and TopK indices for the raw input logits
    *
    */
    class Likelihood : public Component
    {
    public:
        typedef std::shared_ptr<Likelihood> ptr;

        Likelihood() = default;

        virtual LikelihoodCombinationOperator::ptr getLikelihoodCombinationOperator() const = 0;

        /**
        * \brief add calculation of likelihood and TopK indices to the network
        */
        virtual void addToModel(
            nvinfer1::INetworkDefinition * network,
            int beamWidth,
            nvinfer1::ITensor * inputLogits,
            nvinfer1::ITensor ** newLikelihoods,
            nvinfer1::ITensor ** newVocabularyIndices) = 0;

        virtual ~Likelihood() = default;
    };
}

#endif // SAMPLE_NMT_LIKELIHOOD_
